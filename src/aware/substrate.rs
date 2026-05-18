use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{VarBuilder, VarMap};

use super::cap::CapMatrix;
use super::concept_layer::{ConceptConfig, ConceptLayer};
use super::config::{BlockConfig, CapConfig, Config};
use super::discover::{make_discovery, DiscoveryCtx, DiscoveryKind};
use super::embed::{build_causal_mask, RoPE, TokenEmbedding};
use super::layer::{Block, CapLayer};
use super::norm::RmsNorm;

/// Per-layer cap statistics for benchmark reporting.
#[derive(Debug, Clone)]
pub struct LayerCapStats {
    pub n_caps: usize,
    pub n_frozen: usize,
    pub n_dormant: usize,
    pub kind: String,
    pub trainable: bool,
}

/// Aggregate cap statistics for a substrate.
#[derive(Debug, Clone, Default)]
pub struct CapStats {
    pub input_layer: Option<LayerCapStats>,
    pub shared: Option<LayerCapStats>,
}

/// Context passed during block construction so blocks can see substrate-level
/// shared resources (e.g., a global CapMatrix for Config 1).
pub struct BlockBuildCtx<'a> {
    pub shared_cap_keys: Option<&'a Tensor>,
    pub shared_cap_values: Option<&'a Tensor>,
}

impl<'a> BlockBuildCtx<'a> {
    pub fn empty() -> Self {
        Self {
            shared_cap_keys: None,
            shared_cap_values: None,
        }
    }
}

pub struct Substrate {
    pub cfg: Config,
    pub embed: TokenEmbedding,
    pub cap_layer: Option<CapLayer>,
    pub blocks: Vec<Block>,
    pub final_ln: RmsNorm,
    pub rope: RoPE,
    pub causal_mask: Tensor,
    pub device: Device,
    pub varmap: VarMap,
    /// Substrate-level CapMatrix shared by all CapMemory/CapPair attention
    /// layers that declared `CapMatrixSource::Shared`. None if no block
    /// asked for sharing.
    pub shared_cap_matrix: Option<CapMatrix>,
    /// Optional concept layer (window=1 caps with sparse top-K activation,
    /// values added to final_hidden before unembed). Enables concept-
    /// enriched prediction with stable token-level concept identities.
    pub concept_layer: Option<ConceptLayer>,
}

impl Substrate {
    /// Forward: token ids [B, T] -> logits [B, T, vocab].
    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let embeds = self.embed.forward(tokens)?;
        let mut x = embeds.clone();
        if let Some(cl) = &self.cap_layer {
            x = cl.forward(&x)?;
        }
        for block in &self.blocks {
            x = block.forward(&x, &self.rope, &self.causal_mask)?;
        }
        let mut final_hidden = self.final_ln.forward(&x)?;

        // Concept layer enrichment: if present, add concept contribution
        // to final_hidden before unembed.
        if let Some(cl) = &self.concept_layer {
            let (contrib, _acts) = cl.forward(&embeds)?;
            final_hidden = (final_hidden + contrib)?;
        }

        // Tied output: logits = final_hidden @ embedding_table.T
        let emb_w = self.embed.weight();
        let emb_w_t = emb_w.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        final_hidden.broadcast_matmul(&emb_w_t)
    }

    /// Forward and also return concept activations for probing.
    pub fn forward_with_concepts(&self, tokens: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let embeds = self.embed.forward(tokens)?;
        let mut x = embeds.clone();
        if let Some(cl) = &self.cap_layer {
            x = cl.forward(&x)?;
        }
        for block in &self.blocks {
            x = block.forward(&x, &self.rope, &self.causal_mask)?;
        }
        let mut final_hidden = self.final_ln.forward(&x)?;
        let concept_acts = if let Some(cl) = &self.concept_layer {
            let (contrib, acts) = cl.forward(&embeds)?;
            final_hidden = (final_hidden + contrib)?;
            Some(acts)
        } else {
            None
        };
        let emb_w = self.embed.weight();
        let emb_w_t = emb_w.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let logits = final_hidden.broadcast_matmul(&emb_w_t)?;
        Ok((logits, concept_acts))
    }

    /// Hidden state probe (for concept discovery). Returns residual stream
    /// AFTER block `probe_layer`.
    pub fn forward_hidden(&self, tokens: &Tensor, probe_layer: usize) -> Result<Tensor> {
        let target = probe_layer.min(self.blocks.len().saturating_sub(1));
        let mut x = self.embed.forward(tokens)?;
        if let Some(cl) = &self.cap_layer {
            x = cl.forward(&x)?;
        }
        for (li, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, &self.rope, &self.causal_mask)?;
            if li == target {
                return Ok(x);
            }
        }
        Ok(x)
    }

    pub fn n_params(&self) -> usize {
        self.varmap
            .all_vars()
            .iter()
            .map(|v| v.as_tensor().elem_count())
            .sum()
    }

    /// Per-substrate cap statistics, useful for benchmark reports.
    pub fn cap_stats(&self) -> CapStats {
        let cl_stats = self.cap_layer.as_ref().map(|cl| LayerCapStats {
            n_caps: cl.caps.n_caps(),
            n_frozen: cl.caps.metadata.iter().filter(|m| m.frozen).count(),
            n_dormant: cl.caps.metadata.iter().filter(|m| m.dormant).count(),
            kind: format!("{:?}", cl.caps.kind),
            trainable: cl.caps.trainable,
        });
        let shared_stats = self.shared_cap_matrix.as_ref().map(|cm| LayerCapStats {
            n_caps: cm.n_caps(),
            n_frozen: cm.metadata.iter().filter(|m| m.frozen).count(),
            n_dormant: cm.metadata.iter().filter(|m| m.dormant).count(),
            kind: format!("{:?}", cm.kind),
            trainable: cm.trainable,
        });
        CapStats {
            input_layer: cl_stats,
            shared: shared_stats,
        }
    }

    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        self.varmap.save(path)
    }

    pub fn load_checkpoint(&mut self, path: &str) -> Result<()> {
        self.varmap.load(path)
    }

    pub fn builder() -> SubstrateBuilder {
        SubstrateBuilder::default()
    }
}

/// Fluent builder for Substrate.
pub struct SubstrateBuilder {
    pub vcfg: Config,
    pub device: Device,
    pub include_token_embedding: bool,
    pub cap_config: Option<CapConfig>,
    pub blocks: Vec<BlockConfig>,
    /// If set, build a substrate-level shared CapMatrix with this many caps.
    pub shared_cap_n_caps: Option<usize>,
    /// d_v for the shared cap matrix's values; if None, defaults to d_model.
    pub shared_cap_d_v: Option<usize>,
    /// Token IDs sampled from the training corpus, used to bootstrap the cap
    /// layer via real KMeans (or other data-driven discovery). The builder
    /// looks them up in the freshly-initialized embedding table and passes
    /// the resulting [N, d_emb] tensor as DiscoveryCtx.sample.
    /// If None, discovery falls back to random init.
    pub bootstrap_sample_tokens: Option<Vec<u32>>,
    /// Optional concept layer configuration. If set, the built substrate
    /// includes a parallel concept-cap layer with sparse top-K activation
    /// that enriches final_hidden before unembed.
    pub concept_config: Option<ConceptConfig>,
}

impl Default for SubstrateBuilder {
    fn default() -> Self {
        Self {
            vcfg: Config::default(),
            device: Device::Cpu,
            include_token_embedding: true,
            cap_config: None,
            blocks: Vec::new(),
            shared_cap_n_caps: None,
            shared_cap_d_v: None,
            bootstrap_sample_tokens: None,
            concept_config: None,
        }
    }
}

impl SubstrateBuilder {
    pub fn with_vocab(mut self, vocab_size: usize) -> Self {
        self.vcfg.vocab_size = vocab_size;
        self
    }
    pub fn with_d_model(mut self, d_model: usize) -> Self {
        self.vcfg.d_model = d_model;
        self
    }
    pub fn with_max_seq_len(mut self, t: usize) -> Self {
        self.vcfg.max_seq_len = t;
        self
    }
    pub fn with_device(mut self, d: Device) -> Self {
        self.device = d;
        self
    }
    pub fn with_token_embedding(mut self) -> Self {
        self.include_token_embedding = true;
        self
    }
    pub fn with_cap_layer(mut self, cfg: CapConfig) -> Self {
        self.cap_config = Some(cfg);
        self
    }

    pub fn with_block(mut self, bcfg: BlockConfig) -> Self {
        self.blocks.push(bcfg);
        self
    }

    pub fn with_block_repeated(mut self, n: usize, bcfg: BlockConfig) -> Self {
        for _ in 0..n {
            self.blocks.push(bcfg.clone());
        }
        self
    }

    /// Declare a substrate-level shared CapMatrix. Any block configured with
    /// `AttentionKind::CapMemory { source: Shared }` (or `CapPair { Shared }`)
    /// will reference this matrix instead of constructing its own. The
    /// matrix's keys and values are gradient-trainable; gradients from every
    /// block accumulate into the shared tensors automatically.
    pub fn with_shared_cap_matrix(mut self, n_caps: usize) -> Self {
        self.shared_cap_n_caps = Some(n_caps);
        self
    }

    pub fn with_shared_cap_d_v(mut self, d_v: usize) -> Self {
        self.shared_cap_d_v = Some(d_v);
        self
    }

    /// Provide a token sample for cap-layer discovery bootstrap.
    /// Recommended: 1000-5000 random tokens from your training corpus.
    /// The builder embeds them via the (random-init) embedding table and
    /// passes the resulting d_emb-dim vectors to the Discovery::bootstrap
    /// step. KMeansDiscovery then runs real k-means clustering on those
    /// vectors instead of falling back to random init.
    pub fn with_bootstrap_sample_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.bootstrap_sample_tokens = Some(tokens);
        self
    }

    /// Add a concept layer (window=1 caps with sparse top-K activation,
    /// value enrichment of final_hidden before unembed).
    pub fn with_concept_layer(mut self, cfg: ConceptConfig) -> Self {
        self.concept_config = Some(cfg);
        self
    }

    pub fn build(self) -> Result<Substrate> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &self.device);

        let embed = TokenEmbedding::new(&self.vcfg, vb.pp("embed"))?;

        // If bootstrap sample tokens were provided, embed them and shape
        // for the cap layer's d_in.
        // - window=1: each token is a sample -> [N, d_emb]
        // - window=K>1: tokens come as N×K consecutive windows; we embed
        //   each token and concatenate per-window -> [N, K*d_emb]
        let cap_window: usize = self
            .cap_config
            .as_ref()
            .map(|c| c.cap_window.max(1))
            .unwrap_or(1);
        let bootstrap_sample: Option<Tensor> = if let Some(toks) = &self.bootstrap_sample_tokens {
            let n_total = toks.len();
            if n_total == 0 {
                None
            } else if cap_window > 1 && n_total % cap_window == 0 {
                // Treat toks as a flat sequence of N windows × K tokens each
                let n_windows = n_total / cap_window;
                let tok_tensor = Tensor::from_vec(toks.clone(), (n_total,), &self.device)?
                    .to_dtype(DType::U32)?;
                let embs = embed.forward(&tok_tensor)?; // [N*K, d_emb]
                let d_emb = embs.dim(1)?;
                let win = embs.reshape((n_windows, cap_window, d_emb))?;
                Some(win.reshape((n_windows, cap_window * d_emb))?) // [N, K*d_emb]
            } else {
                // window=1 (or non-divisible - treat as per-token samples)
                let tok_tensor = Tensor::from_vec(toks.clone(), (n_total,), &self.device)?
                    .to_dtype(DType::U32)?;
                Some(embed.forward(&tok_tensor)?) // [N, d_emb]
            }
        } else {
            None
        };

        let cap_layer = if let Some(cap_cfg) = self.cap_config.clone() {
            Some(CapLayer::new(
                cap_cfg,
                self.vcfg.d_model,
                self.vcfg.d_model,
                &self.device,
                bootstrap_sample.as_ref(),
                vb.pp("cap_layer"),
            )?)
        } else {
            None
        };

        // Build substrate-level shared CapMatrix BEFORE blocks if requested,
        // so we can pass references to it through Block::new.
        let shared_cap_matrix: Option<CapMatrix> = if let Some(n_caps) = self.shared_cap_n_caps {
            let d_v = self.shared_cap_d_v.unwrap_or(self.vcfg.d_model);
            let disc = make_discovery(DiscoveryKind::NoDiscovery);
            let ctx = DiscoveryCtx {
                device: &self.device,
                d_in: self.vcfg.d_model,
                d_out: Some(d_v),
                n_caps_target: n_caps,
                sample: None,
                audit: &super::config::AuditConfig::default(),
                training_step: 0,
            };
            Some(disc.bootstrap(&ctx)?)
        } else {
            None
        };

        // Build a context that exposes the shared cap matrix's tensors.
        // Tensors are Arc-wrapped internally; cloning is cheap and gradients
        // accumulate into the shared underlying storage.
        let shared_keys_clone = shared_cap_matrix.as_ref().map(|m| m.keys.clone());
        let shared_values_clone = shared_cap_matrix
            .as_ref()
            .and_then(|m| m.values.as_ref().cloned());
        let ctx = BlockBuildCtx {
            shared_cap_keys: shared_keys_clone.as_ref(),
            shared_cap_values: shared_values_clone.as_ref(),
        };

        let mut blocks = Vec::with_capacity(self.blocks.len());
        for (i, bcfg) in self.blocks.into_iter().enumerate() {
            let block = Block::new(
                &self.vcfg,
                &bcfg,
                &self.device,
                &ctx,
                vb.pp(&format!("blocks.{}", i)),
            )?;
            blocks.push(block);
        }

        let final_ln = RmsNorm::new(self.vcfg.d_model, self.vcfg.norm_eps, vb.pp("final_ln"))?;
        let d_k = self.vcfg.d_model / 4; // default; will be overridden by attention's actual d_k
        let rope = RoPE::new(
            d_k,
            self.vcfg.max_seq_len,
            self.vcfg.rope_base,
            &self.device,
        )?;
        let causal_mask = build_causal_mask(self.vcfg.max_seq_len, &self.device)?;

        // Concept layer (optional): window=1 caps discovered from embedding
        // samples (the bootstrap_sample we already prepared). Concept keys
        // are frozen; values are gradient-trained.
        let concept_layer = if let Some(ccfg) = self.concept_config {
            // For concept layer we want per-token (window=1) sample, not
            // windowed. If the cap layer used windowed samples, we re-
            // embed the raw tokens here.
            let single_token_sample: Option<Tensor> =
                if let Some(toks) = &self.bootstrap_sample_tokens {
                    let n_total = toks.len();
                    let cap_window = self
                        .cap_config
                        .as_ref()
                        .map(|c| c.cap_window.max(1))
                        .unwrap_or(1);
                    if n_total == 0 {
                        None
                    } else if cap_window > 1 {
                        // Sample was originally [N_windows*K, d_emb]. For
                        // concept-layer (window=1), we use just the first token
                        // of each window - those are still per-token samples.
                        let n_windows = n_total / cap_window;
                        let first_tokens: Vec<u32> =
                            (0..n_windows).map(|i| toks[i * cap_window]).collect();
                        let tt = Tensor::from_vec(first_tokens, (n_windows,), &self.device)?
                            .to_dtype(DType::U32)?;
                        Some(embed.forward(&tt)?)
                    } else {
                        // Already per-token. Reuse the bootstrap_sample we
                        // computed for the cap layer (it has the same shape).
                        bootstrap_sample.clone()
                    }
                } else {
                    None
                };

            Some(ConceptLayer::new(
                ccfg,
                self.vcfg.d_model,
                self.vcfg.d_model,
                &self.device,
                single_token_sample.as_ref(),
                vb.pp("concept_layer"),
            )?)
        } else {
            None
        };

        Ok(Substrate {
            cfg: self.vcfg,
            embed,
            cap_layer,
            blocks,
            final_ln,
            rope,
            causal_mask,
            device: self.device,
            varmap,
            shared_cap_matrix,
            concept_layer,
        })
    }
}

/// Builder helper for blocks. Returns a configured BlockConfig.
pub struct BlockBuilder {
    pub config: BlockConfig,
}

impl BlockBuilder {
    pub fn new() -> Self {
        Self {
            config: BlockConfig::default(),
        }
    }

    pub fn with_attention(mut self, kind: super::attention::AttentionKind) -> Self {
        self.config.attention = kind;
        self
    }

    pub fn with_heads(mut self, n: usize) -> Self {
        self.config.n_heads = n;
        self
    }

    pub fn with_ffn(mut self, d_ff: usize) -> Self {
        let mut f = self.config.ffn.unwrap_or_default();
        f.d_ff = d_ff;
        self.config.ffn = Some(f);
        self
    }

    pub fn without_ffn(mut self) -> Self {
        self.config.ffn = None;
        self
    }

    pub fn build(self) -> BlockConfig {
        self.config
    }
}

impl Default for BlockBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Convenience: SubstrateBuilder accepts BlockBuilder OR BlockConfig.
impl SubstrateBuilder {
    pub fn add_block_via(self, builder: BlockBuilder) -> Self {
        self.with_block(builder.build())
    }
}
