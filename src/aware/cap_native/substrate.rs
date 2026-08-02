use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Module, Result as CResult, Tensor, Var};
use candle_nn::{loss::cross_entropy, Embedding, VarBuilder, VarMap};

use super::super::layer::cap_layer::CapLayer;
use super::block::{CapNativeBlock, CapNativeBlockConfig};
use super::compression::RoutingMode;
use super::config::CapNativeConfig;
use super::norm::CapKeyedRmsNorm;
use super::sparse_routing::routing_from_cap_acts;
use super::output::CapKeyedOutput;

pub struct CapNativeSubstrate {
    pub config: CapNativeConfig,
    pub varmap: Arc<Mutex<VarMap>>,
    pub device: Device,
    pub dtype: DType,
    pub embeddings: Embedding,
    /// Layer 0: CapLayer (discovered CapMatrix + windowed firing + projection).
    pub cap_layer: CapLayer,
    /// Layer 1: optional second CapLayer for hierarchical mode (Phase E).
    /// Discovered over h_0 activations rather than token embeddings.
    /// When present, downstream cap-keyed components route by its
    /// activations (sized to `cap_layer_1.n_caps`).
    pub cap_layer_1: Option<CapLayer>,
    pub blocks: Vec<CapNativeBlock>,
    pub final_norm: CapKeyedRmsNorm,
    pub output: CapKeyedOutput,
}

impl CapNativeSubstrate {
    pub fn builder() -> CapNativeBuilder {
        CapNativeBuilder::default()
    }

    /// Forward: token_ids (B, S) -> logits (B, S, vocab).
    pub fn forward(&self, token_ids: &Tensor) -> CResult<Tensor> {
        let emb = self.embeddings.forward(token_ids)?;
        // Layer 0: discovered cap layer produces both the d_model input signal
        // and the cap_acts routing signal.
        let h0 = self.cap_layer.forward(&emb)?; // (B, S, d_model)

        // In hierarchical mode the routing signal and the downstream
        // input come from layer 1 (discovered over h_0). In single-
        // discovery mode they come from layer 0.
        let (h, cap_acts) = match &self.cap_layer_1 {
            Some(layer1) => {
                let h1 = layer1.forward(&h0)?; // (B, S, d_model)
                let cap_acts_1 = layer1.cap_activations(&h0)?; // (B, S, n_caps_1)
                (h1, cap_acts_1)
            }
            None => {
                let cap_acts_0 = self.cap_layer.cap_activations(&emb)?; // (B, S, n_caps_0)
                (h0, cap_acts_0)
            }
        };

        // Every cap-keyed projection in the substrate routes by this same
        // `cap_acts` tensor, so the routing decision is derived once here
        // and shared. Deriving it per projection returns an identical
        // structure but costs a device synchronisation each time.
        let routing = match self.config.routing {
            RoutingMode::HardTop1Sparse => Some(routing_from_cap_acts(
                &cap_acts,
                self.config.downstream_n_caps(),
                &self.device,
            )?),
            RoutingMode::SoftTopK => None,
        };

        let mut h = h;
        for block in &self.blocks {
            h = block.forward_with_routing(&h, &cap_acts, routing.as_ref())?;
        }
        let h = self.final_norm.forward(&h, &cap_acts)?;
        self.output.forward_with_routing(&h, &cap_acts, routing.as_ref())
    }

    /// Cross-entropy loss for next-token prediction.
    pub fn loss(&self, tokens: &Tensor) -> CResult<Tensor> {
        let dims = tokens.dims();
        let (b, s_plus1) = (dims[0], dims[1]);
        if s_plus1 < 2 {
            return Tensor::zeros((), DType::F32, &self.device);
        }
        let s = s_plus1 - 1;
        let inputs = tokens.narrow(1, 0, s)?;
        let targets = tokens.narrow(1, 1, s)?;
        let logits = self.forward(&inputs)?;
        let logits = logits.reshape((b * s, self.config.vocab))?;
        let targets = targets.reshape((b * s,))?;
        cross_entropy(&logits, &targets)
    }

    pub fn n_params(&self) -> usize {
        let trainable: usize = self
            .varmap
            .lock()
            .unwrap()
            .all_vars()
            .iter()
            .map(|v| v.elem_count())
            .sum();
        // CapLayer's CapMatrix keys are frozen (not in varmap when discovered).
        let mut frozen: usize = self.cap_layer.caps.keys.elem_count()
            + self
                .cap_layer
                .caps
                .values
                .as_ref()
                .map(|v| v.elem_count())
                .unwrap_or(0);
        if let Some(layer1) = &self.cap_layer_1 {
            frozen += layer1.caps.keys.elem_count()
                + layer1
                    .caps
                    .values
                    .as_ref()
                    .map(|v| v.elem_count())
                    .unwrap_or(0);
        }
        trainable + frozen
    }
}

// ──────────────────────────────────────────────────────────────────────
// Factory builder
// ──────────────────────────────────────────────────────────────────────

pub struct CapNativeBuilder {
    config: CapNativeConfig,
    device: Device,
    seed: u64,
    /// Optional bootstrap token sample for cap discovery (KMeans uses this).
    bootstrap_sample_tokens: Option<Vec<u32>>,
}

impl Default for CapNativeBuilder {
    fn default() -> Self {
        Self {
            config: CapNativeConfig::default(),
            device: Device::Cpu,
            seed: 42,
            bootstrap_sample_tokens: None,
        }
    }
}

impl CapNativeBuilder {
    pub fn with_vocab(mut self, v: usize) -> Self {
        self.config.vocab = v;
        self
    }
    pub fn with_d_model(mut self, d: usize) -> Self {
        self.config.d_model = d;
        self
    }
    pub fn with_n_blocks(mut self, n: usize) -> Self {
        self.config.n_blocks = n;
        self
    }
    pub fn with_n_heads(mut self, n: usize) -> Self {
        self.config.n_heads = n;
        self
    }
    pub fn with_d_ff(mut self, n: usize) -> Self {
        self.config.d_ff = n;
        self
    }
    pub fn with_max_seq_len(mut self, n: usize) -> Self {
        self.config.max_seq_len = n;
        self
    }
    pub fn with_rope_base(mut self, b: f64) -> Self {
        self.config.rope_base = b;
        self
    }
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.config.top_k = k;
        self
    }
    pub fn with_cap_indexed_mask(mut self, b: bool) -> Self {
        self.config.cap_indexed_mask = b;
        self
    }
    pub fn with_routing(mut self, r: RoutingMode) -> Self {
        self.config.routing = r;
        self
    }
    pub fn with_device(mut self, d: Device) -> Self {
        self.device = d;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
    pub fn with_config(mut self, c: CapNativeConfig) -> Self {
        self.config = c;
        self
    }

    /// Configure the input cap layer (discovery, n_caps, cap_window).
    pub fn with_cap_layer_config(mut self, cap_config: super::super::config::CapConfig) -> Self {
        self.config.cap_config = cap_config;
        self
    }

    /// Enable hierarchical mode (Phase E): second discovered CapLayer
    /// over h_0 drives the downstream cap-keyed components.
    pub fn with_hierarchical(mut self, enabled: bool) -> Self {
        self.config.hierarchical = enabled;
        self
    }

    /// Configure layer 1 (hierarchical mode only).
    pub fn with_cap_layer_1_config(
        mut self,
        cap_layer_1: super::config::CapLayer1Config,
    ) -> Self {
        self.config.cap_layer_1 = cap_layer_1;
        self
    }

    /// Provide tokens for cap discovery bootstrap (KMeans uses this).
    pub fn with_bootstrap_sample_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.bootstrap_sample_tokens = Some(tokens);
        self
    }

    pub fn build(self) -> CResult<CapNativeSubstrate> {
        let cfg = self.config;
        let device = self.device;
        let dtype = cfg.compression.dtype.to_candle();
        let varmap = Arc::new(Mutex::new(VarMap::new()));
        let cap_window = cfg.cap_window();

        // Token embeddings (trainable, registered in varmap).
        let embeddings = {
            let init_data = (Tensor::randn(0f32, 1f32, (cfg.vocab, cfg.d_model), &device)?
                * (1.0 / (cfg.d_model as f64).sqrt()))?
            .to_dtype(dtype)?;
            let var = Var::from_tensor(&init_data)?;
            {
                let vm = varmap.lock().unwrap();
                let data = vm.data();
                let mut data = data.lock().unwrap();
                data.insert("embeddings.weight".to_string(), var.clone());
            }
            Embedding::new(var.as_tensor().clone(), cfg.d_model)
        };

        // Shape bootstrap sample for cap-layer discovery.
        // window=1: [N, d_emb]; window>1: [N, K*d_emb].
        let bootstrap_sample: Option<Tensor> = if let Some(toks) = &self.bootstrap_sample_tokens {
            let n_total = toks.len();
            if n_total == 0 {
                None
            } else if cap_window > 1 && n_total % cap_window == 0 {
                let n_windows = n_total / cap_window;
                let tok_tensor =
                    Tensor::from_vec(toks.clone(), (n_total,), &device)?.to_dtype(DType::U32)?;
                let embs = embeddings.forward(&tok_tensor)?;
                let d_emb = embs.dim(1)?;
                let win = embs.reshape((n_windows, cap_window, d_emb))?;
                Some(win.reshape((n_windows, cap_window * d_emb))?)
            } else {
                let tok_tensor =
                    Tensor::from_vec(toks.clone(), (n_total,), &device)?.to_dtype(DType::U32)?;
                Some(embeddings.forward(&tok_tensor)?)
            }
        } else {
            None
        };

        // Build CapLayer with the same VarMap so its w_proj is trainable.
        let cap_layer = {
            let vm_guard = varmap.lock().unwrap();
            let vb = VarBuilder::from_varmap(&vm_guard, dtype, &device);
            CapLayer::new(
                cfg.cap_config.clone(),
                cfg.d_model, // d_emb == d_model in this setup
                cfg.d_model,
                &device,
                bootstrap_sample.as_ref(),
                vb.pp("cap_layer"),
            )?
        };

        // Hierarchical: discover layer 1 over h_0 samples.
        // Phase E: forwards layer 0 on the bootstrap tokens to get h_0
        // activations, then builds layer 1 CapLayer with h_0 as its
        // KMeans sample. Downstream cap-keyed components are sized to
        // n_caps_1 and routed by cap_acts_1.
        //
        // For W_1 > 1 (Phase G), we additionally window h_0 to width W_1
        // before passing to KMeans — analogous to layer 0's windowing of
        // token embeddings, but operating in d_model space over
        // contextualised representations.
        let cap_layer_1: Option<CapLayer> = if cfg.hierarchical {
            let l1_window = cfg.cap_layer_1.cap_window.max(1);
            // Build h_0 samples by forwarding layer 0 on the bootstrap
            // tokens. Use the same token list that drove layer 0's
            // discovery (the user-supplied bootstrap_sample_tokens).
            let h0_sample: Option<Tensor> = if let Some(toks) = &self.bootstrap_sample_tokens {
                let n_total = toks.len();
                if n_total == 0 {
                    None
                } else {
                    let tok_tensor = Tensor::from_vec(toks.clone(), (1, n_total), &device)?
                        .to_dtype(DType::U32)?;
                    let emb_3d = embeddings.forward(&tok_tensor)?; // (1, n_total, d_model)
                    let h0 = cap_layer.forward(&emb_3d)?; // (1, n_total, d_model)
                    if l1_window == 1 {
                        // Flat per-position sample for W_1 = 1.
                        Some(h0.reshape((n_total, cfg.d_model))?)
                    } else {
                        // Causal windowing of h_0 to width W_1: pad left
                        // with W_1 - 1 zero positions, then concat W_1
                        // shifted slices along the feature dim. Output
                        // shape: (n_total, W_1 * d_model).
                        let pad = Tensor::zeros(
                            (1, l1_window - 1, cfg.d_model),
                            h0.dtype(),
                            &device,
                        )?;
                        let padded = Tensor::cat(&[&pad, &h0], 1)?; // (1, n_total+W_1-1, d_model)
                        let mut pieces: Vec<Tensor> = Vec::with_capacity(l1_window);
                        for off in 0..l1_window {
                            pieces.push(padded.narrow(1, off, n_total)?);
                        }
                        let refs: Vec<&Tensor> = pieces.iter().collect();
                        let windowed = Tensor::cat(&refs, candle_core::D::Minus1)?; // (1, n_total, W_1 * d_model)
                        Some(windowed.reshape((n_total, l1_window * cfg.d_model))?)
                    }
                }
            } else {
                None
            };

            let mut l1_cap_config = super::super::config::CapConfig::default();
            l1_cap_config.n_caps_target = cfg.cap_layer_1.n_caps_target;
            l1_cap_config.n_caps_budget = cfg.cap_layer_1.n_caps_budget;
            l1_cap_config.gradient_train = cfg.cap_layer_1.gradient_train;
            l1_cap_config.cap_window = l1_window;
            l1_cap_config.discovery = cfg.cap_layer_1.discovery;

            let vm_guard = varmap.lock().unwrap();
            let vb = VarBuilder::from_varmap(&vm_guard, dtype, &device);
            Some(CapLayer::new(
                l1_cap_config,
                cfg.d_model, // d_emb = d_model — layer 1's input is h_0
                cfg.d_model,
                &device,
                h0_sample.as_ref(),
                vb.pp("cap_layer_1"),
            )?)
        } else {
            None
        };

        // Cap-native blocks (all cap-keyed, no input projection - h0 comes from cap_layer.forward).
        // Hierarchical: downstream blocks sized to n_caps_1 so they
        // consume cap_acts_1 from layer 1.
        let downstream_n_caps = cfg.downstream_n_caps();
        let block_cfg = CapNativeBlockConfig {
            d_model: cfg.d_model,
            n_heads: cfg.n_heads,
            d_ff: cfg.d_ff,
            max_seq_len: cfg.max_seq_len,
            rope_base: cfg.rope_base,
            rms_eps: cfg.rms_eps,
            n_caps: downstream_n_caps,
            top_k: cfg.top_k,
            cap_indexed_mask: cfg.cap_indexed_mask,
            routing: cfg.routing,
        };
        let mut blocks: Vec<CapNativeBlock> = Vec::with_capacity(cfg.n_blocks);
        for b in 0..cfg.n_blocks {
            blocks.push(CapNativeBlock::new(
                b,
                &block_cfg,
                &format!("block.{}", b),
                varmap.clone(),
                device.clone(),
                dtype,
            )?);
        }

        // Final cap-keyed norm.
        let final_norm = CapKeyedRmsNorm::new(
            downstream_n_caps,
            cfg.d_model,
            cfg.top_k,
            cfg.rms_eps,
            "final_norm",
            varmap.clone(),
            device.clone(),
            dtype,
        )?;

        // Cap-keyed output projection.
        let output = CapKeyedOutput::new(
            downstream_n_caps,
            cfg.d_model,
            cfg.vocab,
            cfg.top_k,
            "output",
            varmap.clone(),
            device.clone(),
            dtype,
        )?
        .with_routing(cfg.routing);

        Ok(CapNativeSubstrate {
            config: cfg,
            varmap,
            device,
            dtype,
            embeddings,
            cap_layer,
            cap_layer_1,
            blocks,
            final_norm,
            output,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aware::cap_native::config::CapNativeConfig;
    use crate::aware::config::CapConfig;
    use crate::aware::discover::DiscoveryKind;

    fn cpu() -> Device {
        Device::Cpu
    }

    fn tiny_config() -> CapNativeConfig {
        let mut cap_config = CapConfig::default();
        cap_config.n_caps_target = 8;
        cap_config.n_caps_budget = 16;
        cap_config.gradient_train = false;
        cap_config.cap_window = 1; // simplest for tests
        cap_config.discovery = DiscoveryKind::NoDiscovery;
        CapNativeConfig {
            vocab: 32,
            d_model: 16,
            n_blocks: 2,
            n_heads: 2,
            d_ff: 32,
            max_seq_len: 16,
            rope_base: 10_000.0,
            rms_eps: 1e-5,
            cap_config,
            top_k: 0,
            cap_indexed_mask: false,
            routing: RoutingMode::SoftTopK,
            compression: Default::default(),
            hierarchical: false,
            cap_layer_1: Default::default(),
        }
    }

    #[test]
    fn build_and_forward() {
        let model = CapNativeSubstrate::builder()
            .with_config(tiny_config())
            .with_device(cpu())
            .build()
            .unwrap();
        let tokens = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &cpu()).unwrap();
        let logits = model.forward(&tokens).unwrap();
        assert_eq!(logits.dims(), &[1, 4, 32]);
    }

    #[test]
    fn n_params_nonzero() {
        let model = CapNativeSubstrate::builder()
            .with_config(tiny_config())
            .with_device(cpu())
            .build()
            .unwrap();
        let n = model.n_params();
        assert!(n > 100, "n_params seems too small: {}", n);
    }

    #[test]
    fn loss_returns_scalar() {
        let model = CapNativeSubstrate::builder()
            .with_config(tiny_config())
            .with_device(cpu())
            .build()
            .unwrap();
        let tokens = Tensor::from_vec(vec![0u32, 1, 2, 3, 4], (1, 5), &cpu()).unwrap();
        let loss = model.loss(&tokens).unwrap();
        let v: f32 = loss.to_scalar().unwrap();
        assert!(v.is_finite(), "loss not finite: {}", v);
        assert!(v > 0.0, "loss non-positive: {}", v);
    }

    #[test]
    fn build_with_kmeans_discovery_and_bootstrap() {
        let mut cfg = tiny_config();
        cfg.cap_config.discovery = DiscoveryKind::KMeans;
        cfg.cap_config.cap_window = 2; // exercise windowed bootstrap shaping

        // Need 2-token windows: bootstrap_tokens length should be a multiple of 2.
        let bootstrap = vec![0u32, 1, 2, 3, 4, 5, 6, 7]; // 4 windows of 2 tokens
        let model = CapNativeSubstrate::builder()
            .with_config(cfg)
            .with_device(cpu())
            .with_bootstrap_sample_tokens(bootstrap)
            .build()
            .unwrap();
        let tokens = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &cpu()).unwrap();
        let logits = model.forward(&tokens).unwrap();
        assert_eq!(logits.dims(), &[1, 4, 32]);
    }

    #[test]
    fn build_and_forward_hierarchical_w1_2() {
        // Phase G prep: W_1=2 windowing over h_0. Layer 1 sees 2-position
        // windows of the contextualised representations from layer 0.
        let mut cfg = tiny_config();
        cfg.cap_config.discovery = DiscoveryKind::KMeans;
        cfg.cap_config.cap_window = 2;
        cfg.cap_config.n_caps_target = 8;
        cfg.hierarchical = true;
        cfg.cap_layer_1.n_caps_target = 4;
        cfg.cap_layer_1.cap_window = 2; // W_1 > 1
        cfg.cap_layer_1.discovery = DiscoveryKind::KMeans;

        let bootstrap = (0u32..16).collect::<Vec<_>>();
        let model = CapNativeSubstrate::builder()
            .with_config(cfg)
            .with_device(cpu())
            .with_bootstrap_sample_tokens(bootstrap)
            .build()
            .unwrap();

        assert!(model.cap_layer_1.is_some());
        // Layer 1 should be set up with cap_window=2; its d_in_per_window
        // is d_model * W_1 = 16 * 2 = 32.
        let layer1 = model.cap_layer_1.as_ref().unwrap();
        assert_eq!(layer1.d_in_per_window, 32);
        assert_eq!(layer1.config.cap_window, 2);

        let tokens = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &cpu()).unwrap();
        let logits = model.forward(&tokens).unwrap();
        assert_eq!(logits.dims(), &[1, 4, 32]);
    }

    #[test]
    fn build_and_forward_hierarchical() {
        // Phase E: two-layer cap discovery. Layer 0 fires over token
        // windows, layer 1 fires over the layer-0 output (h_0). All
        // downstream cap-keyed components are sized to n_caps_1.
        let mut cfg = tiny_config();
        cfg.cap_config.discovery = DiscoveryKind::KMeans;
        cfg.cap_config.cap_window = 2;
        cfg.cap_config.n_caps_target = 8;
        cfg.hierarchical = true;
        cfg.cap_layer_1.n_caps_target = 4; // smaller than layer 0
        cfg.cap_layer_1.cap_window = 1;
        cfg.cap_layer_1.discovery = DiscoveryKind::KMeans;

        let bootstrap = (0u32..16).collect::<Vec<_>>();
        let model = CapNativeSubstrate::builder()
            .with_config(cfg)
            .with_device(cpu())
            .with_bootstrap_sample_tokens(bootstrap)
            .build()
            .unwrap();

        // Layer 1 must be present.
        assert!(model.cap_layer_1.is_some());
        // Downstream blocks/norm/output sized to layer-1 n_caps.
        for block in &model.blocks {
            assert_eq!(block.attn.n_caps, 4);
        }
        assert_eq!(model.final_norm.n_caps, 4);
        assert_eq!(model.output.n_caps, 4);

        let tokens = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &cpu()).unwrap();
        let logits = model.forward(&tokens).unwrap();
        assert_eq!(logits.dims(), &[1, 4, 32]);
    }
}
