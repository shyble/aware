use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Module, Result as CResult, Tensor, Var};
use candle_nn::{loss::cross_entropy, Embedding, VarBuilder, VarMap};

use super::super::layer::cap_layer::CapLayer;
use super::block::{CapNativeBlock, CapNativeBlockConfig};
use super::compression::RoutingMode;
use super::config::CapNativeConfig;
use super::norm::CapKeyedRmsNorm;
use super::output::CapKeyedOutput;

pub struct CapNativeSubstrate {
    pub config: CapNativeConfig,
    pub varmap: Arc<Mutex<VarMap>>,
    pub device: Device,
    pub dtype: DType,
    pub embeddings: Embedding,
    /// Layer 0: CapLayer (discovered CapMatrix + windowed firing + projection).
    pub cap_layer: CapLayer,
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
        let cap_acts = self.cap_layer.cap_activations(&emb)?; // (B, S, n_caps)

        let mut h = h0;
        for block in &self.blocks {
            h = block.forward(&h, &cap_acts)?;
        }
        let h = self.final_norm.forward(&h, &cap_acts)?;
        self.output.forward(&h, &cap_acts)
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
        let frozen: usize = self.cap_layer.caps.keys.elem_count()
            + self
                .cap_layer
                .caps
                .values
                .as_ref()
                .map(|v| v.elem_count())
                .unwrap_or(0);
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
        let vb = VarBuilder::from_varmap(&varmap.lock().unwrap(), dtype, &device);
        let cap_layer = CapLayer::new(
            cfg.cap_config.clone(),
            cfg.d_model, // d_emb == d_model in this setup
            cfg.d_model,
            &device,
            bootstrap_sample.as_ref(),
            vb.pp("cap_layer"),
        )?;

        // Cap-native blocks (all cap-keyed, no input projection - h0 comes from cap_layer.forward).
        let block_cfg = CapNativeBlockConfig {
            d_model: cfg.d_model,
            n_heads: cfg.n_heads,
            d_ff: cfg.d_ff,
            max_seq_len: cfg.max_seq_len,
            rope_base: cfg.rope_base,
            rms_eps: cfg.rms_eps,
            n_caps: cfg.n_caps(),
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
            cfg.n_caps(),
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
            cfg.n_caps(),
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
}
