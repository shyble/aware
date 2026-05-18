use super::super::config::CapConfig;
use super::compression::{CompressionConfig, RoutingMode};

// Note: CapConfig doesn't impl Serialize/Deserialize, so neither does this.
// Bench runner constructs CapNativeConfig from env vars at runtime; no
// serialization needed.
#[derive(Debug, Clone)]
pub struct CapNativeConfig {
    pub vocab: usize,
    pub d_model: usize,
    pub n_blocks: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub rope_base: f64,
    pub rms_eps: f64,

    /// Input cap layer config - discovery, n_caps, window, audit.
    /// Same primitive the cap-augmented transformer uses (CapConfig).
    pub cap_config: CapConfig,

    /// Top-K for downstream cap-keyed routing (norm/attn/moe/output).
    /// 0 = soft softmax over all caps.
    pub top_k: usize,

    /// R4.C - content-addressable attention mask via cap-overlap.
    pub cap_indexed_mask: bool,

    /// Routing mode for cap-keyed components.
    pub routing: RoutingMode,

    /// Optimization (dtype, compression).
    pub compression: CompressionConfig,
}

impl Default for CapNativeConfig {
    fn default() -> Self {
        let mut cap_config = CapConfig::default();
        cap_config.n_caps_target = 330;
        cap_config.n_caps_budget = 1024;
        cap_config.gradient_train = false; // discovered caps frozen
        cap_config.cap_window = 4;
        cap_config.discovery = super::super::discover::DiscoveryKind::KMeans;
        Self {
            vocab: 512,
            d_model: 128,
            n_blocks: 4,
            n_heads: 4,
            d_ff: 512,
            max_seq_len: 128,
            rope_base: 10_000.0,
            rms_eps: 1e-5,
            cap_config,
            top_k: 0,
            cap_indexed_mask: false,
            routing: RoutingMode::SoftTopK,
            compression: CompressionConfig::default(),
        }
    }
}

impl CapNativeConfig {
    pub fn n_caps(&self) -> usize {
        self.cap_config.n_caps_target
    }
    pub fn cap_window(&self) -> usize {
        self.cap_config.cap_window.max(1)
    }
    pub fn d_head(&self) -> usize {
        self.d_model / self.n_heads
    }

    pub fn label(&self) -> String {
        format!(
 "cap_native(d={},blocks={},heads={},dff={}, n_caps={},top_k={},window={},disc={:?},c4c={})",
 self.d_model, self.n_blocks, self.n_heads, self.d_ff,
 self.n_caps(), self.top_k, self.cap_window(),
 self.cap_config.discovery, self.cap_indexed_mask,
 )
    }
}
