use super::super::config::CapConfig;
use super::super::discover::DiscoveryKind;
use super::compression::{CompressionConfig, RoutingMode};

/// Layer 1 cap config for the hierarchical variant. Layer 1's caps are
/// discovered over the d_model-dimensional output of layer 0 (h_0)
/// rather than over token embeddings, so its scale differs from the
/// input cap layer.
#[derive(Debug, Clone)]
pub struct CapLayer1Config {
    pub n_caps_target: usize,
    pub n_caps_budget: usize,
    pub cap_window: usize,
    pub discovery: DiscoveryKind,
    pub gradient_train: bool,
}

impl Default for CapLayer1Config {
    fn default() -> Self {
        Self {
            n_caps_target: 128,
            n_caps_budget: 512,
            cap_window: 1,
            discovery: DiscoveryKind::KMeans,
            gradient_train: false,
        }
    }
}

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

    /// Path Y / Phase E: two-layer cap discovery. When `hierarchical`
    /// is true, the cap-keyed blocks downstream are sized to and
    /// routed by `cap_layer_1` (the layer-1 caps discovered over h_0).
    pub hierarchical: bool,
    pub cap_layer_1: CapLayer1Config,
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
            hierarchical: false,
            cap_layer_1: CapLayer1Config::default(),
        }
    }
}

impl CapNativeConfig {
    /// Number of caps that drives the cap-keyed components downstream.
    /// In single-discovery mode this is layer 0's n_caps; in
    /// hierarchical mode it's layer 1's n_caps.
    pub fn downstream_n_caps(&self) -> usize {
        if self.hierarchical {
            self.cap_layer_1.n_caps_target
        } else {
            self.cap_config.n_caps_target
        }
    }
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
        if self.hierarchical {
            format!(
                "cap_native_hier(d={},blocks={},heads={},dff={}, n_caps_0={},n_caps_1={},top_k={},w_0={},w_1={},disc_0={:?},disc_1={:?},c4c={})",
                self.d_model, self.n_blocks, self.n_heads, self.d_ff,
                self.n_caps(), self.cap_layer_1.n_caps_target, self.top_k,
                self.cap_window(), self.cap_layer_1.cap_window.max(1),
                self.cap_config.discovery, self.cap_layer_1.discovery,
                self.cap_indexed_mask,
            )
        } else {
            format!(
                "cap_native(d={},blocks={},heads={},dff={}, n_caps={},top_k={},window={},disc={:?},c4c={})",
                self.d_model, self.n_blocks, self.n_heads, self.d_ff,
                self.n_caps(), self.top_k, self.cap_window(),
                self.cap_config.discovery, self.cap_indexed_mask,
            )
        }
    }
}
