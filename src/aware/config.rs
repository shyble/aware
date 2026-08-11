//
// Config is the top-level. Sub-configs compose: CapConfig, BlockConfig,
// AttentionConfig, AuditConfig. Sensible defaults for TinyStories-scale.

use super::attention::AttentionKind;
use super::cap::CapKind;
use super::discover::DiscoveryKind;

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub max_seq_len: usize,
    pub rope_base: f64,
    pub norm_eps: f64,
    pub dropout_p: f64,
    pub tied_embeddings: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 512,
            d_model: 128,
            max_seq_len: 128,
            rope_base: 10_000.0,
            norm_eps: 1e-6,
            dropout_p: 0.0,
            tied_embeddings: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CapConfig {
    pub kind: CapKind,
    pub discovery: DiscoveryKind,
    /// Initial / target cap count. For NoDiscovery and Random this is exact.
    /// For KMeans / Hybrid it's an upper hint; bootstrap may produce fewer
    /// if variance plateaus.
    pub n_caps_target: usize,
    /// Hard ceiling; audit / growth cannot exceed this.
    pub n_caps_budget: usize,
    pub gradient_train: bool,
    /// Let discovered keys move during training, via a zero-initialised
    /// delta on the frozen base.
    ///
    /// Deliberately SEPARATE from `gradient_train`, which means something
    /// else here: "these keys were not discovered from data, so learn
    /// them". Adapting discovered keys and learning undiscovered ones are
    /// different intentions, and conflating them silently changes every
    /// `nodiscovery` and every checkpoint-loading run.
    pub adapt_keys: bool,
    /// Window size for cap input. 1 = each cap sees one token's embedding.
    /// N > 1 = cap sees concatenated past N embeddings (causal window).
    pub cap_window: usize,
    pub audit: AuditConfig,
}

impl Default for CapConfig {
    fn default() -> Self {
        Self {
            kind: CapKind::Discovered,
            discovery: DiscoveryKind::KMeans,
            n_caps_target: 330,
            n_caps_budget: 512,
            gradient_train: false,
            adapt_keys: false,
            cap_window: 1,
            audit: AuditConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AuditConfig {
    pub dead_rate_threshold: f64,
    pub novelty_threshold: f64,
    pub min_dead_fraction: f64,
    pub min_novelty_fraction: f64,
    pub replace_budget_per_call: usize,
    pub audit_every_steps: usize,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            dead_rate_threshold: 0.01,
            novelty_threshold: 0.5,
            min_dead_fraction: 0.10,
            min_novelty_fraction: 0.05,
            replace_budget_per_call: 20,
            audit_every_steps: 500,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BlockConfig {
    pub attention: AttentionKind,
    pub n_heads: usize,
    pub ffn: Option<FfnConfig>,
    pub norm_eps: f64,
}

impl Default for BlockConfig {
    fn default() -> Self {
        Self {
            attention: AttentionKind::Standard,
            n_heads: 4,
            ffn: Some(FfnConfig::default()),
            norm_eps: 1e-6,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FfnConfig {
    pub d_ff: usize,
    pub activation: ActivationKind,
}

impl Default for FfnConfig {
    fn default() -> Self {
        Self {
            d_ff: 512,
            activation: ActivationKind::Gelu,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ActivationKind {
    Gelu,
    Relu,
    Silu,
}
