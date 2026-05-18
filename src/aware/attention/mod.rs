pub mod cap_memory;
pub mod cap_pair;
pub mod standard;

use candle_core::{Result, Tensor};
use serde::{Deserialize, Serialize};

/// Where this attention layer's CapMatrix lives (for cap-native variants).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CapMatrixSource {
    /// Block owns its own CapMatrix.
    Local { n_caps: usize },
    /// All blocks share a substrate-level global CapMatrix.
    Shared,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionKind {
    /// Transformer self-attention (Q, K, V, O matrices, multi-head, RoPE).
    Standard,
    /// Caps as memory: Q from input, K & V from a CapMatrix.
    /// Positions attend to caps; T × n_caps attention pattern.
    CapMemory { source: CapMatrixSource },
    /// Capsule-style routing: cap_acts (from a CapMatrix fired on input)
    /// drive attention scores via cap-pair affinity matrix A_pair.
    CapPair { source: CapMatrixSource },
}

pub trait Attention: Send + Sync {
    /// Forward through attention. Input: [B, T, d_model], output same shape.
    fn forward(
        &self,
        x: &Tensor,
        rope: &super::embed::RoPE,
        causal_mask: &Tensor,
    ) -> Result<Tensor>;
}
