pub mod discovered;
pub mod frozen;
pub mod gradient;
pub mod hybrid;
pub mod matrix;

pub use matrix::{CapMatrix, CapMeta};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapKind {
    /// Discovered cap with stored direction (legacy AWARE).
    /// Bootstrap via clustering; audit replaces dead caps. Frozen content
    /// by default; gradient updates only if explicitly enabled.
    Discovered,

    /// Gradient-trained cap: random init, fully gradient-updated.
    /// Equivalent to a transformer FFN slot. No audit.
    Gradient,

    /// Frozen-random cap: random init, never updates.
    /// Reservoir computing style. Edges learn what to do with them.
    FrozenRandom,

    /// Hybrid: bootstrap by discovery, then both gradient + audit apply.
    Hybrid,
}

impl CapKind {
    /// Whether this kind accepts gradient updates on its keys.
    pub fn is_gradient_trainable(&self) -> bool {
        matches!(self, Self::Gradient | Self::Hybrid)
    }

    /// Whether this kind supports the audit lifecycle.
    pub fn supports_audit(&self) -> bool {
        matches!(self, Self::Discovered | Self::Hybrid)
    }

    /// Whether this kind needs bootstrap from data.
    pub fn needs_bootstrap(&self) -> bool {
        matches!(self, Self::Discovered | Self::Hybrid)
    }
}

/// A per-cap view over CapMatrix. Most ops are on the matrix; this is for
/// debugging and individual cap inspection.
pub trait Cap {
    fn id(&self) -> u64;
    fn key(&self) -> &[f32];
    fn value(&self) -> Option<&[f32]>;
    fn is_frozen(&self) -> bool;
    fn meta(&self) -> &CapMeta;
}
