use candle_core::{DType, Device};
use serde::{Deserialize, Serialize};

/// Floating-point precision used by params + activations.
///
/// **Tested invariants** (per-component, see off_domain_cap_slabs_receive_zero_gradient tests):
/// - F32: safe on CPU + Metal + CUDA. Reference precision.
/// - BF16: safe on Metal (verified). CUDA expected; CPU candle 0.10 has
///   no BF16 matmul kernel - auto-select downgrades to F32 on CPU.
/// - F16: works on Metal but candle CPU init underflows to zero. Avoid.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CompressionDType {
    F32,
    F16,
    BF16,
}

impl Default for CompressionDType {
    fn default() -> Self {
        Self::F32
    }
}

impl CompressionDType {
    pub fn to_candle(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
        }
    }

    pub fn from_candle(d: DType) -> Self {
        match d {
            DType::F32 => Self::F32,
            DType::F16 => Self::F16,
            DType::BF16 => Self::BF16,
            other => panic!("unsupported candle dtype for compression: {:?}", other),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Metal,
    Cuda,
}

impl DeviceKind {
    pub fn from_device(device: &Device) -> Self {
        match device {
            Device::Cpu => Self::Cpu,
            Device::Metal(_) => Self::Metal,
            Device::Cuda(_) => Self::Cuda,
        }
    }
    pub fn is_gpu(self) -> bool {
        !matches!(self, Self::Cpu)
    }
}

/// Routing strategy for cap-keyed components.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RoutingMode {
    /// Soft top-K softmax routing. Computes ALL n_caps projections each
    /// forward, masks to top-K via softmax. Memory scales linearly with
    /// n_caps. R2c.7-validated for continual learning.
    SoftTopK,
    /// Hard top-1 sparse-gather routing. Only the winning cap's
    /// projection is computed per token. Off-domain caps receive
    /// structurally zero gradient (by not being computed at all).
    /// Memory scales with d_model × out_dim, NOT with n_caps × ...
    /// Forward: argmax over cap_acts -> partition tokens by winner ->
    /// per-cap matmul on subset -> scatter back via index_add.
    HardTop1Sparse,
}

impl Default for RoutingMode {
    fn default() -> Self {
        Self::SoftTopK
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    #[serde(default)]
    pub dtype: CompressionDType,
    #[serde(default)]
    pub max_n_caps: Option<usize>,
    #[serde(default)]
    pub state_space_checkpoint_every: usize,
    #[serde(default)]
    pub routing: RoutingMode,
    #[serde(default)]
    pub auto: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            dtype: CompressionDType::F32,
            max_n_caps: None,
            state_space_checkpoint_every: 0,
            routing: RoutingMode::SoftTopK,
            auto: false,
        }
    }
}

pub struct AutoInputs {
    pub device_kind: DeviceKind,
    pub estimated_params: usize,
    pub n_caps: usize,
    pub use_state_space: bool,
}

impl CompressionConfig {
    /// Resolve auto-select knobs. Routing is NEVER auto-decided - user
    /// architectural choice survives `auto = true`.
    pub fn resolve(mut self, inputs: &AutoInputs) -> Self {
        if !self.auto {
            return self;
        }
        let user_routing = self.routing;
        let user_max_n_caps = self.max_n_caps;

        self.dtype = CompressionDType::F32;
        self.state_space_checkpoint_every = 0;
        self.auto = false;
        self.routing = user_routing;
        self.max_n_caps = user_max_n_caps;

        if !inputs.device_kind.is_gpu() {
            return self;
        }
        if inputs.estimated_params >= 5_000_000 {
            self.dtype = CompressionDType::BF16;
        }
        if inputs.estimated_params >= 50_000_000 && self.max_n_caps.is_none() {
            self.max_n_caps = Some((inputs.n_caps + 16).max(64));
        }
        if inputs.use_state_space && inputs.estimated_params >= 5_000_000 {
            self.state_space_checkpoint_every = 16;
        }
        self
    }

    pub fn label(&self) -> String {
        let routing_str = match self.routing {
            RoutingMode::SoftTopK => "soft_topk",
            RoutingMode::HardTop1Sparse => "hard_top1",
        };
        format!(
            "dtype={} max_n_caps={} ssp_ckpt={} routing={} auto={}",
            self.dtype.label(),
            self.max_n_caps
                .map(|n| n.to_string())
                .unwrap_or_else(|| "∞".into()),
            self.state_space_checkpoint_every,
            routing_str,
            if self.auto { "yes" } else { "no" },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_roundtrip() {
        for d in [
            CompressionDType::F32,
            CompressionDType::F16,
            CompressionDType::BF16,
        ] {
            assert_eq!(d, CompressionDType::from_candle(d.to_candle()));
        }
    }

    #[test]
    fn auto_disabled_passes_through() {
        let cfg = CompressionConfig {
            dtype: CompressionDType::BF16,
            max_n_caps: Some(32),
            state_space_checkpoint_every: 8,
            routing: RoutingMode::SoftTopK,
            auto: false,
        };
        let inputs = AutoInputs {
            device_kind: DeviceKind::Cpu,
            estimated_params: 100_000_000,
            n_caps: 16,
            use_state_space: true,
        };
        let resolved = cfg.resolve(&inputs);
        assert_eq!(resolved.dtype, CompressionDType::BF16);
        assert_eq!(resolved.max_n_caps, Some(32));
        assert_eq!(resolved.state_space_checkpoint_every, 8);
    }

    #[test]
    fn auto_cpu_stays_f32() {
        let cfg = CompressionConfig {
            auto: true,
            ..Default::default()
        };
        let inputs = AutoInputs {
            device_kind: DeviceKind::Cpu,
            estimated_params: 100_000_000,
            n_caps: 32,
            use_state_space: true,
        };
        let resolved = cfg.resolve(&inputs);
        assert_eq!(resolved.dtype, CompressionDType::F32);
        assert_eq!(resolved.max_n_caps, None);
    }

    #[test]
    fn auto_preserves_user_supplied_routing() {
        let cfg = CompressionConfig {
            auto: true,
            routing: RoutingMode::HardTop1Sparse,
            ..Default::default()
        };
        let inputs = AutoInputs {
            device_kind: DeviceKind::Metal,
            estimated_params: 10_000_000,
            n_caps: 16,
            use_state_space: false,
        };
        let resolved = cfg.resolve(&inputs);
        assert_eq!(
            resolved.routing,
            RoutingMode::HardTop1Sparse,
            "auto must preserve user-supplied routing"
        );
        assert_eq!(resolved.dtype, CompressionDType::BF16);
    }
}
