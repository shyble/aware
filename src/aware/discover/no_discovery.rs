// NoDiscovery: Xavier random init for gradient-trained caps. No audit.
// Equivalent to standard transformer FFN slot initialization.

use candle_core::{Result, Tensor};

use super::{Discovery, DiscoveryCtx};
use crate::aware::cap::{CapKind, CapMatrix};

pub struct NoDiscovery {
    pub init_std: f64,
}

impl Default for NoDiscovery {
    fn default() -> Self {
        Self { init_std: 0.02 }
    }
}

impl Discovery for NoDiscovery {
    fn bootstrap(&self, ctx: &DiscoveryCtx) -> Result<CapMatrix> {
        let mut m = CapMatrix::empty(
            CapKind::Gradient,
            ctx.d_in,
            ctx.d_out,
            ctx.n_caps_target,
            true, // trainable = true
            ctx.device.clone(),
        )?;
        m.keys = Tensor::randn(
            0.0f32,
            self.init_std as f32,
            (ctx.n_caps_target, ctx.d_in),
            ctx.device,
        )?;
        if let Some(d_out) = ctx.d_out {
            m.values = Some(Tensor::randn(
                0.0f32,
                self.init_std as f32,
                (ctx.n_caps_target, d_out),
                ctx.device,
            )?);
        }
        Ok(m)
    }
}
