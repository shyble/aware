// Random discovery: keys initialized to random unit vectors at bootstrap.
// No audit, no further updates. Reservoir computing style - the cap matrix
// is a fixed random projection; the downstream layers learn what to do with
// it.

use candle_core::{Result, Tensor};

use super::{Discovery, DiscoveryCtx};
use crate::aware::cap::{CapKind, CapMatrix};

pub struct RandomDiscovery {
    pub seed: u64,
}

impl Default for RandomDiscovery {
    fn default() -> Self {
        Self { seed: 0xC0FFEE }
    }
}

impl Discovery for RandomDiscovery {
    fn bootstrap(&self, ctx: &DiscoveryCtx) -> Result<CapMatrix> {
        let mut m = CapMatrix::empty(
            CapKind::FrozenRandom,
            ctx.d_in,
            ctx.d_out,
            ctx.n_caps_target,
            false, // trainable = false (frozen)
            ctx.device.clone(),
        )?;
        // Replace zero keys with random unit vectors.
        let keys = Tensor::randn(0.0f32, 1.0f32, (ctx.n_caps_target, ctx.d_in), ctx.device)?;
        let norm = keys.sqr()?.sum_keepdim(1)?.sqrt()?;
        m.keys = keys.broadcast_div(&norm)?;
        Ok(m)
    }

    // step is a no-op: random caps never update.
}
