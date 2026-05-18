// RMSNorm: y = x / RMS(x) * scale. Applied along last dim.

use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(d_model: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints((d_model,), "weight", candle_nn::Init::Const(1.0))?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x2 = x.sqr()?;
        let mean = x2.mean_keepdim(candle_core::D::Minus1)?;
        let rms = (mean + self.eps)?.sqrt()?;
        let normalized = x.broadcast_div(&rms)?;
        normalized.broadcast_mul(&self.weight)
    }
}
