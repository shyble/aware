// FfnLayer: Pre-LN + (W_up -> activation -> W_down) + residual.

use candle_core::{Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use super::super::config::{ActivationKind, Config, FfnConfig};
use super::super::norm::RmsNorm;

pub struct FfnLayer {
    pub ln: RmsNorm,
    pub w_up: Linear,
    pub w_down: Linear,
    pub activation: ActivationKind,
}

impl FfnLayer {
    pub fn new(vcfg: &Config, fcfg: &FfnConfig, vb: VarBuilder) -> Result<Self> {
        let ln = RmsNorm::new(vcfg.d_model, vcfg.norm_eps, vb.pp("ln"))?;
        let w_up = candle_nn::linear_no_bias(vcfg.d_model, fcfg.d_ff, vb.pp("w_up"))?;
        let w_down = candle_nn::linear_no_bias(fcfg.d_ff, vcfg.d_model, vb.pp("w_down"))?;
        Ok(Self {
            ln,
            w_up,
            w_down,
            activation: fcfg.activation,
        })
    }

    fn apply_activation(&self, x: &Tensor) -> Result<Tensor> {
        match self.activation {
            ActivationKind::Gelu => x.gelu(),
            ActivationKind::Relu => x.relu(),
            ActivationKind::Silu => x.silu(),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.ln.forward(x)?;
        let h = self.w_up.forward(&h)?;
        let h = self.apply_activation(&h)?;
        let h = self.w_down.forward(&h)?;
        x + h
    }
}
