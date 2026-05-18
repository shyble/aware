// Block: composes AttnLayer + optional FfnLayer.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::super::config::{BlockConfig, Config};
use super::super::embed::RoPE;
use super::super::substrate::BlockBuildCtx;
use super::{AttnLayer, FfnLayer};

pub struct Block {
    pub attn: AttnLayer,
    pub ffn: Option<FfnLayer>,
}

impl Block {
    pub fn new(
        vcfg: &Config,
        bcfg: &BlockConfig,
        device: &candle_core::Device,
        ctx: &BlockBuildCtx,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attn = AttnLayer::new(vcfg, bcfg, device, ctx, vb.pp("attn"))?;
        let ffn = if let Some(fcfg) = &bcfg.ffn {
            Some(FfnLayer::new(vcfg, fcfg, vb.pp("ffn"))?)
        } else {
            None
        };
        Ok(Self { attn, ffn })
    }

    pub fn forward(&self, x: &Tensor, rope: &RoPE, causal_mask: &Tensor) -> Result<Tensor> {
        let mut x = self.attn.forward(x, rope, causal_mask)?;
        if let Some(ffn) = &self.ffn {
            x = ffn.forward(&x)?;
        }
        Ok(x)
    }
}
