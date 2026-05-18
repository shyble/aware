// AttnLayer: Pre-LN + Attention + residual.

use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

use super::super::attention::{
    cap_memory::CapMemoryAttention, cap_pair::CapPairAttention, standard::StandardAttention,
    Attention, AttentionKind,
};
use super::super::config::{BlockConfig, Config};
use super::super::embed::RoPE;
use super::super::norm::RmsNorm;
use super::super::substrate::BlockBuildCtx;

pub struct AttnLayer {
    pub ln: RmsNorm,
    pub attn: Box<dyn Attention>,
}

impl AttnLayer {
    pub fn new(
        vcfg: &Config,
        bcfg: &BlockConfig,
        device: &candle_core::Device,
        ctx: &BlockBuildCtx,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ln = RmsNorm::new(vcfg.d_model, bcfg.norm_eps, vb.pp("ln"))?;
        let attn: Box<dyn Attention> = match &bcfg.attention {
            AttentionKind::Standard => {
                Box::new(StandardAttention::new(vcfg, bcfg.n_heads, vb.pp("attn"))?)
            }
            AttentionKind::CapMemory { source } => Box::new(CapMemoryAttention::new(
                vcfg,
                source,
                device,
                ctx,
                vb.pp("attn"),
            )?),
            AttentionKind::CapPair { source } => Box::new(CapPairAttention::new(
                vcfg,
                source,
                device,
                ctx,
                vb.pp("attn"),
            )?),
        };
        Ok(Self { ln, attn })
    }

    pub fn forward(&self, x: &Tensor, rope: &RoPE, causal_mask: &Tensor) -> Result<Tensor> {
        let h = self.ln.forward(x)?;
        let h = self.attn.forward(&h, rope, causal_mask)?;
        x + h
    }
}
