// Standard self-attention: W_Q, W_K, W_V, W_O dense matrices, multi-head,
// RoPE on Q and K, causal mask, softmax, output projection.

use candle_core::{Module, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};

use super::super::config::Config;
use super::super::embed::RoPE;
use super::Attention;

pub struct StandardAttention {
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,
    pub n_heads: usize,
    pub d_k: usize,
    pub scale: f64,
}

impl StandardAttention {
    pub fn new(cfg: &Config, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        let d_model = cfg.d_model;
        let w_q = candle_nn::linear_no_bias(d_model, d_model, vb.pp("w_q"))?;
        let w_k = candle_nn::linear_no_bias(d_model, d_model, vb.pp("w_k"))?;
        let w_v = candle_nn::linear_no_bias(d_model, d_model, vb.pp("w_v"))?;
        let w_o = candle_nn::linear_no_bias(d_model, d_model, vb.pp("w_o"))?;
        let d_k = d_model / n_heads;
        Ok(Self {
            w_q,
            w_k,
            w_v,
            w_o,
            n_heads,
            d_k,
            scale: 1.0 / (d_k as f64).sqrt(),
        })
    }
}

impl Attention for StandardAttention {
    fn forward(&self, x: &Tensor, rope: &RoPE, causal_mask: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let h = self.n_heads;
        let d_k = self.d_k;

        let q = self
            .w_q
            .forward(x)?
            .reshape((b, t, h, d_k))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .w_k
            .forward(x)?
            .reshape((b, t, h, d_k))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .w_v
            .forward(x)?
            .reshape((b, t, h, d_k))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = rope.apply(&q, t)?;
        let k = rope.apply(&k, t)?;

        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let scores = (q.matmul(&k_t)? * self.scale)?;
        let mask_slice = causal_mask.narrow(0, 0, t)?.narrow(1, 0, t)?;
        let scores = scores.broadcast_add(&mask_slice.unsqueeze(0)?.unsqueeze(0)?)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, h * d_k))?;
        self.w_o.forward(&out)
    }
}
