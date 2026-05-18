// Token embedding + Rotary positional encoding (RoPE).
//
// semantics. TokenEmbedding wraps candle_nn::Embedding; RoPE precomputes
// cos/sin tables and rotates Q/K tensors per position.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder};

use super::config::Config;

pub struct TokenEmbedding {
    pub emb: Embedding,
    pub d_model: usize,
}

impl TokenEmbedding {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let emb = candle_nn::embedding(cfg.vocab_size, cfg.d_model, vb)?;
        Ok(Self {
            emb,
            d_model: cfg.d_model,
        })
    }

    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        self.emb.forward(tokens)
    }

    pub fn weight(&self) -> &Tensor {
        self.emb.embeddings()
    }
}

pub struct RoPE {
    pub cos: Tensor,
    pub sin: Tensor,
    pub d_k: usize,
}

impl RoPE {
    pub fn new(d_k: usize, max_seq: usize, base: f64, device: &Device) -> Result<Self> {
        let mut cos = vec![0f32; max_seq * d_k];
        let mut sin = vec![0f32; max_seq * d_k];
        for p in 0..max_seq {
            for i in 0..(d_k / 2) {
                let theta = (base.powf(-2.0 * (i as f64) / (d_k as f64))) as f32;
                let angle = (p as f32) * theta;
                let c = angle.cos();
                let s = angle.sin();
                cos[p * d_k + 2 * i] = c;
                cos[p * d_k + 2 * i + 1] = c;
                sin[p * d_k + 2 * i] = s;
                sin[p * d_k + 2 * i + 1] = s;
            }
        }
        let cos = Tensor::from_vec(cos, (max_seq, d_k), device)?.to_dtype(DType::F32)?;
        let sin = Tensor::from_vec(sin, (max_seq, d_k), device)?.to_dtype(DType::F32)?;
        Ok(Self { cos, sin, d_k })
    }

    /// Apply RoPE rotation to Q or K. Input shape: [B, H, T, d_k].
    pub fn apply(&self, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        let cos = self.cos.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let x_rot = rotate_half(x)?;
        let out = (x.broadcast_mul(&cos)? + x_rot.broadcast_mul(&sin)?)?;
        Ok(out)
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last = x.dim(candle_core::D::Minus1)?;
    let half = last / 2;
    let mut shape: Vec<usize> = x.dims().to_vec();
    let last_pos = shape.len() - 1;
    shape[last_pos] = half;
    shape.push(2);
    let paired = x.reshape(shape.as_slice())?;
    let even = paired.narrow(candle_core::D::Minus1, 0, 1)?;
    let odd = paired.narrow(candle_core::D::Minus1, 1, 1)?;
    let neg_odd = odd.neg()?;
    let recombined = Tensor::cat(&[&neg_odd, &even], candle_core::D::Minus1)?;
    let mut out_shape = x.dims().to_vec();
    let last_pos = out_shape.len() - 1;
    out_shape[last_pos] = last;
    recombined.reshape(out_shape.as_slice())
}

pub fn build_causal_mask(max_seq: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; max_seq * max_seq];
    for i in 0..max_seq {
        for j in (i + 1)..max_seq {
            data[i * max_seq + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (max_seq, max_seq), device)
}
