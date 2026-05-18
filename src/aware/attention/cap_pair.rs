// Cap-pair attention: attention scores from cap-cap affinity.
//
//   cap_acts = GELU(x @ K_capᵀ)                 [B, T, n_caps]
//   scores   = (cap_acts @ A_pair @ cap_actsᵀ) / √d_model + causal_mask
//   attn     = softmax(scores)                   [B, T, T]
//   out      = (attn @ V) @ W_O,  V = x @ W_V

use candle_core::{Device, Module, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};

use super::super::cap::CapMatrix;
use super::super::config::Config;
use super::super::discover::{make_discovery, DiscoveryCtx, DiscoveryKind};
use super::super::embed::RoPE;
use super::super::substrate::BlockBuildCtx;
use super::{Attention, CapMatrixSource};

/// Where this attention's cap keys come from.
enum KeySource {
    Owned(CapMatrix),
    Shared(Tensor),
}

pub struct CapPairAttention {
    keys_src: KeySource, // Caps that fire on input to produce cap_acts.
    pub a_pair: Tensor,  // [n_caps, n_caps] learnable affinity matrix.
    pub w_v: Linear,
    pub w_o: Linear,
    pub d_v: usize,
    pub scale: f64,
    pub n_caps: usize,
}

impl CapPairAttention {
    pub fn new(
        cfg: &Config,
        source: &CapMatrixSource,
        device: &Device,
        ctx: &BlockBuildCtx,
        vb: VarBuilder,
    ) -> Result<Self> {
        let d_model = cfg.d_model;

        let (keys_src, n_caps) = match source {
            CapMatrixSource::Local { n_caps } => {
                let disc = make_discovery(DiscoveryKind::NoDiscovery);
                let dctx = DiscoveryCtx {
                    device,
                    d_in: d_model,
                    d_out: None,
                    n_caps_target: *n_caps,
                    sample: None,
                    audit: &super::super::config::AuditConfig::default(),
                    training_step: 0,
                };
                (KeySource::Owned(disc.bootstrap(&dctx)?), *n_caps)
            }
            CapMatrixSource::Shared => {
                let shared = ctx
                    .shared_cap_keys
                    .ok_or_else(|| {
                        candle_core::Error::Msg(
                            "CapPair Shared requested but no shared cap matrix declared"
                                .to_string(),
                        )
                    })?
                    .clone();
                let n_caps = shared.dim(0)?;
                (KeySource::Shared(shared), n_caps)
            }
        };

        // A_pair: small random init.
        let a_pair = vb.get_with_hints(
            (n_caps, n_caps),
            "a_pair",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        )?;

        let w_v = candle_nn::linear_no_bias(d_model, d_model, vb.pp("w_v"))?;
        let w_o = candle_nn::linear_no_bias(d_model, d_model, vb.pp("w_o"))?;

        Ok(Self {
            keys_src,
            a_pair,
            w_v,
            w_o,
            d_v: d_model,
            scale: 1.0 / (d_model as f64).sqrt(),
            n_caps,
        })
    }

    fn keys(&self) -> &Tensor {
        match &self.keys_src {
            KeySource::Owned(m) => &m.keys,
            KeySource::Shared(k) => k,
        }
    }
}

impl Attention for CapPairAttention {
    fn forward(&self, x: &Tensor, _rope: &RoPE, causal_mask: &Tensor) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;

        // 1. cap_acts = gelu(x @ keys^T): [B, T, n_caps]
        let keys_t = self.keys().transpose(0, 1)?.contiguous()?;
        let cap_acts = x.broadcast_matmul(&keys_t)?.gelu()?;

        // 2. Compute attention scores via A_pair.
        // First: c_a = cap_acts @ A_pair -> [B, T, n_caps]
        let c_a = cap_acts.broadcast_matmul(&self.a_pair)?;
        // Then: scores[i, j] = c_a[i] · cap_acts[j]^T -> [B, T, T]
        let cap_acts_t = cap_acts.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let scores = c_a.matmul(&cap_acts_t)?;
        let scores = (scores * self.scale)?;

        // Causal mask + softmax
        let mask_slice = causal_mask.narrow(0, 0, t)?.narrow(1, 0, t)?;
        let scores = scores.broadcast_add(&mask_slice.unsqueeze(0)?)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?; // [B, T, T]

        // 3. V from input
        let v = self.w_v.forward(x)?; // [B, T, d_v]

        // 4. attn @ V -> [B, T, d_v]
        let out = attn.matmul(&v)?;
        let out = self.w_o.forward(&out)?;

        let _ = b;
        Ok(out)
    }
}
