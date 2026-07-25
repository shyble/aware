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
    /// Learnable affinity. `[n_heads, n_caps, n_caps]` — one affinity
    /// pattern per head, so heads can specialise on different cap-cap
    /// relations. At n_heads=1 this is the original `[n_caps, n_caps]`.
    pub a_pair: Tensor,
    pub w_v: Linear,
    pub w_o: Linear,
    pub n_heads: usize,
    pub d_head: usize,
    pub d_v: usize,
    pub scale: f64,
    pub n_caps: usize,
}

impl CapPairAttention {
    pub fn new(
        cfg: &Config,
        source: &CapMatrixSource,
        n_heads: usize,
        discovery: DiscoveryKind,
        device: &Device,
        ctx: &BlockBuildCtx,
        vb: VarBuilder,
    ) -> Result<Self> {
        let d_model = cfg.d_model;
        let n_heads = n_heads.max(1);
        if d_model % n_heads != 0 {
            return Err(candle_core::Error::Msg(format!(
                "CapPairAttention: d_model {} not divisible by n_heads {}",
                d_model, n_heads
            )));
        }
        let d_head = d_model / n_heads;

        let (keys_src, n_caps) = match source {
            CapMatrixSource::Local { n_caps } => {
                let disc = make_discovery(discovery);
                let dctx = DiscoveryCtx {
                    device,
                    d_in: d_model,
                    d_out: None,
                    n_caps_target: *n_caps,
                    sample: ctx.bootstrap_sample,
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

        // A_pair: one [n_caps, n_caps] affinity per head, small random init.
        let a_pair = vb.get_with_hints(
            (n_heads, n_caps, n_caps),
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
            n_heads,
            d_head,
            d_v: d_model,
            scale: 1.0 / (d_head as f64).sqrt(),
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
        let (b, t, d_model) = x.dims3()?;
        let h = self.n_heads;
        let dh = self.d_head;

        // 1. cap_acts = gelu(x @ keys^T): [B, T, n_caps]. Shared across heads —
        //    heads differ in their affinity pattern, not in which caps fire.
        let keys_t = self.keys().transpose(0, 1)?.contiguous()?;
        let cap_acts = x.broadcast_matmul(&keys_t)?.gelu()?;

        // 2. Per-head scores via that head's affinity matrix.
        //    c_a[h] = cap_acts @ A_pair[h]                -> [B, h, T, n_caps]
        //    scores[h][i, j] = c_a[h][i] · cap_acts[j]    -> [B, h, T, T]
        let cap_acts_b = cap_acts.unsqueeze(1)?; // [B, 1, T, n_caps]
        let a_pair_b = self.a_pair.unsqueeze(0)?; // [1, h, n_caps, n_caps]
        let c_a = cap_acts_b.broadcast_matmul(&a_pair_b)?; // [B, h, T, n_caps]
        let cap_acts_t = cap_acts_b
            .transpose(D::Minus2, D::Minus1)?
            .contiguous()?; // [B, 1, n_caps, T]
        let scores = c_a.broadcast_matmul(&cap_acts_t)?; // [B, h, T, T]
        let scores = (scores * self.scale)?;

        // Causal mask + softmax
        let mask_slice = causal_mask.narrow(0, 0, t)?.narrow(1, 0, t)?;
        let scores = scores.broadcast_add(&mask_slice.unsqueeze(0)?.unsqueeze(0)?)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?; // [B, h, T, T]

        // 3. V from input, split into heads: [B, h, T, dh]
        let v = self
            .w_v
            .forward(x)?
            .reshape((b, t, h, dh))?
            .transpose(1, 2)?
            .contiguous()?;

        // 4. attn @ V per head, then merge heads: [B, T, d_model]
        let out = attn.matmul(&v)?; // [B, h, T, dh]
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, d_model))?;

        self.w_o.forward(&out)
    }
}
