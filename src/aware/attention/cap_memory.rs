// Cap-memory attention: Q from input, K and V from a CapMatrix.
// Attention pattern is [B, T, n_caps] (positions attend over caps,
// not other positions). No RoPE, no causal mask.

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use super::super::cap::CapMatrix;
use super::super::config::Config;
use super::super::discover::{make_discovery, DiscoveryCtx, DiscoveryKind};
use super::super::embed::RoPE;
use super::super::substrate::BlockBuildCtx;
use super::{Attention, CapMatrixSource};

/// Where this attention's keys/values come from.
enum KeyValueSource {
    /// Owned local cap matrix (Local source).
    Owned(CapMatrix),
    /// Reference (Tensor clone, Arc-shared underneath) to substrate-level
    /// shared cap matrix (Shared source).
    Shared { keys: Tensor, values: Tensor },
}

pub struct CapMemoryAttention {
    pub w_q: Linear,
    pub w_o: Linear,
    kv: KeyValueSource,
    pub d_k: usize,
    pub d_v: usize,
    pub scale: f64,
}

impl CapMemoryAttention {
    pub fn new(
        cfg: &Config,
        source: &CapMatrixSource,
        device: &Device,
        ctx: &BlockBuildCtx,
        vb: VarBuilder,
    ) -> Result<Self> {
        let d_model = cfg.d_model;

        let kv = match source {
            CapMatrixSource::Local { n_caps } => {
                // Build a per-block local cap matrix with NoDiscovery init.
                let disc = make_discovery(DiscoveryKind::NoDiscovery);
                let dctx = DiscoveryCtx {
                    device,
                    d_in: d_model,
                    d_out: Some(d_model),
                    n_caps_target: *n_caps,
                    sample: None,
                    audit: &super::super::config::AuditConfig::default(),
                    training_step: 0,
                };
                KeyValueSource::Owned(disc.bootstrap(&dctx)?)
            }
            CapMatrixSource::Shared => {
                let keys = ctx.shared_cap_keys.ok_or_else(|| candle_core::Error::Msg(
                    "CapMatrixSource::Shared requested but no shared cap matrix declared on SubstrateBuilder".to_string()
                ))?.clone();
                let values = ctx
                    .shared_cap_values
                    .ok_or_else(|| {
                        candle_core::Error::Msg(
                            "CapMatrixSource::Shared requires a shared cap matrix WITH values"
                                .to_string(),
                        )
                    })?
                    .clone();
                KeyValueSource::Shared { keys, values }
            }
        };

        let w_q = candle_nn::linear_no_bias(d_model, d_model, vb.pp("w_q"))?;
        let w_o = candle_nn::linear_no_bias(d_model, d_model, vb.pp("w_o"))?;
        let d_k = d_model;
        let d_v = d_model;

        Ok(Self {
            w_q,
            w_o,
            kv,
            d_k,
            d_v,
            scale: 1.0 / (d_k as f64).sqrt(),
        })
    }

    fn keys_values(&self) -> (&Tensor, &Tensor) {
        match &self.kv {
            KeyValueSource::Owned(m) => {
                let v = m.values.as_ref().expect("CapMemory requires values");
                (&m.keys, v)
            }
            KeyValueSource::Shared { keys, values } => (keys, values),
        }
    }
}

impl Attention for CapMemoryAttention {
    fn forward(&self, x: &Tensor, _rope: &RoPE, _causal_mask: &Tensor) -> Result<Tensor> {
        let (b, t, _d_model) = x.dims3()?;

        // Q from input: [B, T, d_k]
        let q = self.w_q.forward(x)?;

        // Cap keys and values: [n_caps, d_k] and [n_caps, d_v]
        let (keys, values) = self.keys_values();

        // Compute scores: Q @ keys^T / sqrt(d_k)
        // q: [B, T, d_k], keys: [n_caps, d_k]
        // -> keys^T: [d_k, n_caps]
        let keys_t = keys.transpose(0, 1)?.contiguous()?;
        let scores = q.broadcast_matmul(&keys_t)?;
        let scores = (scores * self.scale)?;

        // softmax over n_caps dimension (no causal mask: caps don't have a
        // temporal position; every position can attend to every cap)
        let attn = candle_nn::ops::softmax_last_dim(&scores)?; // [B, T, n_caps]

        // Weighted blend of cap values: attn @ values -> [B, T, d_v]
        let out = attn.broadcast_matmul(values)?;

        // Output projection
        let out = self.w_o.forward(&out)?;
        let _ = (b, t);
        Ok(out)
    }
}
