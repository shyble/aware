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
    pub n_heads: usize,
    pub d_head: usize,
    pub d_k: usize,
    pub d_v: usize,
    pub scale: f64,
}

impl CapMemoryAttention {
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
                "CapMemoryAttention: d_model {} not divisible by n_heads {}",
                d_model, n_heads
            )));
        }
        let d_head = d_model / n_heads;

        let kv = match source {
            CapMatrixSource::Local { n_caps } => {
                // Build a per-block local cap matrix. NoDiscovery reproduces
                // the paper's first-pass Xavier init; KMeans clusters the
                // per-token embedding sample threaded through BlockBuildCtx.
                let disc = make_discovery(discovery);
                let dctx = DiscoveryCtx {
                    device,
                    d_in: d_model,
                    d_out: Some(d_model),
                    n_caps_target: *n_caps,
                    sample: ctx.bootstrap_sample,
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

        Ok(Self {
            w_q,
            w_o,
            kv,
            n_heads,
            d_head,
            d_k: d_model,
            d_v: d_model,
            // Scale by the per-head dimension, matching multi-head standard
            // attention. At n_heads=1 this reduces to the original 1/sqrt(d_model).
            scale: 1.0 / (d_head as f64).sqrt(),
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
        let (b, t, d_model) = x.dims3()?;
        let h = self.n_heads;
        let dh = self.d_head;

        // Cap keys and values: [n_caps, d_model] each.
        let (keys, values) = self.keys_values();
        let n_caps = keys.dim(0)?;

        // Q from input, split into heads: [B, T, d_model] -> [B, h, T, dh]
        let q = self
            .w_q
            .forward(x)?
            .reshape((b, t, h, dh))?
            .transpose(1, 2)?
            .contiguous()?;

        // Caps split into the same head layout. Each head attends over the
        // same cap bank but through its own dh-wide slice.
        //   keys:   [n_caps, d_model] -> [1, h, dh, n_caps]  (already transposed)
        //   values: [n_caps, d_model] -> [1, h, n_caps, dh]
        let keys_h = keys
            .reshape((n_caps, h, dh))?
            .transpose(0, 1)?
            .contiguous()?
            .transpose(1, 2)?
            .contiguous()?
            .unsqueeze(0)?; // [1, h, dh, n_caps]
        let values_h = values
            .reshape((n_caps, h, dh))?
            .transpose(0, 1)?
            .contiguous()?
            .unsqueeze(0)?; // [1, h, n_caps, dh]

        // scores: [B, h, T, n_caps]
        let scores = q.broadcast_matmul(&keys_h)?;
        let scores = (scores * self.scale)?;

        // softmax over n_caps (no causal mask: caps have no temporal position,
        // so every position may attend to every cap)
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;

        // Blend cap values per head, then merge heads back: [B, T, d_model]
        let out = attn.broadcast_matmul(&values_h)?; // [B, h, T, dh]
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, d_model))?;

        self.w_o.forward(&out)
    }
}
