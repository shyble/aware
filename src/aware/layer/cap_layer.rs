// CapLayer: caps fire on a windowed view of embeddings; activations are
// GELU'd and projected to d_model. For window>1, prior positions are
// zero-padded (causal).

use candle_core::{Module, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};

use super::super::cap::CapMatrix;
use super::super::config::CapConfig;
use super::super::discover::{make_discovery, DiscoveryCtx};

pub struct CapLayer {
    /// Trainable offset on the discovered keys, present only when
    /// `gradient_train` is set. `None` reproduces the frozen behaviour
    /// exactly.
    pub key_delta: Option<Tensor>,
    pub config: CapConfig,
    pub caps: CapMatrix,
    /// Projection from [n_caps] -> [d_model].
    pub w_proj: Linear,
    /// Cap-input dim: d_emb × cap_window. Caps' keys are this shape.
    pub d_in_per_window: usize,
}

impl CapLayer {
    pub fn new(
        config: CapConfig,
        d_emb: usize,
        d_model: usize,
        device: &candle_core::Device,
        sample: Option<&candle_core::Tensor>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let window = config.cap_window.max(1);
        let d_in_per_window = d_emb * window;

        let disc = make_discovery(config.discovery);
        let ctx = DiscoveryCtx {
            device,
            d_in: d_in_per_window,
            d_out: None,
            n_caps_target: config.n_caps_target,
            sample, // pass corpus sample for real KMeans
            audit: &config.audit,
            training_step: 0,
        };
        let caps = disc.bootstrap(&ctx)?;

        // Adaptable keys, as a zero-initialised delta on the discovered
        // base rather than a replacement for it.
        //
        // `CapMatrix.keys` is a plain Tensor outside the VarMap, so it
        // takes no gradient however `CapKind` is set -- `Gradient` and
        // `Hybrid` were declared but never wired. Registering a delta is
        // the smallest change that makes caps adaptable while keeping
        // what the frozen keys are FOR: discovery still sets where a cap
        // starts, identity is unchanged, the audit lifecycle still reads
        // the base, and zero-init means the model begins bit-identical
        // to the frozen version.
        //
        // This matters most above layer 0. A deeper layer's keys are
        // discovered over representations that do not exist until the
        // model is trained, so freezing them pins the layer to a guess;
        // the delta lets it follow what the layer below becomes.
        let key_delta = if config.adapt_keys {
            Some(vb.get_with_hints(
                caps.keys.shape(),
                "key_delta",
                candle_nn::Init::Const(0.0),
            )?)
        } else {
            None
        };

        let w_proj = candle_nn::linear_no_bias(config.n_caps_target, d_model, vb.pp("w_proj"))?;

        Ok(Self {
            config,
            caps,
            key_delta,
            w_proj,
            d_in_per_window,
        })
    }

    /// Build cap-input from embeddings: [B, T, d_emb] -> [B, T, window × d_emb].
    /// Past tokens before position 0 are zero-padded (causal-safe).
    fn build_windowed_input(&self, embeddings: &Tensor) -> Result<Tensor> {
        let w = self.config.cap_window.max(1);
        if w == 1 {
            return Ok(embeddings.clone());
        }
        let (b, t, d_emb) = embeddings.dims3()?;
        let device = embeddings.device();

        // Pad with zeros at the start: prepend (w - 1) zero positions
        // so that position 0 in the original gets a full window.
        let pad = Tensor::zeros((b, w - 1, d_emb), embeddings.dtype(), device)?;
        let padded = Tensor::cat(&[&pad, embeddings], 1)?; // [B, T+w-1, d_emb]

        // For each output position t in 0..T, gather padded[t..t+w]
        // and concatenate along the last dim.
        // We achieve this with `narrow` per offset and `cat` along last dim.
        let mut pieces: Vec<Tensor> = Vec::with_capacity(w);
        for off in 0..w {
            // padded[:, off..off+T, :]  shape [B, T, d_emb]
            let slice = padded.narrow(1, off, t)?;
            pieces.push(slice);
        }
        // Concatenate along the last (feature) dimension: [B, T, w * d_emb]
        let refs: Vec<&Tensor> = pieces.iter().collect();
        Tensor::cat(&refs, D::Minus1)
    }

    /// Cap activations, using the adapted keys when a delta is present.
    fn fire_adapted(&self, cap_input: &Tensor) -> Result<Tensor> {
        match &self.key_delta {
            None => self.caps.fire(cap_input),
            Some(delta) => {
                let keys = (self.caps.keys.detach() + delta)?;
                let dims = cap_input.dims();
                let n = dims.len();
                let d_in = dims[n - 1];
                let total: usize = dims[..n - 1].iter().product();
                let flat = cap_input.reshape((total, d_in))?.contiguous()?;
                let out = flat.matmul(&keys.transpose(0, 1)?.contiguous()?)?;
                let mut shape = dims[..n - 1].to_vec();
                shape.push(keys.dims()[0]);
                out.reshape(shape)
            }
        }
    }

    pub fn forward(&self, embeddings: &Tensor) -> Result<Tensor> {
        let cap_input = self.build_windowed_input(embeddings)?; // [B, T, w*d_emb]
        let cap_acts = self.fire_adapted(&cap_input)?; // [B, T, n_caps]
        let cap_acts = cap_acts.gelu()?;
        self.w_proj.forward(&cap_acts) // [B, T, d_model]
    }

    /// Expose cap activations (for cap-pair attention to consume).
    pub fn cap_activations(&self, embeddings: &Tensor) -> Result<Tensor> {
        let cap_input = self.build_windowed_input(embeddings)?;
        let cap_acts = self.fire_adapted(&cap_input)?;
        cap_acts.gelu()
    }
}
