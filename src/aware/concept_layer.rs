// Concept layer: window=1 caps placed in parallel to the prediction path.
// Keys are discovered (KMeans) and frozen; values are gradient-trained.
// Activations are sparse top-K; the value contribution is added to
// final_hidden before unembed.

use candle_core::{Device, Result, Tensor, D};
use candle_nn::VarBuilder;

use super::cap::{CapKind, CapMatrix};
use super::discover::{make_discovery, DiscoveryCtx, DiscoveryKind};

#[derive(Debug, Clone)]
pub struct ConceptConfig {
    pub n_concepts: usize,
    pub top_k: usize,
    /// Discovery strategy for keys. Typically KMeans (clusters from
    /// embedding sample). Other strategies are accepted for ablations.
    pub discovery: DiscoveryKind,
    /// Gradient-train concept values (recommended true)
    pub trainable_values: bool,
    /// Gradient-train concept keys (typically false - keys are frozen
    /// after discovery, which is what gives concepts stable identity)
    pub trainable_keys: bool,
}

impl Default for ConceptConfig {
    fn default() -> Self {
        Self {
            n_concepts: 128,
            top_k: 8,
            discovery: DiscoveryKind::KMeans,
            trainable_values: true,
            trainable_keys: false,
        }
    }
}

pub struct ConceptLayer {
    pub config: ConceptConfig,
    pub caps: CapMatrix, // keys: [n_concepts, d_emb], values: [n_concepts, d_model]
    pub d_emb: usize,
    pub d_model: usize,
}

impl ConceptLayer {
    pub fn new(
        config: ConceptConfig,
        d_emb: usize,
        d_model: usize,
        device: &Device,
        sample: Option<&Tensor>,
        _vb: VarBuilder,
    ) -> Result<Self> {
        let disc = make_discovery(config.discovery);
        let ctx = DiscoveryCtx {
            device,
            d_in: d_emb,
            d_out: Some(d_model),
            n_caps_target: config.n_concepts,
            sample,
            audit: &super::config::AuditConfig::default(),
            training_step: 0,
        };
        let mut caps = disc.bootstrap(&ctx)?;
        // For ConceptLayer:
        //   - Keys typically frozen after discovery (trainable_keys=false default)
        //   - Values typically trainable (trainable_values=true default)
        // The CapMatrix.trainable flag is coarse (either both or neither). To
        // get fine-grained control we manage gradient flow at the forward
        // level via .detach() on keys when trainable_keys=false.
        caps.trainable = config.trainable_values || config.trainable_keys;
        caps.kind = CapKind::Discovered;
        Ok(Self {
            config,
            caps,
            d_emb,
            d_model,
        })
    }

    /// Compute concept activations from input embeddings.
    /// Input shape: [B, T, d_emb] (per-token embeddings)
    /// Output shape: [B, T, n_concepts]
    pub fn activations(&self, embeddings: &Tensor) -> Result<Tensor> {
        let keys = if self.config.trainable_keys {
            self.caps.keys.clone()
        } else {
            // Detach keys from gradient flow (keys are frozen)
            self.caps.keys.detach()
        };
        let keys_t = keys.transpose(0, 1)?.contiguous()?;
        let acts = embeddings.broadcast_matmul(&keys_t)?;
        // Apply softmax over n_concepts to normalize (or could use raw
        // dot-products with top-K). Softmax gives smoother gradients.
        candle_nn::ops::softmax_last_dim(&acts)
    }

    /// Sparse top-K activation: keep only the top K values per position,
    /// zero out the rest. Returns a [B, T, n_concepts] tensor where
    /// each [B, T, :] row has at most K non-zero entries.
    pub fn sparse_top_k(&self, activations: &Tensor) -> Result<Tensor> {
        let k = self.config.top_k.min(self.config.n_concepts);
        if k == 0 || k == self.config.n_concepts {
            return Ok(activations.clone());
        }

        // candle doesn't have a built-in top-k mask. Implement via the
        // sort-and-threshold approach: get the K-th-largest value per
        // row, mask anything below it.
        //
        // For simplicity and gradient-flow we use the threshold-as-zero
        // approach: compute per-row K-th-largest as threshold, then
        // multiply activations by (acts >= threshold).
        let (b, t, n) = activations.dims3()?;
        let flat = activations.reshape((b * t, n))?;
        // sort_last_dim returns descending sort, then we take the K-th value
        let (sorted, _) = flat.sort_last_dim(false)?;
        let threshold = sorted.narrow(D::Minus1, k - 1, 1)?;
        // Compare each element to threshold; keep if >= threshold
        let mask = flat.broadcast_ge(&threshold)?;
        let mask = mask.to_dtype(flat.dtype())?;
        let masked = (flat * mask)?;
        masked.reshape((b, t, n))
    }

    /// Compute the concept contribution to the final hidden state.
    /// Input shape: [B, T, n_concepts] (sparse activations)
    /// Output shape: [B, T, d_model] (contribution to add to final_hidden)
    pub fn contribute(&self, activations: &Tensor) -> Result<Tensor> {
        let values = self.caps.values.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("ConceptLayer requires CapMatrix with values".to_string())
        })?;
        // [B, T, n_concepts] @ [n_concepts, d_model] -> [B, T, d_model]
        activations.broadcast_matmul(values)
    }

    /// Convenience: forward = sparse_top_k(activations) -> contribute
    pub fn forward(&self, embeddings: &Tensor) -> Result<(Tensor, Tensor)> {
        let acts = self.activations(embeddings)?;
        let sparse_acts = self.sparse_top_k(&acts)?;
        let contrib = self.contribute(&sparse_acts)?;
        Ok((contrib, sparse_acts))
    }

    /// Return the top-K concept IDs per position. For probing /
    /// interpretability. Returns [B, T, K] of u32 indices.
    pub fn top_k_indices(&self, activations: &Tensor) -> Result<Tensor> {
        let k = self.config.top_k.min(self.config.n_concepts);
        let (_b, _t, _n) = activations.dims3()?;
        let (_, indices) = activations.sort_last_dim(false)?;
        indices.narrow(D::Minus1, 0, k)
    }

    /// Cap stats for reporting (consistent with Substrate::cap_stats)
    pub fn n_concepts(&self) -> usize {
        self.caps.n_caps()
    }
}
