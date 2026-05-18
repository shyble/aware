// CapMatrix: packed (n_caps × d_in) keys + optional (n_caps × d_out)
// values + stable u64 ids + per-cap metadata.

use candle_core::{Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use super::CapKind;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapMeta {
    pub id: u64,
    pub activation_ema: f64,
    pub age: u64,
    pub frozen: bool,
    pub dormant: bool,
}

impl CapMeta {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            activation_ema: 0.0,
            age: 0,
            frozen: false,
            dormant: false,
        }
    }
}

/// Packed cap storage. Keys are always present; values are present for
/// cap variants that act as KV memory (e.g., cap-memory attention).
pub struct CapMatrix {
    pub kind: CapKind,
    pub d_in: usize,
    pub d_out: Option<usize>,
    pub keys: Tensor,           // [n_caps, d_in]
    pub values: Option<Tensor>, // [n_caps, d_out] if d_out set
    pub ids: Vec<u64>,
    pub metadata: Vec<CapMeta>,
    pub trainable: bool, // whether keys/values are gradient-updated
    pub device: Device,
}

impl CapMatrix {
    /// Build a zero-initialized matrix shell. Actual values come from a
    /// Discovery::bootstrap() call.
    pub fn empty(
        kind: CapKind,
        d_in: usize,
        d_out: Option<usize>,
        n_caps: usize,
        trainable: bool,
        device: Device,
    ) -> Result<Self> {
        let keys = Tensor::zeros((n_caps, d_in), candle_core::DType::F32, &device)?;
        let values = match d_out {
            Some(d) => Some(Tensor::zeros(
                (n_caps, d),
                candle_core::DType::F32,
                &device,
            )?),
            None => None,
        };
        let ids: Vec<u64> = (1..=n_caps as u64).collect();
        let metadata: Vec<CapMeta> = ids.iter().map(|&id| CapMeta::new(id)).collect();
        Ok(Self {
            kind,
            d_in,
            d_out,
            keys,
            values,
            ids,
            metadata,
            trainable,
            device,
        })
    }

    pub fn n_caps(&self) -> usize {
        self.ids.len()
    }

    /// Replace the row at slot `idx` with a new (id, key, optional value).
    /// Used by audit: old cap retires, new cap takes its slot.
    pub fn replace_row(
        &mut self,
        idx: usize,
        new_id: u64,
        new_key: &[f32],
        new_value: Option<&[f32]>,
    ) -> Result<()> {
        if idx >= self.n_caps() {
            return Ok(());
        }
        if new_key.len() != self.d_in {
            return Err(candle_core::Error::Msg(format!(
                "key dim {} != d_in {}",
                new_key.len(),
                self.d_in
            )));
        }
        // Update tensor row in-place via slice assignment.
        let key_t = Tensor::from_vec(new_key.to_vec(), (1, self.d_in), &self.device)?;
        self.keys = self
            .keys
            .slice_assign(&[idx..idx + 1, 0..self.d_in], &key_t)?;

        if let (Some(values), Some(new_v)) = (&mut self.values, new_value) {
            let d_out = self.d_out.unwrap_or(0);
            if new_v.len() != d_out {
                return Err(candle_core::Error::Msg(format!(
                    "value dim {} != d_out {}",
                    new_v.len(),
                    d_out
                )));
            }
            let val_t = Tensor::from_vec(new_v.to_vec(), (1, d_out), &self.device)?;
            *values = values.slice_assign(&[idx..idx + 1, 0..d_out], &val_t)?;
        }

        self.ids[idx] = new_id;
        self.metadata[idx] = CapMeta::new(new_id);
        Ok(())
    }

    /// Append new rows (for growth). Returns the slot indices of new rows.
    pub fn append_rows(
        &mut self,
        new_ids: &[u64],
        new_keys: &Tensor,
        new_values: Option<&Tensor>,
    ) -> Result<Vec<usize>> {
        let n_new = new_ids.len();
        if n_new == 0 {
            return Ok(vec![]);
        }
        let start_idx = self.n_caps();

        self.keys = Tensor::cat(&[&self.keys, new_keys], 0)?;
        if let (Some(values), Some(nv)) = (&mut self.values, new_values) {
            *values = Tensor::cat(&[&*values, nv], 0)?;
        }
        for &id in new_ids {
            self.ids.push(id);
            self.metadata.push(CapMeta::new(id));
        }
        Ok((start_idx..start_idx + n_new).collect())
    }

    /// Fire on input via dot product. input: [..., d_in].
    /// Returns activations of shape [..., n_caps].
    pub fn fire(&self, input: &Tensor) -> Result<Tensor> {
        // input @ keys.T -> [..., n_caps]
        let keys_t = self.keys.transpose(0, 1)?.contiguous()?;
        input.broadcast_matmul(&keys_t)
    }
}
