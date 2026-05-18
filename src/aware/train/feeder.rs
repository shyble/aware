// StreamingFeeder: chunk-iterative batches from a tokenized corpus.
//
// Holds a Vec<u32> of all tokens in memory; samples random (B, T+1) windows
// per batch. Memory-bounded: just the token vec + the batch tensor at any
// time.

use candle_core::{DType, Device, Result, Tensor};

// 64-bit LCG (Knuth/MMIX): full period over 2^64, golden-ratio increment.
// The high bits (>> 33) are used because LCG low bits have short period.
const LCG_MULTIPLIER: u64 = 6364136223846793005;
const LCG_INCREMENT: u64 = 1442695040888963407;
const DEFAULT_SEED: u64 = 0xC0FFEE_BABE;

pub struct StreamingFeeder {
    pub tokens: Vec<u32>,
    pub batch_size: usize,
    pub seq_len: usize,
    pub rng_state: u64,
}

impl StreamingFeeder {
    pub fn from_tokens(tokens: Vec<u32>, batch_size: usize, seq_len: usize) -> Self {
        Self {
            tokens,
            batch_size,
            seq_len,
            rng_state: DEFAULT_SEED,
        }
    }

    /// Set the deterministic RNG seed used to sample windows. A seed of 0
    /// falls back to the default.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng_state = if seed == 0 { DEFAULT_SEED } else { seed };
        self
    }

    /// Sample a batch. Returns (input_tokens [B, T], target_tokens [B, T]).
    pub fn next_batch(&mut self, device: &Device) -> Result<(Tensor, Tensor)> {
        let mut ids: Vec<u32> = Vec::with_capacity(self.batch_size * (self.seq_len + 1));
        let max_start = self.tokens.len().saturating_sub(self.seq_len + 2);
        if max_start == 0 {
            return Err(candle_core::Error::Msg(
                "token corpus too small for current seq_len".to_string(),
            ));
        }
        for _ in 0..self.batch_size {
            self.rng_state = self
                .rng_state
                .wrapping_mul(LCG_MULTIPLIER)
                .wrapping_add(LCG_INCREMENT);
            let start = ((self.rng_state >> 33) as usize) % max_start;
            ids.extend_from_slice(&self.tokens[start..start + self.seq_len + 1]);
        }
        let full = Tensor::from_vec(ids, (self.batch_size, self.seq_len + 1), device)?;
        let inp = full
            .narrow(1, 0, self.seq_len)?
            .contiguous()?
            .to_dtype(DType::U32)?;
        let tgt = full
            .narrow(1, 1, self.seq_len)?
            .contiguous()?
            .to_dtype(DType::U32)?;
        Ok((inp, tgt))
    }
}
