// StreamingFeeder: batches sampled from a tokenized corpus.
//
// Two backing modes:
//   - Memory: holds Vec<u32> of all tokens (good for small corpora).
//   - Disk:   reads raw u32 (little-endian) windows from a file on demand;
//             constant memory regardless of corpus size.
//
// Both expose the same next_batch API. Choose by construction: from_tokens
// for memory mode, from_file for disk mode.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};

// 64-bit LCG (Knuth/MMIX): full period over 2^64, golden-ratio increment.
// The high bits (>> 33) are used because LCG low bits have short period.
const LCG_MULTIPLIER: u64 = 6364136223846793005;
const LCG_INCREMENT: u64 = 1442695040888963407;
const DEFAULT_SEED: u64 = 0xC0FFEE_BABE;

enum Source {
    Memory(Vec<u32>),
    Disk { file: File, n_tokens: usize },
}

pub struct StreamingFeeder {
    source: Source,
    pub batch_size: usize,
    pub seq_len: usize,
    pub rng_state: u64,
}

impl StreamingFeeder {
    /// Construct from an in-memory token vector.
    pub fn from_tokens(tokens: Vec<u32>, batch_size: usize, seq_len: usize) -> Self {
        Self {
            source: Source::Memory(tokens),
            batch_size,
            seq_len,
            rng_state: DEFAULT_SEED,
        }
    }

    /// Construct from a tokenized binary file (raw u32, little-endian).
    /// The file is read on demand per batch; total memory stays at
    /// batch_size * (seq_len + 1) * 4 bytes regardless of corpus size.
    pub fn from_file(
        path: &Path,
        batch_size: usize,
        seq_len: usize,
    ) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let n_bytes = file.metadata()?.len() as usize;
        if n_bytes % 4 != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "token file {} length {} is not a multiple of 4 bytes",
                    path.display(),
                    n_bytes
                ),
            ));
        }
        Ok(Self {
            source: Source::Disk {
                file,
                n_tokens: n_bytes / 4,
            },
            batch_size,
            seq_len,
            rng_state: DEFAULT_SEED,
        })
    }

    /// Set the deterministic RNG seed used to sample windows. A seed of 0
    /// falls back to the default.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng_state = if seed == 0 { DEFAULT_SEED } else { seed };
        self
    }

    /// Total number of tokens in the source.
    pub fn n_tokens(&self) -> usize {
        match &self.source {
            Source::Memory(toks) => toks.len(),
            Source::Disk { n_tokens, .. } => *n_tokens,
        }
    }

    /// Sample a batch. Returns (input_tokens [B, T], target_tokens [B, T]).
    pub fn next_batch(&mut self, device: &Device) -> Result<(Tensor, Tensor)> {
        let row_len = self.seq_len + 1;
        let n_tokens = self.n_tokens();
        let max_start = n_tokens.saturating_sub(row_len + 1);
        if max_start == 0 {
            return Err(candle_core::Error::Msg(
                "token corpus too small for current seq_len".to_string(),
            ));
        }
        let mut ids: Vec<u32> = Vec::with_capacity(self.batch_size * row_len);
        for _ in 0..self.batch_size {
            self.rng_state = self
                .rng_state
                .wrapping_mul(LCG_MULTIPLIER)
                .wrapping_add(LCG_INCREMENT);
            let start = ((self.rng_state >> 33) as usize) % max_start;
            read_window(&mut self.source, start, row_len, &mut ids)
                .map_err(|e| candle_core::Error::Msg(format!("token read failed: {}", e)))?;
        }
        let full = Tensor::from_vec(ids, (self.batch_size, row_len), device)?;
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

fn read_window(
    source: &mut Source,
    start: usize,
    row_len: usize,
    out: &mut Vec<u32>,
) -> std::io::Result<()> {
    match source {
        Source::Memory(tokens) => {
            out.extend_from_slice(&tokens[start..start + row_len]);
            Ok(())
        }
        Source::Disk { file, .. } => {
            file.seek(SeekFrom::Start((start * 4) as u64))?;
            let mut buf = vec![0u8; row_len * 4];
            file.read_exact(&mut buf)?;
            for chunk in buf.chunks_exact(4) {
                let token = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.push(token);
            }
            Ok(())
        }
    }
}

