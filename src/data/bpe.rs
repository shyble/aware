// BPE Tokenizer: byte-pair encoding for subword tokenization.
//
// Train on text → encode/decode. No external dependencies.
// Small vocab (256-512) bridges character-level and word-level.
//
// Usage:
//   let bpe = BPETokenizer::train(text, 192);
//   let tokens = bpe.encode("Alice was");
//   let text = bpe.decode(&tokens);

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
pub struct BPETokenizer {
    merges: Vec<(usize, usize, usize)>,
    vocab: Vec<Vec<u8>>,
}

impl BPETokenizer {
    /// Train BPE tokenizer with N merges.
    pub fn train(text: &str, n_merges: usize) -> Self {
        let bytes = text.as_bytes();
        let mut tokens: Vec<usize> = bytes.iter().map(|&b| b as usize).collect();
        let mut vocab: Vec<Vec<u8>> = (0..=255).map(|b| vec![b as u8]).collect();
        let mut merges = Vec::new();

        for _ in 0..n_merges {
            let mut pairs: HashMap<(usize, usize), usize> = HashMap::new();
            for w in tokens.windows(2) {
                *pairs.entry((w[0], w[1])).or_default() += 1;
            }
            if pairs.is_empty() {
                break;
            }

            // Pick the pair with the highest count. Break ties deterministically
            // by smallest pair tuple — Rust's HashMap iteration order is
            // process-randomized, so without this tiebreak BPE would produce
            // different vocabs across runs on the same input, scrambling token
            // IDs relative to a previously saved brain's embeddings.
            let best = {
                let mut best_pair = (0usize, 0usize);
                let mut best_count = 0usize;
                let mut found = false;
                for (&pair, &count) in &pairs {
                    if !found || count > best_count || (count == best_count && pair < best_pair) {
                        best_pair = pair;
                        best_count = count;
                        found = true;
                    }
                }
                best_pair
            };

            let new_id = vocab.len();
            let mut new_bytes = vocab[best.0].clone();
            new_bytes.extend_from_slice(&vocab[best.1]);
            vocab.push(new_bytes);
            merges.push((best.0, best.1, new_id));

            let mut new_tokens = Vec::with_capacity(tokens.len());
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len() && tokens[i] == best.0 && tokens[i + 1] == best.1 {
                    new_tokens.push(new_id);
                    i += 2;
                } else {
                    new_tokens.push(tokens[i]);
                    i += 1;
                }
            }
            tokens = new_tokens;
        }

        BPETokenizer { merges, vocab }
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens: Vec<usize> = text.as_bytes().iter().map(|&b| b as usize).collect();
        for &(a, b, new_id) in &self.merges {
            let mut new = Vec::with_capacity(tokens.len());
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len() && tokens[i] == a && tokens[i + 1] == b {
                    new.push(new_id);
                    i += 2;
                } else {
                    new.push(tokens[i]);
                    i += 1;
                }
            }
            tokens = new;
        }
        tokens
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[usize]) -> String {
        let bytes: Vec<u8> = tokens
            .iter()
            .flat_map(|&t| self.vocab.get(t).cloned().unwrap_or_default())
            .collect();
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Decode a single token to string.
    pub fn decode_token(&self, token: usize) -> String {
        self.vocab
            .get(token)
            .map(|b| String::from_utf8_lossy(b).to_string())
            .unwrap_or_else(|| format!("<{}>", token))
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    pub fn n_merges(&self) -> usize {
        self.merges.len()
    }

    /// Extend the BPE with `n_extra` additional merges learned from new
    /// text, *without* breaking any existing token IDs. Use this when
    /// continual learning encounters new content with vocabulary that
    /// would otherwise tokenize as long byte sequences.
    ///
    /// Guarantees:
    /// - All existing token IDs (0..self.vocab_size()) keep their byte
    ///   meaning. Old encoded text decodes the same.
    /// - New merges get IDs strictly above the current vocab size.
    /// - Determinism: same input + same n_extra → same merges (smallest-
    ///   pair tiebreak, matching `train()`).
    ///
    /// Returns the new vocab size.
    pub fn extend(&mut self, new_text: &str, n_extra: usize) -> usize {
        if n_extra == 0 || new_text.is_empty() {
            return self.vocab.len();
        }

        // Tokenize new_text with current merges so we count pairs in the
        // *post-merge* token stream. Pairs of bytes that the existing BPE
        // already merges won't show up — we only learn pairs that aren't
        // already covered.
        let mut tokens = self.encode(new_text);

        for _ in 0..n_extra {
            let mut pairs: HashMap<(usize, usize), usize> = HashMap::new();
            for w in tokens.windows(2) {
                *pairs.entry((w[0], w[1])).or_default() += 1;
            }
            if pairs.is_empty() {
                break;
            }

            let best = {
                let mut best_pair = (0usize, 0usize);
                let mut best_count = 0usize;
                let mut found = false;
                for (&pair, &count) in &pairs {
                    if !found || count > best_count || (count == best_count && pair < best_pair) {
                        best_pair = pair;
                        best_count = count;
                        found = true;
                    }
                }
                best_pair
            };

            let new_id = self.vocab.len();
            let mut new_bytes = self.vocab[best.0].clone();
            new_bytes.extend_from_slice(&self.vocab[best.1]);
            self.vocab.push(new_bytes);
            self.merges.push((best.0, best.1, new_id));

            // Re-tokenize new_text via the new merge for the next pair count.
            let mut new_tokens = Vec::with_capacity(tokens.len());
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len() && tokens[i] == best.0 && tokens[i + 1] == best.1 {
                    new_tokens.push(new_id);
                    i += 2;
                } else {
                    new_tokens.push(tokens[i]);
                    i += 1;
                }
            }
            tokens = new_tokens;
        }

        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_roundtrip() {
        let text = "alice alice alice was was very tired";
        let bpe = BPETokenizer::train(text, 10);
        let tokens = bpe.encode(text);
        let decoded = bpe.decode(&tokens);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_bpe_compression() {
        let text = "aaaa bbbb aaaa bbbb aaaa";
        let bpe = BPETokenizer::train(text, 5);
        let tokens = bpe.encode(text);
        // Should be shorter than original bytes due to merges
        assert!(tokens.len() < text.len());
    }

    /// Resume safety: training BPE twice on the same input must produce
    /// identical merges and vocab. Without the deterministic tiebreak in
    /// train(), HashMap iteration order would silently shift token IDs across
    /// runs, scrambling a saved brain's embeddings on resume.
    #[test]
    fn test_bpe_train_is_deterministic() {
        // Pathological input designed to create many tied pair counts so the
        // tiebreak is exercised on most merge steps.
        let text = "abababab cdcdcdcd efefefef ghghghgh ababcdef gh ab cd ef gh";
        let n_runs = 8;
        let first = BPETokenizer::train(text, 20);
        for _ in 1..n_runs {
            let other = BPETokenizer::train(text, 20);
            assert_eq!(
                first.merges, other.merges,
                "BPE merges differ across runs — tiebreak is not deterministic"
            );
            assert_eq!(first.vocab, other.vocab, "BPE vocab differs across runs");
        }
    }

    #[test]
    fn test_bpe_vocab_grows() {
        let text = "the cat sat on the mat";
        let bpe = BPETokenizer::train(text, 3);
        assert_eq!(bpe.vocab_size(), 256 + 3); // base + 3 merges
    }

    /// Extend semantics: extending preserves the *byte meaning* of every
    /// existing token ID. New merges may apply to old text and produce a
    /// shorter encoding, but: (1) old IDs decode to the same bytes, (2)
    /// the old merge prefix is unchanged, (3) decoding old encodings still
    /// roundtrips to the original text, (4) re-encoded old text still
    /// decodes to the same original text.
    ///
    /// This is the contract continual learning needs: the brain's old
    /// embedding row for token N still represents "the byte sequence
    /// vocab[N] meant before". New token IDs are strictly additive.
    #[test]
    fn test_bpe_extend_preserves_old_token_byte_meanings() {
        let original = "the cat sat on the mat";
        let pre_bpe_snapshot;
        let pre_tokens;
        let pre_vocab_size;
        let pre_merges;
        {
            let bpe = BPETokenizer::train(original, 5);
            pre_tokens = bpe.encode(original);
            pre_vocab_size = bpe.vocab_size();
            pre_merges = bpe.merges.clone();
            pre_bpe_snapshot = bpe;
        }

        let mut bpe = BPETokenizer::train(original, 5);
        let new_text = "the dog ran in the park and the cat watched";
        let extra = 5;
        let new_size = bpe.extend(new_text, extra);

        // 1. Vocab grew (with cap of +extra).
        assert!(new_size > pre_vocab_size);
        assert!(new_size <= pre_vocab_size + extra);

        // 2. Old merges are still there in the same order. Extension is
        //    strictly append-only on the merges vec.
        assert_eq!(
            &bpe.merges[..pre_merges.len()],
            &pre_merges[..],
            "extend must not modify old merges"
        );

        // 3. Old token IDs decode to the SAME bytes.
        //    This is the load-bearing invariant: brain.embeddings[id]
        //    still represents the same input pattern.
        for id in 0..pre_vocab_size {
            assert_eq!(
                bpe.vocab[id], pre_bpe_snapshot.vocab[id],
                "byte meaning of token {} changed",
                id
            );
        }

        // 4. The old encoding (pre_tokens) still decodes to the original
        //    text — old saved sequences remain interpretable.
        assert_eq!(pre_bpe_snapshot.decode(&pre_tokens), original);
        // And the extended BPE can also decode them (since old IDs survived).
        assert_eq!(bpe.decode(&pre_tokens), original);

        // 5. Re-encoding the original with extended BPE may produce fewer
        //    tokens (new merges fire) but still decodes to the same text.
        let post_tokens = bpe.encode(original);
        assert_eq!(bpe.decode(&post_tokens), original);

        // 6. New text uses at least one of the new IDs (sanity that
        //    extension learned something useful).
        let new_text_tokens = bpe.encode(new_text);
        assert!(
            new_text_tokens.iter().any(|&t| t >= pre_vocab_size),
            "extended BPE never used a new ID on the new text"
        );
    }

    #[test]
    fn test_bpe_extend_is_deterministic() {
        let original = "the cat sat on the mat";
        let new_text = "the dog ran the cat saw the bird";
        let mut a = BPETokenizer::train(original, 4);
        let mut b = BPETokenizer::train(original, 4);
        a.extend(new_text, 6);
        b.extend(new_text, 6);
        assert_eq!(a.merges, b.merges);
        assert_eq!(a.vocab, b.vocab);
    }
}

/// Load a previously-saved BPE tokenizer from a directory containing `bpe.bin`.
/// Matches the on-disk format used by the AWARE training pipeline.
pub fn load_bpe(dir: &str) -> std::io::Result<BPETokenizer> {
    let bpe_bytes = std::fs::read(format!("{}/bpe.bin", dir))?;
    bincode::deserialize(&bpe_bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}
