use candle_core::{Result, Tensor};

#[derive(Debug, Clone, Copy)]
pub enum LossKind {
    CrossEntropy,
}

impl LossKind {
    /// logits: [N, V], targets: [N]. Returns scalar loss.
    pub fn compute(&self, logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
        match self {
            LossKind::CrossEntropy => candle_nn::loss::cross_entropy(logits, targets),
        }
    }
}
