use candle_core::Result;
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{Optimizer, VarMap};

#[derive(Debug, Clone, Copy)]
pub enum OptimizerKind {
    AdamW { lr: f64 },
}

impl OptimizerKind {
    pub fn build(&self, varmap: &VarMap) -> Result<AdamW> {
        match self {
            OptimizerKind::AdamW { lr } => {
                let params = ParamsAdamW {
                    lr: *lr,
                    ..Default::default()
                };
                AdamW::new(varmap.all_vars(), params)
            }
        }
    }
}
