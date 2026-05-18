// Training infrastructure: Trainer, Feeder, Loss, Optimizer.

pub mod feeder;
pub mod loss;
pub mod optimizer;
pub mod trainer;

pub use feeder::StreamingFeeder;
pub use loss::LossKind;
pub use optimizer::OptimizerKind;
pub use trainer::Trainer;
