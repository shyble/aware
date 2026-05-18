// Trainer: orchestrates one training session.
//
// Caller hands over a built Substrate + feeder + optimizer + loss
// settings. Trainer drives the loop: forward, loss, backward, optimizer
// step, periodic eval, periodic audit (if cap kind supports it).

use std::time::Instant;

use candle_core::Result;
use candle_nn::Optimizer;

use super::super::substrate::Substrate;
use super::feeder::StreamingFeeder;
use super::loss::LossKind;
use super::optimizer::OptimizerKind;

pub struct Trainer {
    pub model: Substrate,
    pub feeder: StreamingFeeder,
    pub optimizer_kind: OptimizerKind,
    pub loss_kind: LossKind,
    pub eval_every_steps: usize,
    pub audit_every_steps: usize,
    pub save_path: Option<String>,
    pub verbose: bool,
}

impl Trainer {
    pub fn new(model: Substrate, feeder: StreamingFeeder) -> Self {
        Self {
            model,
            feeder,
            optimizer_kind: OptimizerKind::AdamW { lr: 3e-4 },
            loss_kind: LossKind::CrossEntropy,
            eval_every_steps: 100,
            audit_every_steps: 500,
            save_path: None,
            verbose: true,
        }
    }

    pub fn with_optimizer(mut self, k: OptimizerKind) -> Self {
        self.optimizer_kind = k;
        self
    }
    pub fn with_loss(mut self, k: LossKind) -> Self {
        self.loss_kind = k;
        self
    }
    pub fn with_eval_every(mut self, n: usize) -> Self {
        self.eval_every_steps = n;
        self
    }
    pub fn with_audit_every(mut self, n: usize) -> Self {
        self.audit_every_steps = n;
        self
    }
    pub fn with_save_path<S: Into<String>>(mut self, p: S) -> Self {
        self.save_path = Some(p.into());
        self
    }
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    pub fn train(&mut self, total_steps: usize) -> Result<TrainReport> {
        let device = self.model.device.clone();
        let mut opt = self.optimizer_kind.build(&self.model.varmap)?;
        let start = Instant::now();

        if self.verbose {
            println!("[trainer] params: {}", self.model.n_params());
            println!("[trainer] training {} steps", total_steps);
        }

        let mut last_loss = 0.0f32;
        for step in 1..=total_steps {
            let (inp, tgt) = self.feeder.next_batch(&device)?;
            let logits = self.model.forward(&inp)?;
            let (b, t, v) = logits.dims3()?;
            let logits_flat = logits.reshape((b * t, v))?;
            let tgt_flat = tgt.reshape((b * t,))?;
            let loss = self.loss_kind.compute(&logits_flat, &tgt_flat)?;
            opt.backward_step(&loss)?;
            last_loss = loss.to_scalar::<f32>()?;

            if self.verbose && step % self.eval_every_steps == 0 {
                let elapsed = start.elapsed().as_secs_f64();
                println!(
                    "  step={:>5}  loss={:.3} (ppl={:.1})  elapsed={:.1}s",
                    step,
                    last_loss,
                    (last_loss as f64).exp(),
                    elapsed,
                );
                if let Some(p) = &self.save_path {
                    let _ = self.model.save_checkpoint(p);
                }
            }

            // Audit hook: future - call cap layer's discovery.step() here
            // at audit_every_steps cadence.
            let _ = self.audit_every_steps;
        }

        Ok(TrainReport {
            final_loss: last_loss,
            total_seconds: start.elapsed().as_secs_f64(),
            total_steps,
        })
    }
}

pub struct TrainReport {
    pub final_loss: f32,
    pub total_seconds: f64,
    pub total_steps: usize,
}

impl std::fmt::Display for TrainReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TrainReport {{ steps: {}, final_loss: {:.3}, perplexity: {:.2}, time: {:.1}s }}",
            self.total_steps,
            self.final_loss,
            (self.final_loss as f64).exp(),
            self.total_seconds,
        )
    }
}
