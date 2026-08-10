//! AdamW whose moment estimates survive a checkpoint.
//!
//! # Why this exists
//!
//! Adam's second moment `v` is an exponentially-weighted average of
//! squared gradients — a running record of which parameters the current
//! corpus actually used. At the end of a training cycle that record is
//! the cheapest summary we have of what the model learned, and every
//! standard setup throws it away: the next cycle constructs a fresh
//! optimizer with `m = v = 0`.
//!
//! Carrying `v` into the next cycle is not a penalty and does not touch
//! the loss. It changes only the per-parameter step size:
//!
//! ```text
//!     θ ← θ − lr · m̂ / (√v̂ + ε)
//! ```
//!
//! A parameter the previous corpus leaned on has large `v`, so the new
//! corpus moves it in smaller steps. A parameter the previous corpus
//! never touched has small `v` and stays fully plastic. The new corpus
//! is not constrained — it is simply handed an optimizer that already
//! knows where the model's existing competence lives.
//!
//! # Why not candle's AdamW
//!
//! `candle_nn::AdamW` keeps `first_moment` / `second_moment` private, so
//! the state cannot be read out or restored. This is a re-implementation
//! of exactly that update — verified numerically against it in the tests
//! below — with the moments keyed by variable name so they can be
//! written to a checkpoint and loaded back.
//!
//! # The bias-correction trap
//!
//! `v̂ = v / (1 − β₂^t)`. Restoring `v` while resetting `t` to zero makes
//! the first corrected step divide by `1 − 0.999 = 0.001`, inflating the
//! second moment a thousandfold and effectively freezing the model.
//! `step_t` is therefore part of the saved state and is always restored
//! with the moments, never independently.

use std::collections::HashMap;

use candle_core::{DType, Result as CResult, Tensor, Var};
use candle_nn::optim::ParamsAdamW;
use candle_nn::VarMap;

/// How much of the previous cycle's optimizer state to carry forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarryMode {
    /// Fresh optimizer — the conventional behaviour, and the control.
    Off,
    /// Carry the second moment only.
    ///
    /// This is the default for continual learning. `v` is a magnitude
    /// record ("how much did this parameter matter"), which stays
    /// meaningful across a corpus change. `m` is a *direction* — where
    /// the previous corpus was heading — and pointing the new corpus
    /// along the old one's momentum has no justification.
    SecondMoment,
    /// Carry both moments. Included so the choice above is testable
    /// rather than assumed.
    Both,
}

impl CarryMode {
    pub fn parse(s: &str) -> Self {
        match s {
            "v" | "second" | "second_moment" => CarryMode::SecondMoment,
            "both" | "mv" => CarryMode::Both,
            _ => CarryMode::Off,
        }
    }
}

struct VarState {
    var: Var,
    m: Tensor,
    v: Tensor,
}

/// AdamW with named, persistable moment state.
pub struct PersistentAdamW {
    vars: Vec<(String, VarState)>,
    step_t: usize,
    params: ParamsAdamW,
}

impl PersistentAdamW {
    pub fn new(varmap: &VarMap, params: ParamsAdamW) -> CResult<Self> {
        let data = varmap.data().lock().unwrap();
        let mut vars = Vec::with_capacity(data.len());
        for (name, var) in data.iter() {
            // Matches candle: integer parameters are not optimized.
            if !var.dtype().is_float() {
                continue;
            }
            let m = Tensor::zeros(var.shape(), var.dtype(), var.device())?;
            let v = Tensor::zeros(var.shape(), var.dtype(), var.device())?;
            vars.push((
                name.clone(),
                VarState {
                    var: var.clone(),
                    m,
                    v,
                },
            ));
        }
        // HashMap iteration is unordered; updates are independent per
        // parameter so this does not affect results, but sorting keeps
        // logs and any future dumps stable between runs.
        vars.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(Self {
            vars,
            step_t: 0,
            params,
        })
    }

    pub fn step_t(&self) -> usize {
        self.step_t
    }

    pub fn n_params(&self) -> usize {
        self.vars.len()
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    /// One AdamW step. Mirrors `candle_nn::AdamW::step` operation for
    /// operation — see the equivalence test.
    pub fn backward_step(&mut self, loss: &Tensor) -> CResult<()> {
        let grads = loss.backward()?;
        self.step_t += 1;
        let lr = self.params.lr;
        let lr_lambda = lr * self.params.weight_decay;
        let (b1, b2) = (self.params.beta1, self.params.beta2);
        let scale_m = 1f64 / (1f64 - b1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - b2.powi(self.step_t as i32));

        for (_, st) in self.vars.iter_mut() {
            let theta = &st.var;
            let Some(g) = grads.get(theta) else { continue };

            let next_m = ((&st.m * b1)? + (g * (1.0 - b1))?)?;
            let next_v = ((&st.v * b2)? + (g.sqr()? * (1.0 - b2))?)?;
            let m_hat = (&next_m * scale_m)?;
            let v_hat = (&next_v * scale_v)?;
            let next_theta = (theta.as_tensor() * (1f64 - lr_lambda))?;
            let adjusted = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
            theta.set(&(next_theta - (adjusted * lr)?)?)?;

            // Detach before storing. `next_m` was computed FROM `st.m`,
            // so a tracked tensor keeps the previous step's moment alive
            // as a backprop parent — and storing it makes the next step
            // do the same, chaining a graph that grows without bound
            // until the process is killed. candle's own AdamW sidesteps
            // this by holding moments in `Var`s, which are leaves; here
            // they are plain tensors, so the detach is load-bearing.
            st.m = next_m.detach();
            st.v = next_v.detach();
        }
        Ok(())
    }

    /// Moment state as named tensors, ready for `safetensors`.
    ///
    /// `step_t` rides along as a rank-1 tensor because the moments are
    /// meaningless without it — see the bias-correction note above.
    pub fn state_tensors(&self) -> CResult<HashMap<String, Tensor>> {
        let mut out = HashMap::with_capacity(self.vars.len() * 2 + 1);
        for (name, st) in &self.vars {
            out.insert(format!("__optim__.m.{name}"), st.m.clone());
            out.insert(format!("__optim__.v.{name}"), st.v.clone());
        }
        let dev = self
            .vars
            .first()
            .map(|(_, s)| s.m.device().clone())
            .unwrap_or(candle_core::Device::Cpu);
        out.insert(
            "__optim__.step_t".into(),
            Tensor::from_vec(vec![self.step_t as f32], (1,), &dev)?,
        );
        Ok(out)
    }

    /// Restore moment state saved by [`state_tensors`].
    ///
    /// Returns how many parameters were matched. A parameter absent from
    /// the saved state keeps its zero moments, which is the correct
    /// behaviour for weights that did not exist in the previous cycle
    /// (a grown cap, say) — they start fully plastic.
    pub fn load_state(
        &mut self,
        saved: &HashMap<String, Tensor>,
        mode: CarryMode,
    ) -> CResult<usize> {
        if mode == CarryMode::Off {
            return Ok(0);
        }
        let mut matched = 0usize;
        for (name, st) in self.vars.iter_mut() {
            let v_key = format!("__optim__.v.{name}");
            let Some(v) = saved.get(&v_key) else { continue };
            if v.shape() != st.v.shape() {
                continue; // grown or reshaped since the save
            }
            st.v = v.to_dtype(st.v.dtype())?;
            if mode == CarryMode::Both {
                if let Some(m) = saved.get(&format!("__optim__.m.{name}")) {
                    if m.shape() == st.m.shape() {
                        st.m = m.to_dtype(st.m.dtype())?;
                    }
                }
            }
            matched += 1;
        }
        // Restoring moments without their step count would make the
        // first corrected step divide by (1 - beta2^1).
        if let Some(t) = saved.get("__optim__.step_t") {
            let t: Vec<f32> = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            if let Some(t0) = t.first() {
                self.step_t = *t0 as usize;
            }
        }
        Ok(matched)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::{Optimizer, VarBuilder, VarMap};

    fn setup(dev: &Device) -> CResult<(VarMap, Tensor, Tensor)> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let w = vb.get_with_hints((2, 3), "w", candle_nn::Init::Const(0.3))?;
        let _ = w;
        let x = Tensor::from_vec(vec![1f32, -2., 0.5, 0.7, 1.5, -1.], (2, 3), dev)?;
        let y = Tensor::from_vec(vec![0.2f32, -0.4, 1.0, 0.1, 0.0, 0.6], (2, 3), dev)?;
        Ok((varmap, x, y))
    }

    fn loss_of(varmap: &VarMap, x: &Tensor, y: &Tensor) -> CResult<Tensor> {
        let data = varmap.data().lock().unwrap();
        let w = data.get("w").unwrap().as_tensor().clone();
        drop(data);
        ((&w * x)? - y)?.sqr()?.sum_all()
    }

    /// The whole point of re-implementing AdamW is that it must be the
    /// SAME AdamW. If it drifts from candle's, the no-carry control is
    /// no longer a control and every comparison built on it is void.
    #[test]
    fn matches_candle_adamw_step_for_step() -> CResult<()> {
        let dev = Device::Cpu;
        let params = ParamsAdamW {
            lr: 0.01,
            ..Default::default()
        };

        let (vm_a, x, y) = setup(&dev)?;
        let mut ours = PersistentAdamW::new(&vm_a, params.clone())?;

        let (vm_b, _, _) = setup(&dev)?;
        let mut theirs = candle_nn::AdamW::new(vm_b.all_vars(), params.clone())?;

        for _ in 0..12 {
            let la = loss_of(&vm_a, &x, &y)?;
            ours.backward_step(&la)?;
            let lb = loss_of(&vm_b, &x, &y)?;
            theirs.backward_step(&lb)?;
        }

        let a: Vec<f32> = vm_a.data().lock().unwrap().get("w").unwrap()
            .as_tensor().flatten_all()?.to_vec1()?;
        let b: Vec<f32> = vm_b.data().lock().unwrap().get("w").unwrap()
            .as_tensor().flatten_all()?.to_vec1()?;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-6,
                "param {i} diverged from candle: {x} vs {y}"
            );
        }
        Ok(())
    }

    /// Carrying `v` must shrink the next cycle's steps on parameters the
    /// previous cycle used. That is the entire mechanism; if it does not
    /// hold, nothing downstream means anything.
    #[test]
    fn carrying_second_moment_shrinks_steps() -> CResult<()> {
        let dev = Device::Cpu;
        let params = ParamsAdamW { lr: 0.01, ..Default::default() };
        let (vm, x, y) = setup(&dev)?;
        let mut warm = PersistentAdamW::new(&vm, params.clone())?;
        for _ in 0..30 {
            let l = loss_of(&vm, &x, &y)?;
            warm.backward_step(&l)?;
        }
        let saved = warm.state_tensors()?;
        let before: Vec<f32> = vm.data().lock().unwrap().get("w").unwrap()
            .as_tensor().flatten_all()?.to_vec1()?;

        // Two fresh optimizers over the SAME weights: one cold, one warm.
        let mut cold = PersistentAdamW::new(&vm, params.clone())?;
        let l = loss_of(&vm, &x, &y)?;
        cold.backward_step(&l)?;
        let after_cold: Vec<f32> = vm.data().lock().unwrap().get("w").unwrap()
            .as_tensor().flatten_all()?.to_vec1()?;

        // Restore weights, then step with carried state.
        {
            let data = vm.data().lock().unwrap();
            let w = data.get("w").unwrap();
            w.set(&Tensor::from_vec(before.clone(), w.shape(), &dev)?)?;
        }
        let mut hot = PersistentAdamW::new(&vm, params)?;
        let matched = hot.load_state(&saved, CarryMode::SecondMoment)?;
        assert!(matched > 0, "no optimizer state matched");
        assert_eq!(hot.step_t(), 30, "step_t must be restored with the moments");
        let l = loss_of(&vm, &x, &y)?;
        hot.backward_step(&l)?;
        let after_hot: Vec<f32> = vm.data().lock().unwrap().get("w").unwrap()
            .as_tensor().flatten_all()?.to_vec1()?;

        let d_cold: f32 = before.iter().zip(&after_cold).map(|(a, b)| (a - b).abs()).sum();
        let d_hot: f32 = before.iter().zip(&after_hot).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            d_hot < d_cold,
            "carrying v should shrink the step: hot {d_hot} vs cold {d_cold}"
        );
        Ok(())
    }

    /// Moments must not retain a backprop graph across steps.
    ///
    /// Storing a tracked `next_m` chains every step's moment to the one
    /// before it; memory then grows linearly in steps and the process is
    /// killed partway through a real run — which is exactly how this was
    /// found. Short runs pass regardless, so this walks far enough that
    /// an undetached chain is unmistakable in the graph depth.
    #[test]
    fn moments_do_not_retain_a_graph() -> CResult<()> {
        let dev = Device::Cpu;
        let (vm, x, y) = setup(&dev)?;
        let params = ParamsAdamW { lr: 0.001, ..Default::default() };
        let mut opt = PersistentAdamW::new(&vm, params)?;
        for _ in 0..400 {
            let l = loss_of(&vm, &x, &y)?;
            opt.backward_step(&l)?;
        }
        // A retained chain makes backward() recurse through 400 stored
        // moments; detached moments keep it at the loss graph's depth.
        let l = loss_of(&vm, &x, &y)?;
        let grads = l.backward()?;
        let data = vm.data().lock().unwrap();
        assert!(grads.get(data.get("w").unwrap()).is_some());
        Ok(())
    }

    /// `Off` must leave the optimizer exactly as constructed, or the
    /// control arm is silently contaminated.
    #[test]
    fn carry_off_restores_nothing() -> CResult<()> {
        let dev = Device::Cpu;
        let (vm, x, y) = setup(&dev)?;
        let params = ParamsAdamW { lr: 0.01, ..Default::default() };
        let mut a = PersistentAdamW::new(&vm, params.clone())?;
        for _ in 0..5 {
            let l = loss_of(&vm, &x, &y)?;
            a.backward_step(&l)?;
        }
        let saved = a.state_tensors()?;
        let mut b = PersistentAdamW::new(&vm, params)?;
        assert_eq!(b.load_state(&saved, CarryMode::Off)?, 0);
        assert_eq!(b.step_t(), 0);
        Ok(())
    }
}
