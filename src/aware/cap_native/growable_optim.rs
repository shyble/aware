// GrowableAdamW: AdamW for parameters whose row count can change at
// runtime. Moment tensors track grow/reseed via the Var name; weight_decay
// is fixed at 0 to preserve isolation of unused cap rows.

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result as CResult, Tensor};
use candle_nn::VarMap;

pub struct GrowableAdamW {
    pub var_name: String,
    pub varmap: Arc<Mutex<VarMap>>,
    pub m: Tensor,
    pub v: Tensor,
    pub step_count: i32,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub device: Device,
    pub dtype: DType,
}

impl GrowableAdamW {
    pub fn new(
        var_name: &str,
        varmap: Arc<Mutex<VarMap>>,
        param_shape: (usize, usize),
        lr: f64,
    ) -> CResult<Self> {
        Self::new_any_rank(var_name, varmap, &[param_shape.0, param_shape.1], lr)
    }

    /// Construct for a parameter of any rank. The first dim is the
    /// "rows / experts" axis that grow/reseed operate along.
    pub fn new_any_rank(
        var_name: &str,
        varmap: Arc<Mutex<VarMap>>,
        param_shape: &[usize],
        lr: f64,
    ) -> CResult<Self> {
        let (device, dtype) = {
            let vm = varmap.lock().unwrap();
            let data = vm.data();
            let data = data.lock().unwrap();
            let v = data.get(var_name).expect("var not in varmap");
            (v.device().clone(), v.dtype())
        };
        let shape_vec: Vec<usize> = param_shape.to_vec();
        let m = Tensor::zeros(shape_vec.clone(), dtype, &device)?;
        let v = Tensor::zeros(shape_vec, dtype, &device)?;
        Ok(Self {
            var_name: var_name.to_string(),
            varmap,
            m,
            v,
            step_count: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            device,
            dtype,
        })
    }

    /// Step using the gradient `g`. Updates m, v, and the VarMap-tracked param.
    pub fn step(&mut self, grad: &Tensor) -> CResult<()> {
        self.step_count += 1;
        let t = self.step_count;

        let m_new = ((&self.m * self.beta1)? + (grad * (1.0 - self.beta1))?)?;
        let g_sq = grad.sqr()?;
        let v_new = ((&self.v * self.beta2)? + (g_sq * (1.0 - self.beta2))?)?;

        let m_hat = (&m_new / (1.0 - self.beta1.powi(t)))?;
        let v_hat = (&v_new / (1.0 - self.beta2.powi(t)))?;

        let current_w = {
            let vm = self.varmap.lock().unwrap();
            let data = vm.data();
            let data = data.lock().unwrap();
            data.get(&self.var_name)
                .expect("var not in varmap")
                .as_tensor()
                .clone()
        };

        let denom = (v_hat.sqrt()? + self.eps)?;
        let step_term = m_hat.broadcast_div(&denom)?;
        let decay_term = (&current_w * self.weight_decay)?;
        let total_step = ((step_term + decay_term)? * self.lr)?;
        let new_w = (current_w - total_step)?;

        {
            let vm = self.varmap.lock().unwrap();
            let data = vm.data();
            let data = data.lock().unwrap();
            let var = data.get(&self.var_name).expect("var not in varmap");
            var.set(&new_w)?;
        }

        self.m = m_new;
        self.v = v_new;
        Ok(())
    }

    pub fn grow_rows(&mut self, delta: usize) -> CResult<()> {
        if delta == 0 {
            return Ok(());
        }
        let dims = self.m.dims();
        let mut zero_shape: Vec<usize> = dims.to_vec();
        zero_shape[0] = delta;
        let zero_rows = Tensor::zeros(zero_shape, self.dtype, &self.device)?;
        self.m = Tensor::cat(&[&self.m, &zero_rows], 0)?;
        self.v = Tensor::cat(&[&self.v, &zero_rows], 0)?;
        Ok(())
    }

    pub fn shrink_rows(&mut self, keep_indices: &[usize]) -> CResult<()> {
        let n = self.m.dims()[0];
        let kept = keep_indices.len();
        if kept == n {
            return Ok(());
        }
        if kept == 0 {
            return Err(candle_core::Error::Msg(
                "GrowableAdamW.shrink_rows: cannot shrink to 0 rows".into(),
            ));
        }
        let idx_u32: Vec<u32> = keep_indices.iter().map(|&i| i as u32).collect();
        let idx_t = Tensor::from_vec(idx_u32, (kept,), &self.device)?;
        self.m = self.m.index_select(&idx_t, 0)?;
        self.v = self.v.index_select(&idx_t, 0)?;
        Ok(())
    }

    pub fn reset_row(&mut self, k: usize) -> CResult<()> {
        let dims = self.m.dims();
        let n = dims[0];
        if k >= n {
            return Ok(());
        }
        let mut mask_shape: Vec<usize> = vec![1; dims.len()];
        mask_shape[0] = n;
        let mut mask_data = vec![1.0f32; n];
        mask_data[k] = 0.0;
        let mask = Tensor::from_vec(mask_data, mask_shape, &self.device)?.to_dtype(self.dtype)?;
        self.m = self.m.broadcast_mul(&mask)?;
        self.v = self.v.broadcast_mul(&mask)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Var;

    fn cpu() -> Device {
        Device::Cpu
    }

    /// Register a `(rows, cols)` Var in the VarMap under `name` and return it.
    /// Replaces test dependency on the deleted CapProjection module.
    fn register_var(name: &str, rows: usize, cols: usize, vm: &Arc<Mutex<VarMap>>) -> Var {
        let data = (Tensor::randn(0f32, 1f32, (rows, cols), &cpu()).unwrap()
            * (1.0 / (cols as f64).sqrt()))
        .unwrap();
        let var = Var::from_tensor(&data).unwrap();
        let lock = vm.lock().unwrap();
        let d = lock.data();
        let mut d = d.lock().unwrap();
        d.insert(name.to_string(), var.clone());
        var
    }

    /// Grow a (n, cols) Var in the VarMap by `delta` rows of zeros. Replaces
    /// CapProjection.grow() for tests.
    fn grow_var(name: &str, delta: usize, cols: usize, vm: &Arc<Mutex<VarMap>>) {
        let lock = vm.lock().unwrap();
        let d = lock.data();
        let mut d = d.lock().unwrap();
        let var = d.get(name).unwrap();
        let cur = var.as_tensor().clone();
        let zeros = Tensor::zeros((delta, cols), DType::F32, &cpu()).unwrap();
        let new_t = Tensor::cat(&[&cur, &zeros], 0).unwrap();
        let new_var = Var::from_tensor(&new_t).unwrap();
        d.insert(name.to_string(), new_var);
    }

    #[test]
    fn step_changes_param() {
        let varmap = Arc::new(Mutex::new(VarMap::new()));
        let var = register_var("wp.weight", 4, 6, &varmap);
        let before = var.as_tensor().to_vec2::<f32>().unwrap();
        let mut opt = GrowableAdamW::new("wp.weight", varmap.clone(), (4, 6), 0.01).unwrap();
        let grad = Tensor::ones((4, 6), DType::F32, &cpu()).unwrap();
        opt.step(&grad).unwrap();
        let after = {
            let lock = varmap.lock().unwrap();
            let d = lock.data();
            let d = d.lock().unwrap();
            d.get("wp.weight")
                .unwrap()
                .as_tensor()
                .to_vec2::<f32>()
                .unwrap()
        };
        let any_change = (0..4).any(|i| (0..6).any(|d| (before[i][d] - after[i][d]).abs() > 1e-9));
        assert!(any_change, "step did not change weights");
    }

    #[test]
    fn grow_rows_extends_and_preserves() {
        let varmap = Arc::new(Mutex::new(VarMap::new()));
        let _var = register_var("wp.weight", 4, 6, &varmap);
        let mut opt = GrowableAdamW::new("wp.weight", varmap.clone(), (4, 6), 0.01).unwrap();

        let grad = Tensor::ones((4, 6), DType::F32, &cpu()).unwrap();
        for _ in 0..5 {
            opt.step(&grad).unwrap();
        }
        let m_before = opt.m.to_vec2::<f32>().unwrap();
        let v_before = opt.v.to_vec2::<f32>().unwrap();

        grow_var("wp.weight", 3, 6, &varmap);
        opt.grow_rows(3).unwrap();
        assert_eq!(opt.m.dims(), &[7, 6]);
        assert_eq!(opt.v.dims(), &[7, 6]);

        let m_after = opt.m.to_vec2::<f32>().unwrap();
        let v_after = opt.v.to_vec2::<f32>().unwrap();
        for i in 0..4 {
            for d in 0..6 {
                assert!((m_before[i][d] - m_after[i][d]).abs() < 1e-9);
                assert!((v_before[i][d] - v_after[i][d]).abs() < 1e-9);
            }
        }
        for i in 4..7 {
            for d in 0..6 {
                assert_eq!(m_after[i][d], 0.0);
                assert_eq!(v_after[i][d], 0.0);
            }
        }
    }

    #[test]
    fn reset_row_zeros_target_only() {
        let varmap = Arc::new(Mutex::new(VarMap::new()));
        let _var = register_var("wp.weight", 5, 4, &varmap);
        let mut opt = GrowableAdamW::new("wp.weight", varmap.clone(), (5, 4), 0.01).unwrap();
        let grad = Tensor::ones((5, 4), DType::F32, &cpu()).unwrap();
        for _ in 0..3 {
            opt.step(&grad).unwrap();
        }
        let m_before = opt.m.to_vec2::<f32>().unwrap();

        opt.reset_row(2).unwrap();
        let m_after = opt.m.to_vec2::<f32>().unwrap();
        for i in 0..5 {
            for d in 0..4 {
                if i == 2 {
                    assert_eq!(m_after[i][d], 0.0);
                } else {
                    assert!((m_before[i][d] - m_after[i][d]).abs() < 1e-9);
                }
            }
        }
    }
}
