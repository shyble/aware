// CapKeyedRmsNorm: RMSNorm with per-cap gain vectors mixed by cap
// activations. Initialized to ones, so equivalent to identity at start.

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result as CResult, Tensor, Var, D};
use candle_nn::VarMap;

pub struct CapKeyedRmsNorm {
    pub n_caps: usize,
    pub d_model: usize,
    /// Top-K routing. 0 means "use all caps with standard softmax".
    pub top_k: usize,
    pub eps: f64,
    /// (n_caps, d_model). Each row initialized to ones.
    pub weight: Var,
    pub weight_name: String,
    pub varmap: Arc<Mutex<VarMap>>,
    pub device: Device,
    pub dtype: DType,
}

impl CapKeyedRmsNorm {
    pub fn new(
        n_caps: usize,
        d_model: usize,
        top_k: usize,
        eps: f64,
        prefix: &str,
        varmap: Arc<Mutex<VarMap>>,
        device: Device,
        dtype: DType,
    ) -> CResult<Self> {
        let init_data = Tensor::ones((n_caps, d_model), dtype, &device)?;
        let weight = Var::from_tensor(&init_data)?;
        let weight_name = format!("{}.weight", prefix);
        {
            let vm = varmap.lock().unwrap();
            let data = vm.data();
            let mut data = data.lock().unwrap();
            data.insert(weight_name.clone(), weight.clone());
        }
        Ok(Self {
            n_caps,
            d_model,
            top_k,
            eps,
            weight,
            weight_name,
            varmap,
            device,
            dtype,
        })
    }

    /// Forward.
    /// `xs`: (B, S, d_model), `cap_acts`: (B, S, n_caps).
    pub fn forward(&self, xs: &Tensor, cap_acts: &Tensor) -> CResult<Tensor> {
        let xs_dims = xs.dims();
        let n_dim = xs_dims.len();
        let total: usize = xs_dims[..n_dim - 1].iter().product();
        let xs_flat = xs.reshape((total, self.d_model))?;
        let cap_dims = cap_acts.dims();
        let cap_total: usize = cap_dims[..cap_dims.len() - 1].iter().product();
        if cap_total != total {
            return Err(candle_core::Error::Msg(format!(
                "CapKeyedRmsNorm.forward: token count mismatch xs={} cap_acts={}",
                total, cap_total,
            )));
        }
        let cap_flat = cap_acts.reshape((total, self.n_caps))?;

        // RMS along d_model axis: (T, 1)
        let sq = xs_flat.sqr()?;
        let mean_sq = sq.mean_keepdim(D::Minus1)?;
        let rms = (mean_sq + self.eps)?.sqrt()?;
        let normalized = xs_flat.broadcast_div(&rms)?;

        // Per-token mixed weight via top-K (or full) softmax over caps.
        let gate_w = top_k_softmax(&cap_flat, self.top_k)?;
        let mixed_weight = gate_w.matmul(self.weight.as_tensor())?;

        let out_flat = normalized.mul(&mixed_weight)?;
        let mut out_dims: Vec<usize> = xs_dims[..n_dim - 1].to_vec();
        out_dims.push(self.d_model);
        out_flat.reshape(out_dims)
    }

    pub fn shrink_caps(&mut self, keep_indices: &[usize]) -> CResult<()> {
        let n = self.n_caps;
        let kept = keep_indices.len();
        if kept == n {
            return Ok(());
        }
        if kept == 0 {
            return Err(candle_core::Error::Msg(
                "CapKeyedRmsNorm.shrink_caps: cannot shrink to 0".into(),
            ));
        }
        let idx_u32: Vec<u32> = keep_indices.iter().map(|&i| i as u32).collect();
        let idx_t = Tensor::from_vec(idx_u32, (kept,), &self.device)?;
        let new_w = self.weight.as_tensor().index_select(&idx_t, 0)?;
        let new_var = Var::from_tensor(&new_w)?;
        {
            let vm = self.varmap.lock().unwrap();
            let data = vm.data();
            let mut data = data.lock().unwrap();
            data.insert(self.weight_name.clone(), new_var.clone());
        }
        self.weight = new_var;
        self.n_caps = kept;
        Ok(())
    }

    pub fn grow(&mut self, delta: usize) -> CResult<usize> {
        if delta == 0 {
            return Ok(self.n_caps);
        }
        let new_rows = Tensor::ones((delta, self.d_model), self.dtype, &self.device)?;
        let cat = Tensor::cat(&[self.weight.as_tensor(), &new_rows], 0)?;
        let new_var = Var::from_tensor(&cat)?;
        {
            let vm = self.varmap.lock().unwrap();
            let data = vm.data();
            let mut data = data.lock().unwrap();
            data.insert(self.weight_name.clone(), new_var.clone());
        }
        self.weight = new_var;
        self.n_caps += delta;
        Ok(self.n_caps)
    }

    pub fn reseed_row(&mut self, k: usize) -> CResult<()> {
        if k >= self.n_caps {
            return Ok(());
        }
        let new_row = Tensor::ones((1, self.d_model), self.dtype, &self.device)?;
        let n = self.n_caps;
        let pieces: Vec<Tensor> = if k == 0 {
            vec![new_row, self.weight.as_tensor().narrow(0, 1, n - 1)?]
        } else if k == n - 1 {
            vec![self.weight.as_tensor().narrow(0, 0, k)?, new_row]
        } else {
            vec![
                self.weight.as_tensor().narrow(0, 0, k)?,
                new_row,
                self.weight.as_tensor().narrow(0, k + 1, n - k - 1)?,
            ]
        };
        let refs: Vec<&Tensor> = pieces.iter().collect();
        let new_w = Tensor::cat(&refs, 0)?;
        let new_var = Var::from_tensor(&new_w)?;
        {
            let vm = self.varmap.lock().unwrap();
            let data = vm.data();
            let mut data = data.lock().unwrap();
            data.insert(self.weight_name.clone(), new_var.clone());
        }
        self.weight = new_var;
        Ok(())
    }
}

/// Top-K softmax along the last dim.
/// - k = 0 (or k >= n): standard softmax over all dims (no masking)
/// - k > 0 and k < n: mask all but top-K to -1e9, then softmax
pub fn top_k_softmax(x: &Tensor, k: usize) -> CResult<Tensor> {
    let dims = x.dims();
    let n = dims[dims.len() - 1];
    if k == 0 || k >= n {
        return candle_nn::ops::softmax(x, D::Minus1);
    }
    let (sorted, _idx) = x.sort_last_dim(false)?;
    let threshold = sorted.narrow(D::Minus1, k - 1, 1)?;
    let in_topk = x.broadcast_ge(&threshold)?.to_dtype(x.dtype())?;
    let neg_bias = (in_topk.affine(-1.0, 1.0)? * (-1e9))?;
    let masked = (x + neg_bias)?;
    candle_nn::ops::softmax(&masked, D::Minus1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    fn make(n_caps: usize, d_model: usize, top_k: usize) -> (CapKeyedRmsNorm, Arc<Mutex<VarMap>>) {
        let varmap = Arc::new(Mutex::new(VarMap::new()));
        let n = CapKeyedRmsNorm::new(
            n_caps,
            d_model,
            top_k,
            1e-5,
            "cap_norm",
            varmap.clone(),
            cpu(),
            DType::F32,
        )
        .unwrap();
        (n, varmap)
    }

    #[test]
    fn forward_shape() {
        let (n, _vm) = make(6, 16, 3);
        let xs = Tensor::randn(0f32, 1f32, (2, 5, 16), &cpu()).unwrap();
        let caps = Tensor::randn(0f32, 1f32, (2, 5, 6), &cpu()).unwrap();
        let y = n.forward(&xs, &caps).unwrap();
        assert_eq!(y.dims(), &[2, 5, 16]);
    }

    #[test]
    fn forward_shape_soft_no_k() {
        let (n, _vm) = make(6, 16, 0); // top_k=0 -> all caps via standard softmax
        let xs = Tensor::randn(0f32, 1f32, (2, 5, 16), &cpu()).unwrap();
        let caps = Tensor::randn(0f32, 1f32, (2, 5, 6), &cpu()).unwrap();
        let y = n.forward(&xs, &caps).unwrap();
        assert_eq!(y.dims(), &[2, 5, 16]);
    }

    /// At init (weight rows = ones), output equals xs / rms.
    #[test]
    fn identity_at_init() {
        let (norm, _vm) = make(3, 8, 2);
        let xs = Tensor::from_vec(vec![2.0f32; 8], (1, 1, 8), &cpu()).unwrap();
        let caps = Tensor::from_vec(vec![1.0f32, 0.5, 0.1], (1, 1, 3), &cpu()).unwrap();
        let y = norm.forward(&xs, &caps).unwrap();
        let v: Vec<Vec<Vec<f32>>> = y.to_vec3().unwrap();
        for d in 0..8 {
            assert!(
                (v[0][0][d] - 1.0).abs() < 1e-3,
                "init forward should produce ~1.0, got {}",
                v[0][0][d]
            );
        }
    }

    /// Off-domain cap's weight row gets zero gradient under top_k=1.
    #[test]
    fn off_domain_cap_row_receives_zero_gradient() {
        let (n, _vm) = make(4, 8, 1);
        let xs = Tensor::randn(0f32, 1f32, (1, 3, 8), &cpu()).unwrap();
        let mut cap_data = vec![0.0f32; 1 * 3 * 4];
        for t in 0..3 {
            cap_data[t * 4 + 0] = 1.0;
        }
        let caps = Tensor::from_vec(cap_data, (1, 3, 4), &cpu()).unwrap();

        let y = n.forward(&xs, &caps).unwrap();
        let loss = y.sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();

        let g = grads.get(n.weight.as_tensor()).expect("no grad");
        let g_v: Vec<Vec<f32>> = g.to_vec2().unwrap();
        for k in 1..4 {
            for d in 0..8 {
                assert!(
                    g_v[k][d].abs() < 1e-9,
                    "off-domain cap {} got nonzero grad at dim {}: {}",
                    k,
                    d,
                    g_v[k][d]
                );
            }
        }
        let any = (0..8).any(|d| g_v[0][d].abs() > 1e-9);
        assert!(any, "active cap 0 got zero grad");
    }

    #[test]
    fn grow_preserves_existing_rows() {
        let (mut n, _vm) = make(3, 6, 2);
        let before = n.weight.as_tensor().to_vec2::<f32>().unwrap();
        n.grow(2).unwrap();
        let after = n.weight.as_tensor().to_vec2::<f32>().unwrap();
        assert_eq!(after.len(), 5);
        for k in 0..3 {
            for d in 0..6 {
                assert!((before[k][d] - after[k][d]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn top_k_softmax_no_k_returns_standard_softmax() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &cpu()).unwrap();
        let r = top_k_softmax(&x, 0).unwrap();
        let v: Vec<Vec<f32>> = r.to_vec2().unwrap();
        let sum: f32 = v[0].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {}",
            sum
        );
        // All three values should be nonzero (no masking)
        for x in &v[0] {
            assert!(
                *x > 0.01,
                "no top-K mask should mean all values get density, got {}",
                x
            );
        }
    }
}
