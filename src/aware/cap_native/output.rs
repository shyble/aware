// CapKeyedOutput: per-cap (d_model -> vocab) projection stacks mixed by
// cap activations. SoftTopK or HardTop1Sparse routing.

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result as CResult, Tensor, Var, D};
use candle_nn::VarMap;

use super::compression::RoutingMode;
use super::norm::top_k_softmax;

pub struct CapKeyedOutput {
    pub n_caps: usize,
    pub d_model: usize,
    pub vocab: usize,
    pub top_k: usize,
    /// (n_caps, d_model, vocab)
    pub w: Var,
    pub w_name: String,
    pub varmap: Arc<Mutex<VarMap>>,
    pub init_scale: f64,
    pub device: Device,
    pub dtype: DType,
    pub routing: RoutingMode,
}

impl CapKeyedOutput {
    pub fn new(
        n_caps: usize,
        d_model: usize,
        vocab: usize,
        top_k: usize,
        prefix: &str,
        varmap: Arc<Mutex<VarMap>>,
        device: Device,
        dtype: DType,
    ) -> CResult<Self> {
        let init_scale = 1.0 / (d_model as f64).sqrt();
        let init_data = init_3d(n_caps, d_model, vocab, init_scale, &device, dtype)?;
        let w = Var::from_tensor(&init_data)?;
        let w_name = format!("{}.w", prefix);
        {
            let vm = varmap.lock().unwrap();
            let data = vm.data();
            let mut data = data.lock().unwrap();
            data.insert(w_name.clone(), w.clone());
        }
        Ok(Self {
            n_caps,
            d_model,
            vocab,
            top_k,
            w,
            w_name,
            varmap,
            init_scale,
            device,
            dtype,
            routing: RoutingMode::SoftTopK,
        })
    }

    pub fn with_routing(mut self, routing: RoutingMode) -> Self {
        self.routing = routing;
        self
    }

    pub fn forward(&self, h: &Tensor, cap_acts: &Tensor) -> CResult<Tensor> {
        match self.routing {
            RoutingMode::HardTop1Sparse => self.forward_sparse_top1(h, cap_acts),
            RoutingMode::SoftTopK => self.forward_soft_topk(h, cap_acts),
        }
    }

    fn forward_soft_topk(&self, h: &Tensor, cap_acts: &Tensor) -> CResult<Tensor> {
        let h_dims = h.dims();
        let n_dim = h_dims.len();
        let total: usize = h_dims[..n_dim - 1].iter().product();
        let h_flat = h.reshape((total, self.d_model))?;
        let cap_dims = cap_acts.dims();
        let cap_total: usize = cap_dims[..cap_dims.len() - 1].iter().product();
        if cap_total != total {
            return Err(candle_core::Error::Msg(format!(
                "CapKeyedOutput.forward: token count mismatch h={} cap_acts={}",
                total, cap_total,
            )));
        }
        let cap_flat = cap_acts.reshape((total, self.n_caps))?;
        let gate_w = top_k_softmax(&cap_flat, self.top_k)?;

        let mut accum: Option<Tensor> = None;
        for k in 0..self.n_caps {
            let w_k = self.w.as_tensor().narrow(0, k, 1)?.squeeze(0)?;
            let logits_k = h_flat.matmul(&w_k)?;
            let weight_k = gate_w.narrow(D::Minus1, k, 1)?;
            let weighted = logits_k.broadcast_mul(&weight_k)?;
            accum = Some(match accum {
                Some(a) => (a + weighted)?,
                None => weighted,
            });
        }
        let out_flat = accum
            .ok_or_else(|| candle_core::Error::Msg("CapKeyedOutput.forward: zero caps".into()))?;
        let mut out_dims: Vec<usize> = h_dims[..n_dim - 1].to_vec();
        out_dims.push(self.vocab);
        out_flat.reshape(out_dims)
    }

    fn forward_sparse_top1(&self, h: &Tensor, cap_acts: &Tensor) -> CResult<Tensor> {
        let h_dims = h.dims();
        let n_dim = h_dims.len();
        let total: usize = h_dims[..n_dim - 1].iter().product();
        let h_flat = h.reshape((total, self.d_model))?;
        let cap_dims = cap_acts.dims();
        let cap_total: usize = cap_dims[..cap_dims.len() - 1].iter().product();
        if cap_total != total {
            return Err(candle_core::Error::Msg(format!(
                "CapKeyedOutput.forward_sparse_top1: token count mismatch"
            )));
        }
        let cap_flat = cap_acts.reshape((total, self.n_caps))?;

        let winners_t = cap_flat.argmax(D::Minus1)?;
        let winners: Vec<u32> = winners_t.to_vec1::<u32>()?;
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); self.n_caps];
        for (t, &w) in winners.iter().enumerate() {
            let k = (w as usize).min(self.n_caps - 1);
            buckets[k].push(t as u32);
        }

        let mut out_flat = Tensor::zeros((total, self.vocab), self.dtype, &self.device)?;
        for k in 0..self.n_caps {
            if buckets[k].is_empty() {
                continue;
            }
            let n_k = buckets[k].len();
            let idx_t = Tensor::from_vec(buckets[k].clone(), (n_k,), &self.device)?;
            let h_k = h_flat.index_select(&idx_t, 0)?;
            let w_k = self.w.as_tensor().narrow(0, k, 1)?.squeeze(0)?;
            let logits_k = h_k.matmul(&w_k)?;
            out_flat = out_flat.index_add(&idx_t, &logits_k, 0)?;
        }

        let mut out_dims: Vec<usize> = h_dims[..n_dim - 1].to_vec();
        out_dims.push(self.vocab);
        out_flat.reshape(out_dims)
    }

    pub fn grow_vocab(&mut self, delta_vocab: usize) -> CResult<usize> {
        if delta_vocab == 0 {
            return Ok(self.vocab);
        }
        let new_cols = init_3d(
            self.n_caps,
            self.d_model,
            delta_vocab,
            self.init_scale,
            &self.device,
            self.dtype,
        )?;
        let cat_w = Tensor::cat(&[self.w.as_tensor(), &new_cols], 2)?;
        let new_var = Var::from_tensor(&cat_w)?;
        {
            let vm = self.varmap.lock().unwrap();
            let data = vm.data();
            let mut data = data.lock().unwrap();
            data.insert(self.w_name.clone(), new_var.clone());
        }
        self.w = new_var;
        self.vocab += delta_vocab;
        Ok(self.vocab)
    }

    pub fn shrink_caps(&mut self, keep_indices: &[usize]) -> CResult<()> {
        let n = self.n_caps;
        let kept = keep_indices.len();
        if kept == n {
            return Ok(());
        }
        if kept == 0 {
            return Err(candle_core::Error::Msg(
                "CapKeyedOutput.shrink_caps: cannot shrink to 0".into(),
            ));
        }
        let idx_u32: Vec<u32> = keep_indices.iter().map(|&i| i as u32).collect();
        let idx_t = Tensor::from_vec(idx_u32, (kept,), &self.device)?;
        let new_w = self.w.as_tensor().index_select(&idx_t, 0)?;
        let new_var = Var::from_tensor(&new_w)?;
        {
            let vm = self.varmap.lock().unwrap();
            let data = vm.data();
            let mut data = data.lock().unwrap();
            data.insert(self.w_name.clone(), new_var.clone());
        }
        self.w = new_var;
        self.n_caps = kept;
        Ok(())
    }

    pub fn grow(&mut self, delta: usize) -> CResult<usize> {
        if delta == 0 {
            return Ok(self.n_caps);
        }
        let new_w = init_3d(
            delta,
            self.d_model,
            self.vocab,
            self.init_scale,
            &self.device,
            self.dtype,
        )?;
        let cat_w = Tensor::cat(&[self.w.as_tensor(), &new_w], 0)?;
        let new_var = Var::from_tensor(&cat_w)?;
        {
            let vm = self.varmap.lock().unwrap();
            let data = vm.data();
            let mut data = data.lock().unwrap();
            data.insert(self.w_name.clone(), new_var.clone());
        }
        self.w = new_var;
        self.n_caps += delta;
        Ok(self.n_caps)
    }

    pub fn reseed_row(&mut self, k: usize) -> CResult<()> {
        if k >= self.n_caps {
            return Ok(());
        }
        let new_slab = init_3d(
            1,
            self.d_model,
            self.vocab,
            self.init_scale,
            &self.device,
            self.dtype,
        )?;
        let n = self.n_caps;
        let pieces: Vec<Tensor> = if k == 0 {
            vec![new_slab, self.w.as_tensor().narrow(0, 1, n - 1)?]
        } else if k == n - 1 {
            vec![self.w.as_tensor().narrow(0, 0, k)?, new_slab]
        } else {
            vec![
                self.w.as_tensor().narrow(0, 0, k)?,
                new_slab,
                self.w.as_tensor().narrow(0, k + 1, n - k - 1)?,
            ]
        };
        let refs: Vec<&Tensor> = pieces.iter().collect();
        let new_w = Tensor::cat(&refs, 0)?;
        let new_var = Var::from_tensor(&new_w)?;
        {
            let vm = self.varmap.lock().unwrap();
            let data = vm.data();
            let mut data = data.lock().unwrap();
            data.insert(self.w_name.clone(), new_var.clone());
        }
        self.w = new_var;
        Ok(())
    }
}

fn init_3d(
    a: usize,
    b: usize,
    c: usize,
    scale: f64,
    device: &Device,
    dtype: DType,
) -> CResult<Tensor> {
    let raw = Tensor::randn(0f32, 1f32, (a, b, c), device)?;
    (raw * scale)?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    fn make(
        n_caps: usize,
        d_model: usize,
        vocab: usize,
        top_k: usize,
    ) -> (CapKeyedOutput, Arc<Mutex<VarMap>>) {
        let varmap = Arc::new(Mutex::new(VarMap::new()));
        let cko = CapKeyedOutput::new(
            n_caps,
            d_model,
            vocab,
            top_k,
            "out_proj",
            varmap.clone(),
            cpu(),
            DType::F32,
        )
        .unwrap();
        (cko, varmap)
    }

    #[test]
    fn forward_shape() {
        let (cko, _vm) = make(6, 16, 10, 3);
        let h = Tensor::randn(0f32, 1f32, (2, 5, 16), &cpu()).unwrap();
        let caps = Tensor::randn(0f32, 1f32, (2, 5, 6), &cpu()).unwrap();
        let logits = cko.forward(&h, &caps).unwrap();
        assert_eq!(logits.dims(), &[2, 5, 10]);
    }

    #[test]
    fn forward_shape_soft_no_k() {
        let (cko, _vm) = make(6, 16, 10, 0);
        let h = Tensor::randn(0f32, 1f32, (2, 5, 16), &cpu()).unwrap();
        let caps = Tensor::randn(0f32, 1f32, (2, 5, 6), &cpu()).unwrap();
        let logits = cko.forward(&h, &caps).unwrap();
        assert_eq!(logits.dims(), &[2, 5, 10]);
    }

    #[test]
    fn off_domain_cap_slab_receives_zero_gradient() {
        let (cko, _vm) = make(4, 8, 6, 1);
        let h = Tensor::randn(0f32, 1f32, (1, 3, 8), &cpu()).unwrap();
        let mut cap_data = vec![0.0f32; 12];
        for t in 0..3 {
            cap_data[t * 4 + 0] = 1.0;
        }
        let caps = Tensor::from_vec(cap_data, (1, 3, 4), &cpu()).unwrap();

        let logits = cko.forward(&h, &caps).unwrap();
        let loss = logits.sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();

        let g = grads.get(cko.w.as_tensor()).expect("no grad");
        let g_v: Vec<Vec<Vec<f32>>> = g.to_vec3().unwrap();
        for kk in 1..4 {
            for di in 0..8 {
                for dj in 0..6 {
                    assert!(
                        g_v[kk][di][dj].abs() < 1e-9,
                        "off-domain cap {} got nonzero grad",
                        kk
                    );
                }
            }
        }
        let any = (0..8).any(|di| (0..6).any(|dj| g_v[0][di][dj].abs() > 1e-9));
        assert!(any, "active cap 0 got zero grad");
    }

    #[test]
    fn forward_shape_sparse_top1() {
        let (cko_orig, _vm) = make(6, 16, 10, 1);
        let cko = cko_orig.with_routing(RoutingMode::HardTop1Sparse);
        let h = Tensor::randn(0f32, 1f32, (2, 5, 16), &cpu()).unwrap();
        let caps = Tensor::randn(0f32, 1f32, (2, 5, 6), &cpu()).unwrap();
        let logits = cko.forward(&h, &caps).unwrap();
        assert_eq!(logits.dims(), &[2, 5, 10]);
    }

    #[test]
    fn grow_preserves_existing_slabs() {
        let (mut cko, _vm) = make(3, 6, 8, 2);
        let before = cko.w.as_tensor().to_vec3::<f32>().unwrap();
        cko.grow(2).unwrap();
        let after = cko.w.as_tensor().to_vec3::<f32>().unwrap();
        assert_eq!(after.len(), 5);
        for k in 0..3 {
            for di in 0..6 {
                for dj in 0..8 {
                    assert!((before[k][di][dj] - after[k][di][dj]).abs() < 1e-12);
                }
            }
        }
    }
}
