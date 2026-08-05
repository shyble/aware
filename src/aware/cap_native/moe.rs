// CapMoeMlp: per-cap SwiGLU experts mixed by cap activations.
// SoftTopK routes via softmax over all caps; HardTop1Sparse picks the
// argmax winner per token.

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result as CResult, Tensor, Var, D};
use candle_nn::VarMap;

use super::compression::RoutingMode;
use super::norm::top_k_softmax;
use super::sparse_routing::{
    apply_bounded_grouped_projection, routing_from_cap_acts, BoundedGroupedRouting,
};

pub struct CapMoeMlp {
    pub n_caps: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub top_k: usize,
    pub w_gate: Var,
    pub w_value: Var,
    pub w_out: Var,
    pub w_gate_name: String,
    pub w_value_name: String,
    pub w_out_name: String,
    pub varmap: Arc<Mutex<VarMap>>,
    pub init_scale_in: f64,
    pub init_scale_out: f64,
    pub device: Device,
    pub dtype: DType,
    pub routing: RoutingMode,
}

impl CapMoeMlp {
    pub fn new(
        n_caps: usize,
        d_model: usize,
        d_ff: usize,
        top_k: usize,
        prefix: &str,
        varmap: Arc<Mutex<VarMap>>,
        device: Device,
        dtype: DType,
    ) -> CResult<Self> {
        let init_scale_in = 1.0 / (d_model as f64).sqrt();
        let init_scale_out = 1.0 / (d_ff as f64).sqrt();
        let w_gate_data = init_3d(n_caps, d_model, d_ff, init_scale_in, &device, dtype)?;
        let w_value_data = init_3d(n_caps, d_model, d_ff, init_scale_in, &device, dtype)?;
        let w_out_data = init_3d(n_caps, d_ff, d_model, init_scale_out, &device, dtype)?;
        let w_gate = Var::from_tensor(&w_gate_data)?;
        let w_value = Var::from_tensor(&w_value_data)?;
        let w_out = Var::from_tensor(&w_out_data)?;
        let w_gate_name = format!("{}.w_gate", prefix);
        let w_value_name = format!("{}.w_value", prefix);
        let w_out_name = format!("{}.w_out", prefix);
        {
            let vm = varmap.lock().unwrap();
            let data = vm.data();
            let mut data = data.lock().unwrap();
            data.insert(w_gate_name.clone(), w_gate.clone());
            data.insert(w_value_name.clone(), w_value.clone());
            data.insert(w_out_name.clone(), w_out.clone());
        }
        Ok(Self {
            n_caps,
            d_model,
            d_ff,
            top_k,
            w_gate,
            w_value,
            w_out,
            w_gate_name,
            w_value_name,
            w_out_name,
            varmap,
            init_scale_in,
            init_scale_out,
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
        self.forward_with_routing(h, cap_acts, None)
    }

    /// See `CapKeyedMha::forward_with_routing`.
    pub fn forward_with_routing(
        &self,
        h: &Tensor,
        cap_acts: &Tensor,
        routing_in: Option<&BoundedGroupedRouting>,
    ) -> CResult<Tensor> {
        match self.routing {
            RoutingMode::HardTop1Sparse => self.forward_sparse_top1(h, cap_acts, routing_in),
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
                "CapMoeMlp.forward: token count mismatch h={} cap_acts={}",
                total, cap_total,
            )));
        }
        let cap_flat = cap_acts.reshape((total, self.n_caps))?;
        let gate_w = top_k_softmax(&cap_flat, self.top_k)?;

        let mut accum: Option<Tensor> = None;
        for k in 0..self.n_caps {
            let w_g_k = self.w_gate.as_tensor().narrow(0, k, 1)?.squeeze(0)?;
            let w_v_k = self.w_value.as_tensor().narrow(0, k, 1)?.squeeze(0)?;
            let w_o_k = self.w_out.as_tensor().narrow(0, k, 1)?.squeeze(0)?;

            let gate = h_flat.matmul(&w_g_k)?;
            let gate_act = silu(&gate)?;
            let val = h_flat.matmul(&w_v_k)?;
            let hidden = gate_act.mul(&val)?;
            let expert_out = hidden.matmul(&w_o_k)?;

            let weight_k = gate_w.narrow(D::Minus1, k, 1)?;
            let weighted = expert_out.broadcast_mul(&weight_k)?;
            accum = Some(match accum {
                Some(a) => (a + weighted)?,
                None => weighted,
            });
        }
        let out_flat =
            accum.ok_or_else(|| candle_core::Error::Msg("CapMoeMlp.forward: zero caps".into()))?;
        let mut out_dims: Vec<usize> = h_dims[..n_dim - 1].to_vec();
        out_dims.push(self.d_model);
        out_flat.reshape(out_dims)
    }

    pub(crate) fn forward_sparse_top1(
        &self,
        h: &Tensor,
        cap_acts: &Tensor,
        routing_in: Option<&BoundedGroupedRouting>,
    ) -> CResult<Tensor> {
        let h_dims = h.dims();
        let n_dim = h_dims.len();
        let total: usize = h_dims[..n_dim - 1].iter().product();
        let h_flat = h.reshape((total, self.d_model))?;
        let cap_dims = cap_acts.dims();
        let cap_total: usize = cap_dims[..cap_dims.len() - 1].iter().product();
        if cap_total != total {
            return Err(candle_core::Error::Msg(
                "CapMoeMlp.forward_sparse_top1: token count mismatch".into(),
            ));
        }
        // Bounded grouped dispatch: three SwiGLU projections share one
        // routing state. Each projection becomes one batched matmul
        // plus a (typically empty) overflow loop, instead of n_caps
        // small matmuls. Memory bounded by n_caps × bound × d.
        // Opt-in device-routed path: builds dispatch indices with tensor
        // ops instead of a host round trip, and needs no sort because it
        // works in original token order. Guarded so the default path and
        // every published number are untouched.
        if let (true, Some(bs)) = (
            super::blocksparse::enabled(),
            super::blocksparse::try_blocksparse_routing(cap_acts, self.n_caps, &self.device)?,
        ) {
            let gate = super::blocksparse::apply_blocksparse_projection(
                &h_flat, self.w_gate.as_tensor(), &bs)?;
            let value = super::blocksparse::apply_blocksparse_projection(
                &h_flat, self.w_value.as_tensor(), &bs)?;
            let hidden = silu(&gate)?.mul(&value)?;
            let out_flat = super::blocksparse::apply_blocksparse_projection(
                &hidden, self.w_out.as_tensor(), &bs)?;
            let mut out_dims: Vec<usize> = h_dims[..n_dim - 1].to_vec();
            out_dims.push(self.d_model);
            return out_flat.reshape(out_dims);
        }

        let routing = match routing_in {
            Some(r) => r.clone(),
            None => routing_from_cap_acts(cap_acts, self.n_caps, &self.device)?,
        };

        let h_sorted = h_flat.index_select(&routing.perm_t, 0)?;
        let gate_sorted =
            apply_bounded_grouped_projection(&h_sorted, self.w_gate.as_tensor(), &routing)?;
        let value_sorted =
            apply_bounded_grouped_projection(&h_sorted, self.w_value.as_tensor(), &routing)?;
        let hidden_sorted = silu(&gate_sorted)?.mul(&value_sorted)?;
        let out_sorted =
            apply_bounded_grouped_projection(&hidden_sorted, self.w_out.as_tensor(), &routing)?;
        let out_flat = out_sorted.index_select(&routing.inv_perm_t, 0)?;

        let mut out_dims: Vec<usize> = h_dims[..n_dim - 1].to_vec();
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
                "CapMoeMlp.shrink_caps: cannot shrink to 0".into(),
            ));
        }
        let idx_u32: Vec<u32> = keep_indices.iter().map(|&i| i as u32).collect();
        let idx_t = Tensor::from_vec(idx_u32, (kept,), &self.device)?;
        let new_g = self.w_gate.as_tensor().index_select(&idx_t, 0)?;
        let new_v = self.w_value.as_tensor().index_select(&idx_t, 0)?;
        let new_o = self.w_out.as_tensor().index_select(&idx_t, 0)?;
        let new_var_g = Var::from_tensor(&new_g)?;
        let new_var_v = Var::from_tensor(&new_v)?;
        let new_var_o = Var::from_tensor(&new_o)?;
        self.replace_in_varmap(&[
            (self.w_gate_name.clone(), new_var_g.clone()),
            (self.w_value_name.clone(), new_var_v.clone()),
            (self.w_out_name.clone(), new_var_o.clone()),
        ]);
        self.w_gate = new_var_g;
        self.w_value = new_var_v;
        self.w_out = new_var_o;
        self.n_caps = kept;
        Ok(())
    }

    pub fn grow(&mut self, delta: usize) -> CResult<usize> {
        if delta == 0 {
            return Ok(self.n_caps);
        }
        let new_g = init_3d(
            delta,
            self.d_model,
            self.d_ff,
            self.init_scale_in,
            &self.device,
            self.dtype,
        )?;
        let new_v = init_3d(
            delta,
            self.d_model,
            self.d_ff,
            self.init_scale_in,
            &self.device,
            self.dtype,
        )?;
        let new_o = init_3d(
            delta,
            self.d_ff,
            self.d_model,
            self.init_scale_out,
            &self.device,
            self.dtype,
        )?;
        let cat_g = Tensor::cat(&[self.w_gate.as_tensor(), &new_g], 0)?;
        let cat_v = Tensor::cat(&[self.w_value.as_tensor(), &new_v], 0)?;
        let cat_o = Tensor::cat(&[self.w_out.as_tensor(), &new_o], 0)?;
        let new_var_g = Var::from_tensor(&cat_g)?;
        let new_var_v = Var::from_tensor(&cat_v)?;
        let new_var_o = Var::from_tensor(&cat_o)?;
        self.replace_in_varmap(&[
            (self.w_gate_name.clone(), new_var_g.clone()),
            (self.w_value_name.clone(), new_var_v.clone()),
            (self.w_out_name.clone(), new_var_o.clone()),
        ]);
        self.w_gate = new_var_g;
        self.w_value = new_var_v;
        self.w_out = new_var_o;
        self.n_caps += delta;
        Ok(self.n_caps)
    }

    pub fn reseed_row(&mut self, k: usize) -> CResult<()> {
        if k >= self.n_caps {
            return Ok(());
        }
        let new_g = init_3d(
            1,
            self.d_model,
            self.d_ff,
            self.init_scale_in,
            &self.device,
            self.dtype,
        )?;
        let new_v = init_3d(
            1,
            self.d_model,
            self.d_ff,
            self.init_scale_in,
            &self.device,
            self.dtype,
        )?;
        let new_o = init_3d(
            1,
            self.d_ff,
            self.d_model,
            self.init_scale_out,
            &self.device,
            self.dtype,
        )?;
        let new_w_gate = splice_row(self.w_gate.as_tensor(), k, &new_g, self.n_caps)?;
        let new_w_value = splice_row(self.w_value.as_tensor(), k, &new_v, self.n_caps)?;
        let new_w_out = splice_row(self.w_out.as_tensor(), k, &new_o, self.n_caps)?;
        let new_var_g = Var::from_tensor(&new_w_gate)?;
        let new_var_v = Var::from_tensor(&new_w_value)?;
        let new_var_o = Var::from_tensor(&new_w_out)?;
        self.replace_in_varmap(&[
            (self.w_gate_name.clone(), new_var_g.clone()),
            (self.w_value_name.clone(), new_var_v.clone()),
            (self.w_out_name.clone(), new_var_o.clone()),
        ]);
        self.w_gate = new_var_g;
        self.w_value = new_var_v;
        self.w_out = new_var_o;
        Ok(())
    }

    pub fn var_names(&self) -> [&str; 3] {
        [&self.w_gate_name, &self.w_value_name, &self.w_out_name]
    }

    fn replace_in_varmap(&self, items: &[(String, Var)]) {
        let vm = self.varmap.lock().unwrap();
        let data = vm.data();
        let mut data = data.lock().unwrap();
        for (name, var) in items {
            data.insert(name.clone(), var.clone());
        }
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

/// SiLU = x * sigmoid(x). Via tanh for Metal portability.
fn silu(x: &Tensor) -> CResult<Tensor> {
    let half = (x * 0.5)?;
    let tanh_half = half.tanh()?;
    let sig = ((tanh_half + 1.0)? * 0.5)?;
    x.mul(&sig)
}

fn splice_row(t: &Tensor, k: usize, new_slab: &Tensor, n: usize) -> CResult<Tensor> {
    let pieces: Vec<Tensor> = if k == 0 {
        vec![new_slab.clone(), t.narrow(0, 1, n - 1)?]
    } else if k == n - 1 {
        vec![t.narrow(0, 0, k)?, new_slab.clone()]
    } else {
        vec![
            t.narrow(0, 0, k)?,
            new_slab.clone(),
            t.narrow(0, k + 1, n - k - 1)?,
        ]
    };
    let refs: Vec<&Tensor> = pieces.iter().collect();
    Tensor::cat(&refs, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    fn make_moe(
        n_caps: usize,
        d_model: usize,
        d_ff: usize,
        top_k: usize,
    ) -> (CapMoeMlp, Arc<Mutex<VarMap>>) {
        let varmap = Arc::new(Mutex::new(VarMap::new()));
        let moe = CapMoeMlp::new(
            n_caps,
            d_model,
            d_ff,
            top_k,
            "moe",
            varmap.clone(),
            cpu(),
            DType::F32,
        )
        .unwrap();
        (moe, varmap)
    }

    #[test]
    fn forward_shape() {
        let (moe, _vm) = make_moe(6, 16, 32, 3);
        let h = Tensor::randn(0f32, 1f32, (2, 5, 16), &cpu()).unwrap();
        let caps = Tensor::randn(0f32, 1f32, (2, 5, 6), &cpu()).unwrap();
        let y = moe.forward(&h, &caps).unwrap();
        assert_eq!(y.dims(), &[2, 5, 16]);
    }

    #[test]
    fn forward_shape_soft_no_k() {
        let (moe, _vm) = make_moe(6, 16, 32, 0);
        let h = Tensor::randn(0f32, 1f32, (2, 5, 16), &cpu()).unwrap();
        let caps = Tensor::randn(0f32, 1f32, (2, 5, 6), &cpu()).unwrap();
        let y = moe.forward(&h, &caps).unwrap();
        assert_eq!(y.dims(), &[2, 5, 16]);
    }

    #[test]
    fn forward_shape_sparse_top1() {
        let (moe_orig, _vm) = make_moe(6, 16, 32, 1);
        let moe = moe_orig.with_routing(RoutingMode::HardTop1Sparse);
        let h = Tensor::randn(0f32, 1f32, (2, 5, 16), &cpu()).unwrap();
        let caps = Tensor::randn(0f32, 1f32, (2, 5, 6), &cpu()).unwrap();
        let y = moe.forward(&h, &caps).unwrap();
        assert_eq!(y.dims(), &[2, 5, 16]);
    }

    #[test]
    fn off_domain_cap_slabs_receive_zero_gradient() {
        let (moe, _vm) = make_moe(4, 8, 16, 1);
        let h = Tensor::randn(0f32, 1f32, (1, 3, 8), &cpu()).unwrap();
        let mut cap_data = vec![0.0f32; 12];
        for t in 0..3 {
            cap_data[t * 4 + 0] = 1.0;
        }
        let caps = Tensor::from_vec(cap_data, (1, 3, 4), &cpu()).unwrap();

        let y = moe.forward(&h, &caps).unwrap();
        let loss = y.sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();

        for (var, name) in [
            (moe.w_gate.as_tensor(), "w_gate"),
            (moe.w_value.as_tensor(), "w_value"),
            (moe.w_out.as_tensor(), "w_out"),
        ] {
            let g = grads
                .get(var)
                .unwrap_or_else(|| panic!("no grad for {}", name));
            let g_v: Vec<Vec<Vec<f32>>> = g.to_vec3().unwrap();
            for kk in 1..4 {
                for di in 0..g_v[kk].len() {
                    for dj in 0..g_v[kk][di].len() {
                        assert!(
                            g_v[kk][di][dj].abs() < 1e-9,
                            "{} off-domain cap {} got nonzero grad",
                            name,
                            kk
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn grow_preserves_existing_slabs() {
        let (mut moe, _vm) = make_moe(3, 6, 12, 2);
        let before = moe.w_gate.as_tensor().to_vec3::<f32>().unwrap();
        moe.grow(2).unwrap();
        let after = moe.w_gate.as_tensor().to_vec3::<f32>().unwrap();
        assert_eq!(after.len(), 5);
        for k in 0..3 {
            for di in 0..6 {
                for dj in 0..12 {
                    assert!((before[k][di][dj] - after[k][di][dj]).abs() < 1e-12);
                }
            }
        }
    }
}

#[cfg(test)]
mod blocksparse_parity_tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarMap;
    use std::sync::{Arc, Mutex};

    /// The device-routed MoE path must agree with the host-routed one on
    /// the same weights and inputs. Any gap here is a wiring bug: the
    /// projection itself is covered by
    /// `blocksparse::tests::projection_matches_naive_reference`.
    #[test]
    fn moe_paths_agree() -> CResult<()> {
        let dev = Device::Cpu;
        let (n_caps, d_model, d_ff, total) = (8, 16, 32, 40);
        let varmap = Arc::new(Mutex::new(VarMap::new()));
        let moe = CapMoeMlp::new(
            n_caps, d_model, d_ff, 1, "test", varmap, dev.clone(), DType::F32,
        )?;

        let h = Tensor::randn(0f32, 1f32, (total, d_model), &dev)?;
        // Skewed on purpose so at least one bucket needs a second pass.
        let mut acts = vec![0f32; total * n_caps];
        for t in 0..total {
            acts[t * n_caps + if t < 25 { 0 } else { t % n_caps }] = 1.0;
        }
        let cap_acts = Tensor::from_vec(acts, (total, n_caps), &dev)?;

        std::env::remove_var("AWARE_CN_BLOCKSPARSE");
        let host = moe.forward_sparse_top1(&h, &cap_acts, None)?;
        std::env::set_var("AWARE_CN_BLOCKSPARSE", "1");
        let device_routed = moe.forward_sparse_top1(&h, &cap_acts, None)?;
        std::env::remove_var("AWARE_CN_BLOCKSPARSE");

        let diff = (host - device_routed)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-4, "MoE paths disagree by {diff}");
        Ok(())
    }
}
