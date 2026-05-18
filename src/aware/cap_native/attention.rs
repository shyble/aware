// CapKeyedMha: multi-head attention with per-cap QKV and O projection
// stacks. Attention math is standard scaled-dot-product over heads with
// RoPE; only the projections are cap-keyed.
//   - HardTop1Sparse: argmax winner per token, per-cap matmul on subset.
//
//

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result as CResult, Tensor, Var, D};
use candle_nn::{ops, VarMap};

use super::compression::RoutingMode;
use super::norm::top_k_softmax;
use crate::aware::embed::{build_causal_mask, RoPE};

/// Per-token routing state, threaded between the QKV and O projections.
enum GateState {
    Soft(Tensor),
    Sparse(Vec<Vec<u32>>),
}

pub struct CapKeyedMha {
    pub n_caps: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_head: usize,
    pub top_k: usize,
    pub w_qkv: Var,
    pub w_o: Var,
    pub w_qkv_name: String,
    pub w_o_name: String,
    pub varmap: Arc<Mutex<VarMap>>,
    pub rope: RoPE,
    pub causal_mask: Tensor,
    pub init_scale: f64,
    pub cap_indexed_mask: bool,
    pub device: Device,
    pub dtype: DType,
    pub routing: RoutingMode,
}

impl CapKeyedMha {
    pub fn new(
        n_caps: usize,
        d_model: usize,
        n_heads: usize,
        max_seq_len: usize,
        rope_base: f64,
        top_k: usize,
        prefix: &str,
        varmap: Arc<Mutex<VarMap>>,
        device: Device,
        dtype: DType,
    ) -> CResult<Self> {
        if d_model % n_heads != 0 {
            return Err(candle_core::Error::Msg(format!(
                "CapKeyedMha: d_model {} not divisible by n_heads {}",
                d_model, n_heads
            )));
        }
        let d_head = d_model / n_heads;
        let init_scale = 1.0 / (d_model as f64).sqrt();
        let qkv_data = init_3d(n_caps, d_model, 3 * d_model, init_scale, &device, dtype)?;
        let o_data = init_3d(n_caps, d_model, d_model, init_scale, &device, dtype)?;
        let w_qkv = Var::from_tensor(&qkv_data)?;
        let w_o = Var::from_tensor(&o_data)?;
        let w_qkv_name = format!("{}.w_qkv", prefix);
        let w_o_name = format!("{}.w_o", prefix);
        {
            let vm = varmap.lock().unwrap();
            let data = vm.data();
            let mut data = data.lock().unwrap();
            data.insert(w_qkv_name.clone(), w_qkv.clone());
            data.insert(w_o_name.clone(), w_o.clone());
        }
        let rope = RoPE::new(d_head, max_seq_len, rope_base, &device)?;
        let causal_mask = build_causal_mask(max_seq_len, &device)?;
        Ok(Self {
            n_caps,
            d_model,
            n_heads,
            d_head,
            top_k,
            w_qkv,
            w_o,
            w_qkv_name,
            w_o_name,
            varmap,
            rope,
            causal_mask,
            init_scale,
            cap_indexed_mask: false,
            device,
            dtype,
            routing: RoutingMode::SoftTopK,
        })
    }

    pub fn with_cap_indexed_mask(mut self, enable: bool) -> Self {
        self.cap_indexed_mask = enable;
        self
    }

    pub fn with_routing(mut self, routing: RoutingMode) -> Self {
        self.routing = routing;
        self
    }

    pub fn forward(&self, xs: &Tensor, cap_acts: &Tensor) -> CResult<Tensor> {
        let (b, s, dm) = xs.dims3()?;
        if dm != self.d_model {
            return Err(candle_core::Error::Msg(format!(
                "CapKeyedMha.forward: d_model mismatch xs={} cfg={}",
                dm, self.d_model,
            )));
        }
        let cap_dims = cap_acts.dims();
        let cap_total: usize = cap_dims[..cap_dims.len() - 1].iter().product();
        if cap_total != b * s {
            return Err(candle_core::Error::Msg(format!(
                "CapKeyedMha.forward: token count mismatch xs={} cap_acts={}",
                b * s,
                cap_total,
            )));
        }
        let cap_flat = cap_acts.reshape((b * s, self.n_caps))?;
        let xs_flat = xs.reshape((b * s, self.d_model))?;
        let total = b * s;
        let qkv_dim = 3 * dm;

        // QKV projection (cap-keyed).
        let (qkv_flat, gate_state): (Tensor, _) = match self.routing {
            RoutingMode::SoftTopK => {
                let gate_w_flat = top_k_softmax(&cap_flat, self.top_k)?;
                let mut qkv_accum: Option<Tensor> = None;
                for k in 0..self.n_caps {
                    let w_qkv_k = self.w_qkv.as_tensor().narrow(0, k, 1)?.squeeze(0)?;
                    let qkv_k = xs_flat.matmul(&w_qkv_k)?;
                    let weight_k = gate_w_flat.narrow(D::Minus1, k, 1)?;
                    let weighted = qkv_k.broadcast_mul(&weight_k)?;
                    qkv_accum = Some(match qkv_accum {
                        Some(a) => (a + weighted)?,
                        None => weighted,
                    });
                }
                let qkv_flat = qkv_accum.ok_or_else(|| {
                    candle_core::Error::Msg("CapKeyedMha.forward: zero caps".into())
                })?;
                (qkv_flat, GateState::Soft(gate_w_flat))
            }
            RoutingMode::HardTop1Sparse => {
                let winners_t = cap_flat.argmax(D::Minus1)?;
                let winners: Vec<u32> = winners_t.to_vec1::<u32>()?;
                let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); self.n_caps];
                for (t, &w) in winners.iter().enumerate() {
                    let k = (w as usize).min(self.n_caps - 1);
                    buckets[k].push(t as u32);
                }
                let mut qkv_flat = Tensor::zeros((total, qkv_dim), self.dtype, &self.device)?;
                for k in 0..self.n_caps {
                    if buckets[k].is_empty() {
                        continue;
                    }
                    let n_k = buckets[k].len();
                    let idx_t = Tensor::from_vec(buckets[k].clone(), (n_k,), &self.device)?;
                    let xs_k = xs_flat.index_select(&idx_t, 0)?;
                    let w_qkv_k = self.w_qkv.as_tensor().narrow(0, k, 1)?.squeeze(0)?;
                    let qkv_k = xs_k.matmul(&w_qkv_k)?;
                    qkv_flat = qkv_flat.index_add(&idx_t, &qkv_k, 0)?;
                }
                (qkv_flat, GateState::Sparse(buckets))
            }
        };
        let qkv = qkv_flat.reshape((b, s, qkv_dim))?;

        // Standard attention math.
        let q = qkv
            .narrow(D::Minus1, 0, dm)?
            .reshape((b, s, self.n_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;
        let k_t = qkv
            .narrow(D::Minus1, dm, dm)?
            .reshape((b, s, self.n_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = qkv
            .narrow(D::Minus1, 2 * dm, dm)?
            .reshape((b, s, self.n_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;
        let q = self.rope.apply(&q, s)?;
        let k_t = self.rope.apply(&k_t, s)?;
        let scale = 1.0 / (self.d_head as f64).sqrt();
        let scores = q.matmul(&k_t.transpose(2, 3)?.contiguous()?)?;
        let scores = (scores * scale)?;
        // Apply causal mask (narrow from precomputed max_seq mask).
        let mask = self
            .causal_mask
            .narrow(0, 0, s)?
            .narrow(1, 0, s)?
            .to_dtype(scores.dtype())?;
        let scores = scores.broadcast_add(&mask)?;
        // Cap-indexed mask (optional).
        let scores = if self.cap_indexed_mask {
            let cap_bias = build_cap_overlap_bias(
                cap_acts,
                self.top_k,
                self.n_caps,
                b,
                s,
                &self.device,
                scores.dtype(),
            )?;
            scores.broadcast_add(&cap_bias)?
        } else {
            scores
        };
        let weights = ops::softmax(&scores, D::Minus1)?;
        let attn = weights.matmul(&v)?;
        let attn = attn.transpose(1, 2)?.contiguous()?;
        let attn = attn.reshape((b, s, dm))?;

        // Cap-keyed output projection.
        let attn_flat = attn.reshape((b * s, dm))?;
        let out_flat = match &gate_state {
            GateState::Soft(gate_w_flat) => {
                let mut out_accum: Option<Tensor> = None;
                for k in 0..self.n_caps {
                    let w_o_k = self.w_o.as_tensor().narrow(0, k, 1)?.squeeze(0)?;
                    let out_k = attn_flat.matmul(&w_o_k)?;
                    let weight_k = gate_w_flat.narrow(D::Minus1, k, 1)?;
                    let weighted = out_k.broadcast_mul(&weight_k)?;
                    out_accum = Some(match out_accum {
                        Some(a) => (a + weighted)?,
                        None => weighted,
                    });
                }
                out_accum.ok_or_else(|| candle_core::Error::Msg("zero caps".into()))?
            }
            GateState::Sparse(buckets) => {
                let mut out_flat = Tensor::zeros((total, dm), self.dtype, &self.device)?;
                for k in 0..self.n_caps {
                    if buckets[k].is_empty() {
                        continue;
                    }
                    let n_k = buckets[k].len();
                    let idx_t = Tensor::from_vec(buckets[k].clone(), (n_k,), &self.device)?;
                    let attn_k = attn_flat.index_select(&idx_t, 0)?;
                    let w_o_k = self.w_o.as_tensor().narrow(0, k, 1)?.squeeze(0)?;
                    let out_k = attn_k.matmul(&w_o_k)?;
                    out_flat = out_flat.index_add(&idx_t, &out_k, 0)?;
                }
                out_flat
            }
        };
        out_flat.reshape((b, s, dm))
    }

    pub fn shrink_caps(&mut self, keep_indices: &[usize]) -> CResult<()> {
        let n = self.n_caps;
        let kept = keep_indices.len();
        if kept == n {
            return Ok(());
        }
        if kept == 0 {
            return Err(candle_core::Error::Msg(
                "CapKeyedMha.shrink_caps: cannot shrink to 0".into(),
            ));
        }
        let idx_u32: Vec<u32> = keep_indices.iter().map(|&i| i as u32).collect();
        let idx_t = Tensor::from_vec(idx_u32, (kept,), &self.device)?;
        let new_qkv = self.w_qkv.as_tensor().index_select(&idx_t, 0)?;
        let new_o = self.w_o.as_tensor().index_select(&idx_t, 0)?;
        let new_var_qkv = Var::from_tensor(&new_qkv)?;
        let new_var_o = Var::from_tensor(&new_o)?;
        {
            let vm = self.varmap.lock().unwrap();
            let data = vm.data();
            let mut data = data.lock().unwrap();
            data.insert(self.w_qkv_name.clone(), new_var_qkv.clone());
            data.insert(self.w_o_name.clone(), new_var_o.clone());
        }
        self.w_qkv = new_var_qkv;
        self.w_o = new_var_o;
        self.n_caps = kept;
        Ok(())
    }

    pub fn grow(&mut self, delta: usize) -> CResult<usize> {
        if delta == 0 {
            return Ok(self.n_caps);
        }
        let new_qkv = init_3d(
            delta,
            self.d_model,
            3 * self.d_model,
            self.init_scale,
            &self.device,
            self.dtype,
        )?;
        let new_o = init_3d(
            delta,
            self.d_model,
            self.d_model,
            self.init_scale,
            &self.device,
            self.dtype,
        )?;
        let cat_qkv = Tensor::cat(&[self.w_qkv.as_tensor(), &new_qkv], 0)?;
        let cat_o = Tensor::cat(&[self.w_o.as_tensor(), &new_o], 0)?;
        let new_var_qkv = Var::from_tensor(&cat_qkv)?;
        let new_var_o = Var::from_tensor(&cat_o)?;
        {
            let vm = self.varmap.lock().unwrap();
            let data = vm.data();
            let mut data = data.lock().unwrap();
            data.insert(self.w_qkv_name.clone(), new_var_qkv.clone());
            data.insert(self.w_o_name.clone(), new_var_o.clone());
        }
        self.w_qkv = new_var_qkv;
        self.w_o = new_var_o;
        self.n_caps += delta;
        Ok(self.n_caps)
    }

    pub fn reseed_row(&mut self, k: usize) -> CResult<()> {
        if k >= self.n_caps {
            return Ok(());
        }
        let new_qkv = init_3d(
            1,
            self.d_model,
            3 * self.d_model,
            self.init_scale,
            &self.device,
            self.dtype,
        )?;
        let new_o = init_3d(
            1,
            self.d_model,
            self.d_model,
            self.init_scale,
            &self.device,
            self.dtype,
        )?;
        let new_w_qkv = splice_row(self.w_qkv.as_tensor(), k, &new_qkv, self.n_caps)?;
        let new_w_o = splice_row(self.w_o.as_tensor(), k, &new_o, self.n_caps)?;
        let new_var_qkv = Var::from_tensor(&new_w_qkv)?;
        let new_var_o = Var::from_tensor(&new_w_o)?;
        {
            let vm = self.varmap.lock().unwrap();
            let data = vm.data();
            let mut data = data.lock().unwrap();
            data.insert(self.w_qkv_name.clone(), new_var_qkv.clone());
            data.insert(self.w_o_name.clone(), new_var_o.clone());
        }
        self.w_qkv = new_var_qkv;
        self.w_o = new_var_o;
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

/// Build cap-overlap additive bias for attention scores.
fn build_cap_overlap_bias(
    cap_acts: &Tensor,
    top_k: usize,
    n_caps: usize,
    b: usize,
    s: usize,
    device: &Device,
    dtype: DType,
) -> CResult<Tensor> {
    let cap_flat = cap_acts.reshape((b * s, n_caps))?;
    let (sorted, _) = cap_flat.sort_last_dim(false)?;
    let kk = top_k.min(n_caps).max(1);
    let threshold = sorted.narrow(D::Minus1, kk - 1, 1)?;
    let in_topk_flat = cap_flat.broadcast_ge(&threshold)?.to_dtype(dtype)?;
    let in_topk = in_topk_flat.reshape((b, s, n_caps))?;
    let in_topk_t = in_topk.transpose(1, 2)?.contiguous()?;
    let cap_overlap = in_topk.matmul(&in_topk_t)?;
    let one_t = Tensor::ones((b, s, s), dtype, device)?;
    let allowed = cap_overlap.broadcast_ge(&one_t)?.to_dtype(dtype)?;
    let bias = allowed.affine(-1.0, 1.0)?.affine(-1e9, 0.0)?;
    bias.unsqueeze(1)
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
        n_heads: usize,
        top_k: usize,
    ) -> (CapKeyedMha, Arc<Mutex<VarMap>>) {
        let varmap = Arc::new(Mutex::new(VarMap::new()));
        let mha = CapKeyedMha::new(
            n_caps,
            d_model,
            n_heads,
            32,
            10_000.0,
            top_k,
            "attn",
            varmap.clone(),
            cpu(),
            DType::F32,
        )
        .unwrap();
        (mha, varmap)
    }

    #[test]
    fn forward_shape() {
        let (mha, _vm) = make(6, 16, 2, 3);
        let xs = Tensor::randn(0f32, 1f32, (2, 5, 16), &cpu()).unwrap();
        let caps = Tensor::randn(0f32, 1f32, (2, 5, 6), &cpu()).unwrap();
        let y = mha.forward(&xs, &caps).unwrap();
        assert_eq!(y.dims(), &[2, 5, 16]);
    }

    #[test]
    fn forward_shape_soft_no_k() {
        let (mha, _vm) = make(6, 16, 2, 0);
        let xs = Tensor::randn(0f32, 1f32, (2, 5, 16), &cpu()).unwrap();
        let caps = Tensor::randn(0f32, 1f32, (2, 5, 6), &cpu()).unwrap();
        let y = mha.forward(&xs, &caps).unwrap();
        assert_eq!(y.dims(), &[2, 5, 16]);
    }

    #[test]
    fn forward_shape_sparse_top1() {
        let (mha_orig, _vm) = make(6, 16, 4, 1);
        let mha = mha_orig.with_routing(RoutingMode::HardTop1Sparse);
        let xs = Tensor::randn(0f32, 1f32, (2, 5, 16), &cpu()).unwrap();
        let caps = Tensor::randn(0f32, 1f32, (2, 5, 6), &cpu()).unwrap();
        let y = mha.forward(&xs, &caps).unwrap();
        assert_eq!(y.dims(), &[2, 5, 16]);
    }

    #[test]
    fn off_domain_cap_slabs_receive_zero_gradient() {
        let (mha, _vm) = make(4, 8, 2, 1);
        let xs = Tensor::randn(0f32, 1f32, (1, 3, 8), &cpu()).unwrap();
        let mut cap_data = vec![0.0f32; 12];
        for t in 0..3 {
            cap_data[t * 4 + 0] = 1.0;
        }
        let caps = Tensor::from_vec(cap_data, (1, 3, 4), &cpu()).unwrap();

        let y = mha.forward(&xs, &caps).unwrap();
        let loss = y.sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();

        let g_qkv = grads.get(mha.w_qkv.as_tensor()).expect("no grad");
        let g_o = grads.get(mha.w_o.as_tensor()).expect("no grad");
        let gv_qkv: Vec<Vec<Vec<f32>>> = g_qkv.to_vec3().unwrap();
        let gv_o: Vec<Vec<Vec<f32>>> = g_o.to_vec3().unwrap();

        for kk in 1..4 {
            for di in 0..gv_qkv[kk].len() {
                for dj in 0..gv_qkv[kk][di].len() {
                    assert!(
                        gv_qkv[kk][di][dj].abs() < 1e-9,
                        "off-domain cap {} W_qkv nonzero",
                        kk
                    );
                }
            }
            for di in 0..gv_o[kk].len() {
                for dj in 0..gv_o[kk][di].len() {
                    assert!(
                        gv_o[kk][di][dj].abs() < 1e-9,
                        "off-domain cap {} W_o nonzero",
                        kk
                    );
                }
            }
        }
    }

    #[test]
    fn cap_indexed_mask_forward_runs() {
        let varmap = Arc::new(Mutex::new(VarMap::new()));
        let mha = CapKeyedMha::new(
            4,
            8,
            1,
            16,
            10_000.0,
            1,
            "attn",
            varmap.clone(),
            cpu(),
            DType::F32,
        )
        .unwrap()
        .with_cap_indexed_mask(true);

        let mut cap_data = vec![0.0f32; 16];
        for i in 0..4 {
            cap_data[i * 4 + i] = 1.0;
        }
        let caps = Tensor::from_vec(cap_data, (1, 4, 4), &cpu()).unwrap();
        let xs = Tensor::randn(0f32, 1f32, (1, 4, 8), &cpu()).unwrap();
        let y = mha.forward(&xs, &caps).unwrap();
        assert_eq!(y.dims(), &[1, 4, 8]);
    }

    #[test]
    fn grow_preserves_existing_slabs() {
        let (mut mha, _vm) = make(3, 8, 2, 2);
        let before_qkv = mha.w_qkv.as_tensor().to_vec3::<f32>().unwrap();
        mha.grow(2).unwrap();
        let after_qkv = mha.w_qkv.as_tensor().to_vec3::<f32>().unwrap();
        assert_eq!(after_qkv.len(), 5);
        for k in 0..3 {
            for di in 0..before_qkv[k].len() {
                for dj in 0..before_qkv[k][di].len() {
                    assert!((before_qkv[k][di][dj] - after_qkv[k][di][dj]).abs() < 1e-12);
                }
            }
        }
    }
}
