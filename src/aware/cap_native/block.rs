use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result as CResult, Tensor};
use candle_nn::VarMap;

use super::attention::CapKeyedMha;
use super::compression::RoutingMode;
use super::moe::CapMoeMlp;
use super::norm::CapKeyedRmsNorm;

pub struct CapNativeBlockConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub rope_base: f64,
    pub rms_eps: f64,
    pub n_caps: usize,
    pub top_k: usize,
    pub cap_indexed_mask: bool,
    pub routing: RoutingMode,
}

pub struct CapNativeBlock {
    pub norm1: CapKeyedRmsNorm,
    pub attn: CapKeyedMha,
    pub norm2: CapKeyedRmsNorm,
    pub moe: CapMoeMlp,
    pub d_model: usize,
    pub block_index: usize,
}

impl CapNativeBlock {
    pub fn new(
        block_index: usize,
        cfg: &CapNativeBlockConfig,
        prefix: &str,
        varmap: Arc<Mutex<VarMap>>,
        device: Device,
        dtype: DType,
    ) -> CResult<Self> {
        let norm1 = CapKeyedRmsNorm::new(
            cfg.n_caps,
            cfg.d_model,
            cfg.top_k,
            cfg.rms_eps,
            &format!("{}.norm1", prefix),
            varmap.clone(),
            device.clone(),
            dtype,
        )?;
        let attn = CapKeyedMha::new(
            cfg.n_caps,
            cfg.d_model,
            cfg.n_heads,
            cfg.max_seq_len,
            cfg.rope_base,
            cfg.top_k,
            &format!("{}.attn", prefix),
            varmap.clone(),
            device.clone(),
            dtype,
        )?
        .with_routing(cfg.routing)
        .with_cap_indexed_mask(cfg.cap_indexed_mask);
        let norm2 = CapKeyedRmsNorm::new(
            cfg.n_caps,
            cfg.d_model,
            cfg.top_k,
            cfg.rms_eps,
            &format!("{}.norm2", prefix),
            varmap.clone(),
            device.clone(),
            dtype,
        )?;
        let moe = CapMoeMlp::new(
            cfg.n_caps,
            cfg.d_model,
            cfg.d_ff,
            cfg.top_k,
            &format!("{}.moe", prefix),
            varmap.clone(),
            device.clone(),
            dtype,
        )?
        .with_routing(cfg.routing);

        Ok(Self {
            norm1,
            attn,
            norm2,
            moe,
            d_model: cfg.d_model,
            block_index,
        })
    }

    /// Forward: h (B, S, d_model) + cap_acts (B, S, n_caps) -> h_out (B, S, d_model).
    /// `cap_acts` is the same signal threaded through every block - from
    /// the substrate-level discovered CapLayer.
    pub fn forward(&self, h: &Tensor, cap_acts: &Tensor) -> CResult<Tensor> {
        let normed1 = self.norm1.forward(h, cap_acts)?;
        let attn_out = self.attn.forward(&normed1, cap_acts)?;
        let h = (h + attn_out)?;
        let normed2 = self.norm2.forward(&h, cap_acts)?;
        let moe_out = self.moe.forward(&normed2, cap_acts)?;
        h + moe_out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    fn cfg() -> CapNativeBlockConfig {
        CapNativeBlockConfig {
            d_model: 16,
            n_heads: 2,
            d_ff: 32,
            max_seq_len: 64,
            rope_base: 10_000.0,
            rms_eps: 1e-5,
            n_caps: 8,
            top_k: 0,
            cap_indexed_mask: false,
            routing: RoutingMode::SoftTopK,
        }
    }

    #[test]
    fn block_forward_shape() {
        let varmap = Arc::new(Mutex::new(VarMap::new()));
        let block =
            CapNativeBlock::new(0, &cfg(), "block.0", varmap.clone(), cpu(), DType::F32).unwrap();
        let h = Tensor::randn(0f32, 1f32, (1, 4, 16), &cpu()).unwrap();
        let cap_acts = Tensor::randn(0f32, 1f32, (1, 4, 8), &cpu()).unwrap();
        let ys = block.forward(&h, &cap_acts).unwrap();
        assert_eq!(ys.dims(), &[1, 4, 16]);
    }

    #[test]
    fn two_block_pipeline() {
        let varmap = Arc::new(Mutex::new(VarMap::new()));
        let b0 =
            CapNativeBlock::new(0, &cfg(), "block.0", varmap.clone(), cpu(), DType::F32).unwrap();
        let b1 =
            CapNativeBlock::new(1, &cfg(), "block.1", varmap.clone(), cpu(), DType::F32).unwrap();
        let h = Tensor::randn(0f32, 1f32, (1, 4, 16), &cpu()).unwrap();
        let cap_acts = Tensor::randn(0f32, 1f32, (1, 4, 8), &cpu()).unwrap();
        let h0 = b0.forward(&h, &cap_acts).unwrap();
        let h1 = b1.forward(&h0, &cap_acts).unwrap();
        assert_eq!(h1.dims(), &[1, 4, 16]);
    }
}
