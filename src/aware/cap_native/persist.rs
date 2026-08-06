//! Checkpointing for cap-native substrates.
//!
//! A cap-native model is NOT fully described by its `VarMap`: the cap
//! keys of layer 0 and layer 1 are frozen at discovery time and live
//! outside it (see the parameter accounting in `substrate.rs`), and the
//! cap ids and metadata are what make caps *identifiable* at all. A
//! checkpoint that captured only the varmap would load without error and
//! silently rediscover different caps — invalidating any experiment
//! built on cap identity while appearing to work. This module exists so
//! that failure mode cannot happen.
//!
//! Format: one safetensors file holding every varmap tensor under its
//! usual name plus the cap keys under reserved `__capnative__.*` names
//! (`VarMap::load` iterates its own vars, so the extras are ignored by
//! it and read separately), and a bincode sidecar `<path>.meta` with the
//! ids and `CapMeta` records at full fidelity.
//!
//! Loading is strict by design: every expected tensor must be present
//! with the expected shape, the hierarchical-ness of the checkpoint must
//! match the model, and the metadata sidecar must exist and agree on cap
//! counts. Anything less is an error, never a warning.

use std::collections::HashMap;

use candle_core::{Result as CResult, Tensor};
use serde::{Deserialize, Serialize};

use super::substrate::CapNativeSubstrate;
use crate::aware::cap::CapMeta;

const L0_KEYS: &str = "__capnative__.l0.keys";
const L0_VALUES: &str = "__capnative__.l0.values";
const L1_KEYS: &str = "__capnative__.l1.keys";
const L1_VALUES: &str = "__capnative__.l1.values";

#[derive(Serialize, Deserialize)]
struct LayerMeta {
    ids: Vec<u64>,
    metadata: Vec<CapMeta>,
}

#[derive(Serialize, Deserialize)]
struct CkptMeta {
    version: u32,
    l0: LayerMeta,
    l1: Option<LayerMeta>,
    /// Per-cap decisions. Absent in pre-cycle checkpoints, which load
    /// fine — a model that has never run a cycle has nothing to record.
    #[serde(default)]
    registry: Option<super::registry::CommitRegistry>,
}

fn msg(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// Save every trainable tensor, the frozen cap keys of both layers, and
/// the cap ids/metadata. `path` is the safetensors file; a `<path>.meta`
/// sidecar is written next to it.
pub fn save_checkpoint(model: &CapNativeSubstrate, path: &str) -> CResult<()> {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    {
        let vm = model.varmap.lock().unwrap();
        let data = vm.data().lock().unwrap();
        for (name, var) in data.iter() {
            tensors.insert(name.clone(), var.as_tensor().clone());
        }
    }

    tensors.insert(L0_KEYS.to_string(), model.cap_layer.caps.keys.clone());
    if let Some(v) = &model.cap_layer.caps.values {
        tensors.insert(L0_VALUES.to_string(), v.clone());
    }
    if let Some(l1) = &model.cap_layer_1 {
        tensors.insert(L1_KEYS.to_string(), l1.caps.keys.clone());
        if let Some(v) = &l1.caps.values {
            tensors.insert(L1_VALUES.to_string(), v.clone());
        }
    }

    // During a continual-learning cycle the varmap holds DELTAS and the
    // committed bases live outside it. Saving only the varmap would
    // persist the provisional half of every weight and silently lose all
    // committed knowledge — the same class of failure as dropping the
    // frozen cap keys.
    if model.in_cycle() {
        for (bi, block) in model.blocks.iter().enumerate() {
            let p = format!("__capnative__.base.block.{bi}");
            if let Some(t) = &block.attn.qkv_base {
                tensors.insert(format!("{p}.attn.w_qkv"), t.clone());
            }
            if let Some(t) = &block.attn.o_base {
                tensors.insert(format!("{p}.attn.w_o"), t.clone());
            }
            if let Some(t) = &block.moe.gate_base {
                tensors.insert(format!("{p}.moe.w_gate"), t.clone());
            }
            if let Some(t) = &block.moe.value_base {
                tensors.insert(format!("{p}.moe.w_value"), t.clone());
            }
            if let Some(t) = &block.moe.out_base {
                tensors.insert(format!("{p}.moe.w_out"), t.clone());
            }
            if let Some(t) = &block.norm1.weight_base {
                tensors.insert(format!("{p}.norm1.weight"), t.clone());
            }
            if let Some(t) = &block.norm2.weight_base {
                tensors.insert(format!("{p}.norm2.weight"), t.clone());
            }
        }
        if let Some(t) = &model.final_norm.weight_base {
            tensors.insert("__capnative__.base.final_norm.weight".into(), t.clone());
        }
        if let Some(t) = &model.output.w_base {
            tensors.insert("__capnative__.base.output.w".into(), t.clone());
        }
    }

    candle_core::safetensors::save(&tensors, path)?;

    let meta = CkptMeta {
        version: 1,
        l0: LayerMeta {
            ids: model.cap_layer.caps.ids.clone(),
            metadata: model.cap_layer.caps.metadata.clone(),
        },
        l1: model.cap_layer_1.as_ref().map(|l1| LayerMeta {
            ids: l1.caps.ids.clone(),
            metadata: l1.caps.metadata.clone(),
        }),
        registry: if model.registry.is_empty() {
            None
        } else {
            Some(model.registry.clone())
        },
    };
    let bytes = bincode::serialize(&meta).map_err(msg)?;
    std::fs::write(format!("{path}.meta"), bytes).map_err(msg)?;
    Ok(())
}

/// Load a checkpoint saved by [`save_checkpoint`] into a freshly built
/// model of the same configuration. The model should be built with
/// `NoDiscovery` and no bootstrap sample — whatever keys it was born
/// with are overwritten here, and running KMeans first would only waste
/// time.
///
/// Strict: any missing tensor, shape mismatch, hierarchical mismatch, or
/// absent/inconsistent metadata sidecar is an error.
pub fn load_checkpoint(model: &mut CapNativeSubstrate, path: &str) -> CResult<()> {
    // 1. Trainable tensors. VarMap::load fails on any missing name and
    //    writes in place, so every module holding a clone of a var's
    //    tensor sees the loaded values.
    {
        let mut vm = model.varmap.lock().unwrap();
        vm.load(path)?;
    }

    // 2. Frozen cap keys, read from the same file.
    let all = candle_core::safetensors::load(path, &model.device)?;

    let take = |name: &str| -> CResult<Tensor> {
        all.get(name)
            .cloned()
            .ok_or_else(|| msg(format!("checkpoint {path} is missing {name}")))
    };
    let expect_shape = |name: &str, got: &Tensor, want: &Tensor| -> CResult<()> {
        if got.dims() != want.dims() {
            return Err(msg(format!(
                "checkpoint {path}: {name} has shape {:?}, model expects {:?}",
                got.dims(),
                want.dims()
            )));
        }
        Ok(())
    };

    let l0_keys = take(L0_KEYS)?;
    expect_shape(L0_KEYS, &l0_keys, &model.cap_layer.caps.keys)?;
    model.cap_layer.caps.keys = l0_keys;
    if model.cap_layer.caps.values.is_some() {
        let v = take(L0_VALUES)?;
        model.cap_layer.caps.values = Some(v);
    }

    match (&mut model.cap_layer_1, all.contains_key(L1_KEYS)) {
        (Some(l1), true) => {
            let l1_keys = take(L1_KEYS)?;
            expect_shape(L1_KEYS, &l1_keys, &l1.caps.keys)?;
            l1.caps.keys = l1_keys;
            if l1.caps.values.is_some() {
                l1.caps.values = Some(take(L1_VALUES)?);
            }
        }
        (None, false) => {}
        (Some(_), false) => {
            return Err(msg(format!(
                "checkpoint {path} has no layer-1 caps but the model is hierarchical"
            )))
        }
        (None, true) => {
            return Err(msg(format!(
                "checkpoint {path} contains layer-1 caps but the model is not hierarchical"
            )))
        }
    }

    // 3. Identity: ids and metadata. Without these the caps are tensors,
    //    not identifiable units.
    let meta_path = format!("{path}.meta");
    let bytes = std::fs::read(&meta_path)
        .map_err(|e| msg(format!("checkpoint sidecar {meta_path}: {e}")))?;
    let meta: CkptMeta = bincode::deserialize(&bytes).map_err(msg)?;

    let restore = |layer: &mut crate::aware::cap::CapMatrix,
                   lm: &LayerMeta,
                   which: &str|
     -> CResult<()> {
        if lm.ids.len() != layer.n_caps() || lm.metadata.len() != layer.n_caps() {
            return Err(msg(format!(
                "checkpoint sidecar {meta_path}: {which} has {} ids / {} metadata, model has {} caps",
                lm.ids.len(),
                lm.metadata.len(),
                layer.n_caps()
            )));
        }
        layer.ids = lm.ids.clone();
        layer.metadata = lm.metadata.clone();
        Ok(())
    };

    // Committed bases, if this checkpoint was taken mid-cycle. Their
    // presence is what tells us the model was split.
    if all.contains_key("__capnative__.base.output.w") {
        for (bi, block) in model.blocks.iter_mut().enumerate() {
            let p = format!("__capnative__.base.block.{bi}");
            block.attn.qkv_base = all.get(&format!("{p}.attn.w_qkv")).cloned();
            block.attn.o_base = all.get(&format!("{p}.attn.w_o")).cloned();
            block.moe.gate_base = all.get(&format!("{p}.moe.w_gate")).cloned();
            block.moe.value_base = all.get(&format!("{p}.moe.w_value")).cloned();
            block.moe.out_base = all.get(&format!("{p}.moe.w_out")).cloned();
            block.norm1.weight_base = all.get(&format!("{p}.norm1.weight")).cloned();
            block.norm2.weight_base = all.get(&format!("{p}.norm2.weight")).cloned();
        }
        model.final_norm.weight_base =
            all.get("__capnative__.base.final_norm.weight").cloned();
        model.output.w_base = all.get("__capnative__.base.output.w").cloned();
    }

    restore(&mut model.cap_layer.caps, &meta.l0, "layer 0")?;
    match (&mut model.cap_layer_1, &meta.l1) {
        (Some(l1), Some(lm)) => restore(&mut l1.caps, lm, "layer 1")?,
        (None, None) => {}
        _ => {
            return Err(msg(format!(
                "checkpoint sidecar {meta_path}: hierarchical-ness disagrees with model"
            )))
        }
    }

    if let Some(reg) = meta.registry {
        model.registry = reg;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aware::cap_native::config::CapNativeConfig;
    use crate::aware::cap_native::CapNativeBuilder;
    use candle_core::{DType, Device, Tensor};

    fn small_config(discovery: crate::aware::DiscoveryKind) -> CapNativeConfig {
        let mut cfg = CapNativeConfig::default();
        cfg.vocab = 64;
        cfg.d_model = 16;
        cfg.n_blocks = 2;
        cfg.d_ff = 32;
        cfg.cap_config.n_caps_target = 12;
        cfg.cap_config.cap_window = 3;
        cfg.cap_config.discovery = discovery;
        cfg.hierarchical = true;
        cfg.cap_layer_1.n_caps_target = 8;
        cfg.cap_layer_1.cap_window = 1;
        cfg.cap_layer_1.discovery = discovery;
        cfg
    }

    /// The acceptance test in miniature: a KMeans-discovered model must
    /// round-trip through save/load into a NoDiscovery-built shell and
    /// produce BIT-IDENTICAL logits, with ids and metadata preserved.
    /// Anything less means the checkpoint missed a tensor — exactly the
    /// silent failure this module exists to prevent.
    #[test]
    fn roundtrip_is_bit_identical() -> CResult<()> {
        let dev = Device::Cpu;
        let bootstrap: Vec<u32> = (0..600u32).map(|i| i * 7 % 64).collect();

        let model_a = CapNativeBuilder::default()
            .with_config(small_config(crate::aware::DiscoveryKind::KMeans))
            .with_device(dev.clone())
            .with_bootstrap_sample_tokens(bootstrap)
            .build()?;

        let batch = Tensor::from_vec(
            (0..32u32).map(|i| i % 64).collect::<Vec<_>>(),
            (2, 16),
            &dev,
        )?
        .to_dtype(DType::U32)?;
        let logits_a = model_a.forward(&batch)?;

        let dir = std::env::temp_dir().join("capnative_persist_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ckpt.safetensors");
        let path = path.to_str().unwrap();
        save_checkpoint(&model_a, path)?;

        // Fresh shell: same shapes, no discovery, no bootstrap.
        let mut model_b = CapNativeBuilder::default()
            .with_config(small_config(crate::aware::DiscoveryKind::NoDiscovery))
            .with_device(dev.clone())
            .build()?;
        load_checkpoint(&mut model_b, path)?;

        let logits_b = model_b.forward(&batch)?;
        let diff = (&logits_a - &logits_b)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(diff, 0.0, "logits differ after round-trip: {diff}");

        assert_eq!(model_a.cap_layer.caps.ids, model_b.cap_layer.caps.ids);
        let l1a = model_a.cap_layer_1.as_ref().unwrap();
        let l1b = model_b.cap_layer_1.as_ref().unwrap();
        assert_eq!(l1a.caps.ids, l1b.caps.ids);
        assert_eq!(
            l1a.caps.metadata.len(),
            l1b.caps.metadata.len()
        );
        Ok(())
    }

    /// A hierarchical checkpoint must refuse to load into a
    /// single-discovery model, loudly.
    #[test]
    fn hierarchical_mismatch_fails() -> CResult<()> {
        let dev = Device::Cpu;
        let bootstrap: Vec<u32> = (0..600u32).map(|i| i * 7 % 64).collect();
        let model_a = CapNativeBuilder::default()
            .with_config(small_config(crate::aware::DiscoveryKind::KMeans))
            .with_device(dev.clone())
            .with_bootstrap_sample_tokens(bootstrap)
            .build()?;
        let dir = std::env::temp_dir().join("capnative_persist_test2");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ckpt.safetensors");
        let path = path.to_str().unwrap();
        save_checkpoint(&model_a, path)?;

        let mut flat_cfg = small_config(crate::aware::DiscoveryKind::NoDiscovery);
        flat_cfg.hierarchical = false;
        let mut model_b = CapNativeBuilder::default()
            .with_config(flat_cfg)
            .with_device(dev)
            .build()?;
        assert!(load_checkpoint(&mut model_b, path).is_err());
        Ok(())
    }
}

#[cfg(test)]
mod cycle_persist_tests {
    use super::*;
    use crate::aware::cap_native::config::CapNativeConfig;
    use crate::aware::cap_native::registry::CapDecision;
    use crate::aware::cap_native::CapNativeBuilder;
    use candle_core::{DType, Device, Tensor};

    fn cfg(disc: crate::aware::DiscoveryKind) -> CapNativeConfig {
        let mut c = CapNativeConfig::default();
        c.vocab = 48; c.d_model = 16; c.n_blocks = 2; c.d_ff = 32; c.n_heads = 2;
        c.cap_config.n_caps_target = 8;
        c.cap_config.cap_window = 2;
        c.cap_config.discovery = disc;
        c.routing = crate::aware::cap_native::compression::RoutingMode::HardTop1Sparse;
        c.top_k = 1;
        c
    }

    /// D2 acceptance: a checkpoint taken MID-CYCLE must round-trip
    /// exactly — committed bases, live deltas and per-cap decisions.
    /// Saving only the varmap here would persist the provisional half of
    /// every weight and lose all committed knowledge silently.
    #[test]
    fn mid_cycle_checkpoint_roundtrips() -> CResult<()> {
        let dev = Device::Cpu;
        let mut a = CapNativeBuilder::default()
            .with_config(cfg(crate::aware::DiscoveryKind::Random))
            .with_device(dev.clone())
            .build()?;
        a.begin_cycle()?;

        // A cycle that learned something, then judged two caps.
        for block in &a.blocks {
            let d = block.attn.w_qkv.as_tensor();
            block.attn.w_qkv.set(&Tensor::randn(0f32, 0.1f32, d.dims(), &dev)?)?;
        }
        a.commit_cap(1)?;
        a.rollback_cap(2)?;

        let batch = Tensor::from_vec((0..24u32).map(|i| i % 48).collect::<Vec<_>>(), (2, 12), &dev)?
            .to_dtype(DType::U32)?;
        let logits_a = a.forward(&batch)?.copy()?;

        let dir = std::env::temp_dir().join("capnative_cycle_ckpt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.safetensors");
        let path = path.to_str().unwrap();
        save_checkpoint(&a, path)?;

        let mut b = CapNativeBuilder::default()
            .with_config(cfg(crate::aware::DiscoveryKind::NoDiscovery))
            .with_device(dev)
            .build()?;
        load_checkpoint(&mut b, path)?;

        assert!(b.in_cycle(), "loaded model lost its cycle state");
        let diff = (logits_a - b.forward(&batch)?)?
            .abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(diff, 0.0, "mid-cycle round-trip changed the forward pass: {diff}");

        assert_eq!(b.registry.decision(1), CapDecision::Committed);
        assert_eq!(b.registry.decision(2), CapDecision::RolledBack);
        Ok(())
    }

    /// Untouched caps must be distinguishable from refused ones: a cap
    /// that never fired has an exactly-zero delta.
    #[test]
    fn delta_magnitude_separates_untouched_from_refused() -> CResult<()> {
        let dev = Device::Cpu;
        let mut m = CapNativeBuilder::default()
            .with_config(cfg(crate::aware::DiscoveryKind::Random))
            .with_device(dev.clone())
            .build()?;
        m.begin_cycle()?;
        assert_eq!(m.cap_delta_magnitude(0)?, 0.0, "fresh cycle should have zero deltas");

        let d = m.blocks[0].attn.w_qkv.as_tensor();
        m.blocks[0].attn.w_qkv.set(&Tensor::ones(d.dims(), d.dtype(), &dev)?)?;
        assert!(m.cap_delta_magnitude(0)? > 0.0);
        m.rollback_cap(0)?;
        assert_eq!(m.cap_delta_magnitude(0)?, 0.0, "rollback should zero the delta");
        Ok(())
    }
}
