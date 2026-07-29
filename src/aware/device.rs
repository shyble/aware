// Backend-aware default device selection.
//
// Binaries and examples should call `default_device()` rather than
// hard-coding `Device::Cpu`. The right device is then chosen by
// build-time feature flags and (optionally) the `AWARE_DEVICE` env
// var, so cargo's `--features metal|cuda|candle` selection flows
// through to every entry point without per-example edits.
//
// Resolution order (first hit wins):
//   1. `AWARE_DEVICE` env var (`cpu`, `metal`, `cuda`) — explicit override
//   2. Build-time feature: `cuda`, then `metal`, then `candle`/CPU
//
// On systems without the requested backend the function falls back to
// CPU with a stderr warning rather than panicking.

use candle_core::{Device, Result};

/// Resolve the default device for cap-native and substrate workloads.
pub fn default_device() -> Result<Device> {
    if let Ok(forced) = std::env::var("AWARE_DEVICE") {
        return resolve_named(&forced);
    }

    #[cfg(feature = "cuda")]
    {
        return Device::new_cuda(0).or_else(|e| {
            eprintln!("[aware] CUDA requested but unavailable ({}); using CPU.", e);
            Ok(Device::Cpu)
        });
    }

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return Device::new_metal(0).or_else(|e| {
            eprintln!("[aware] Metal requested but unavailable ({}); using CPU.", e);
            Ok(Device::Cpu)
        });
    }

    #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
    Ok(Device::Cpu)
}

fn resolve_named(name: &str) -> Result<Device> {
    match name.to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0)
            }
            #[cfg(not(feature = "metal"))]
            {
                eprintln!(
                    "[aware] AWARE_DEVICE=metal requested but binary not built with \
                     --features metal; falling back to CPU."
                );
                Ok(Device::Cpu)
            }
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0)
            }
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!(
                    "[aware] AWARE_DEVICE=cuda requested but binary not built with \
                     --features cuda; falling back to CPU."
                );
                Ok(Device::Cpu)
            }
        }
        other => Err(candle_core::Error::Msg(format!(
            "AWARE_DEVICE: unknown backend '{}' (expected cpu|metal|cuda)",
            other
        ))),
    }
}

/// Human-readable name for the resolved device, used in log banners.
pub fn device_label(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Metal(_) => "metal",
        Device::Cuda(_) => "cuda",
    }
}
