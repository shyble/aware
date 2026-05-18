// Discovery strategies.
//
// Pluggable: each strategy implements bootstrap() (initial cap matrix from
// data or randomness) and step() (periodic audit / replacement / growth).
// step() is a no-op for strategies that don't audit.

pub mod audit;
pub mod kmeans;
pub mod no_discovery;
pub mod random_init;

use serde::{Deserialize, Serialize};

use super::cap::CapMatrix;
use super::config::AuditConfig;
use candle_core::{Device, Result, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryKind {
    /// k-means cluster centroids over bootstrap data; audit replaces dead.
    KMeans,
    /// Random unit vectors; no audit.
    Random,
    /// Xavier random init; gradient does all the work; no audit.
    NoDiscovery,
    /// k-means seed + gradient + audit (most expressive, complex).
    Hybrid,
}

/// Context passed to discovery operations.
pub struct DiscoveryCtx<'a> {
    pub device: &'a Device,
    pub d_in: usize,
    pub d_out: Option<usize>,
    pub n_caps_target: usize,
    /// Sample of input vectors (post-embedding) to cluster.
    /// Shape: [n_samples, d_in]. Required for KMeans/Hybrid bootstrap.
    pub sample: Option<&'a Tensor>,
    pub audit: &'a AuditConfig,
    pub training_step: u64,
}

#[derive(Debug, Clone, Default)]
pub struct AuditReport {
    pub n_dead_detected: usize,
    pub n_novel_inputs_found: usize,
    pub n_replaced: usize,
    pub n_grown: usize,
}

impl AuditReport {
    pub fn empty() -> Self {
        Self::default()
    }
}

/// Discovery trait. Bootstrap sets initial caps; step optionally audits.
pub trait Discovery: Send + Sync {
    fn bootstrap(&self, ctx: &DiscoveryCtx) -> Result<CapMatrix>;
    fn step(&self, _caps: &mut CapMatrix, _ctx: &DiscoveryCtx) -> AuditReport {
        AuditReport::empty()
    }
}

/// Factory: build a Discovery from a DiscoveryKind.
pub fn make_discovery(kind: DiscoveryKind) -> Box<dyn Discovery> {
    match kind {
        DiscoveryKind::KMeans => Box::new(kmeans::KMeansDiscovery::default()),
        DiscoveryKind::Random => Box::new(random_init::RandomDiscovery::default()),
        DiscoveryKind::NoDiscovery => Box::new(no_discovery::NoDiscovery::default()),
        DiscoveryKind::Hybrid => Box::new(kmeans::KMeansDiscovery::hybrid()),
    }
}
