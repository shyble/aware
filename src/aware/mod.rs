//! AWARE substrates and primitives.

pub mod attention;
pub mod cap;
pub mod cap_native;
pub mod concept;
pub mod concept_layer;
pub mod config;
pub mod discover;
pub mod embed;
pub mod layer;
pub mod norm;
pub mod substrate;
pub mod train;

pub use attention::{Attention, AttentionKind, CapMatrixSource};
pub use cap::{Cap, CapKind, CapMatrix, CapMeta};
pub use concept::{Concept, ConceptStore};
pub use concept_layer::{ConceptConfig, ConceptLayer};
pub use config::{AuditConfig, BlockConfig, CapConfig, Config};
pub use discover::{AuditReport, Discovery, DiscoveryKind};
pub use substrate::{BlockBuilder, CapStats, LayerCapStats, Substrate, SubstrateBuilder};
pub use train::{LossKind, OptimizerKind, StreamingFeeder, Trainer};
