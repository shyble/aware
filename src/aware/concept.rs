// Concept store.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read, Write};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    pub id: u64,
    pub centroid: Vec<f32>,
    pub layer_idx: usize,
    pub created_at_step: u64,
    pub last_active_step: u64,
    pub n_observations: u64,
    pub activation_ema: f64,
    pub label: Option<String>,
    pub frozen: bool,
    pub cap_members: Vec<(usize, u64, f64)>,
}

impl Concept {
    pub fn new(id: u64, centroid: Vec<f32>, layer_idx: usize, step: u64) -> Self {
        Self {
            id,
            centroid: normalize(centroid),
            layer_idx,
            created_at_step: step,
            last_active_step: step,
            n_observations: 1,
            activation_ema: 1.0,
            label: None,
            frozen: false,
            cap_members: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptStore {
    pub concepts: Vec<Concept>,
    pub next_id: u64,
    pub probe_layer: usize,
    pub similarity_threshold: f64,
    pub max_concepts: usize,
    pub centroid_ema_alpha: f64,
    pub activation_ema_alpha: f64,
}

impl ConceptStore {
    pub fn new(probe_layer: usize, max_concepts: usize, similarity_threshold: f64) -> Self {
        Self {
            concepts: Vec::new(),
            next_id: 1,
            probe_layer,
            similarity_threshold,
            max_concepts,
            centroid_ema_alpha: 0.98,
            activation_ema_alpha: 0.95,
        }
    }

    pub fn match_or_create(&mut self, hidden: &[f32], step: u64) -> Option<u64> {
        if hidden.is_empty() {
            return None;
        }
        let h = normalize(hidden.to_vec());
        let mut best_idx: Option<usize> = None;
        let mut best_cos: f32 = f32::NEG_INFINITY;
        for (i, c) in self.concepts.iter().enumerate() {
            if c.centroid.len() != h.len() {
                continue;
            }
            let cos: f32 = c.centroid.iter().zip(h.iter()).map(|(a, b)| a * b).sum();
            if cos > best_cos {
                best_cos = cos;
                best_idx = Some(i);
            }
        }
        if best_cos as f64 >= self.similarity_threshold {
            let idx = best_idx.unwrap();
            let c = &mut self.concepts[idx];
            let alpha = self.centroid_ema_alpha as f32;
            for (cv, hv) in c.centroid.iter_mut().zip(h.iter()) {
                *cv = *cv * alpha + *hv * (1.0 - alpha);
            }
            c.centroid = normalize(std::mem::take(&mut c.centroid));
            c.last_active_step = step;
            c.n_observations += 1;
            c.activation_ema =
                c.activation_ema * self.activation_ema_alpha + (1.0 - self.activation_ema_alpha);
            Some(c.id)
        } else if self.concepts.len() < self.max_concepts {
            let id = self.next_id;
            self.next_id += 1;
            self.concepts
                .push(Concept::new(id, h, self.probe_layer, step));
            Some(id)
        } else {
            None
        }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let bytes = bincode::serialize(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut f = fs::File::create(path)?;
        f.write_all(&bytes)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut f = fs::File::open(path)?;
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes)?;
        bincode::deserialize(&bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in v.iter_mut() {
        *x /= norm;
    }
    v
}
