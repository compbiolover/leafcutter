//! On-disk cache of null-model fits. The null fit of a cluster depends on the samples, their
//! counts, the confounders and the fit settings, but not on the group labels, so a portal
//! that repeatedly tests different groupings of one cohort can reuse it (the R API exposes
//! the same idea through `fit_null` for sQTL mapping).
//!
//! The cache is keyed on a caller-supplied cohort fingerprint (sample names + confounders +
//! parameters) and, per cluster, on the introns and samples that survived filtering and the
//! null design values.

use crate::ds::PreparedCluster;
use crate::fit::Fit;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Serialize, Deserialize, Default)]
struct CacheFile {
    cohort_key: u64,
    entries: HashMap<String, (u64, Fit)>,
}

pub struct NullCache {
    path: Option<PathBuf>,
    cohort_key: u64,
    entries: Mutex<HashMap<String, (u64, Fit)>>,
    dirty: Mutex<bool>,
}

/// Hash helper for building cohort keys.
pub fn fingerprint<T: Hash>(t: &T) -> u64 {
    let mut h = DefaultHasher::new();
    t.hash(&mut h);
    h.finish()
}

fn cluster_key(prep: &PreparedCluster) -> u64 {
    let mut h = DefaultHasher::new();
    prep.introns.hash(&mut h);
    prep.samples_used.hash(&mut h);
    prep.null_cols.hash(&mut h);
    let x_null = prep.x_full.select_columns(&prep.null_cols);
    for v in &x_null.data {
        v.to_bits().hash(&mut h);
    }
    h.finish()
}

impl NullCache {
    /// In-memory cache (not persisted).
    pub fn in_memory(cohort_key: u64) -> Self {
        NullCache {
            path: None,
            cohort_key,
            entries: Mutex::new(HashMap::new()),
            dirty: Mutex::new(false),
        }
    }

    /// Open (or create) a cache file. An existing file for a different cohort is ignored and
    /// overwritten on save.
    pub fn open(path: &Path, cohort_key: u64) -> Self {
        let mut entries = HashMap::new();
        if let Ok(bytes) = std::fs::read(path) {
            if let Ok(cf) = serde_json::from_slice::<CacheFile>(&bytes) {
                if cf.cohort_key == cohort_key {
                    entries = cf.entries;
                }
            }
        }
        NullCache {
            path: Some(path.to_path_buf()),
            cohort_key,
            entries: Mutex::new(entries),
            dirty: Mutex::new(false),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, cluster: &str, prep: &PreparedCluster) -> Option<Fit> {
        let key = cluster_key(prep);
        let e = self.entries.lock().unwrap();
        e.get(cluster)
            .filter(|(k, _)| *k == key)
            .map(|(_, f)| f.clone())
    }

    pub fn put(&self, cluster: &str, prep: &PreparedCluster, fit: &Fit) {
        let key = cluster_key(prep);
        self.entries
            .lock()
            .unwrap()
            .insert(cluster.to_string(), (key, fit.clone()));
        *self.dirty.lock().unwrap() = true;
    }

    /// Persist to disk (no-op for in-memory caches or when nothing changed).
    pub fn save(&self) -> std::io::Result<()> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        if !*self.dirty.lock().unwrap() {
            return Ok(());
        }
        let cf = CacheFile {
            cohort_key: self.cohort_key,
            entries: self.entries.lock().unwrap().clone(),
        };
        let tmp = path.with_extension("tmp");
        std::fs::write(
            &tmp,
            serde_json::to_vec(&cf).map_err(std::io::Error::other)?,
        )?;
        std::fs::rename(&tmp, path)?;
        *self.dirty.lock().unwrap() = false;
        Ok(())
    }
}
