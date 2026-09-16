//! A memory-mapped, cluster-major count store for large cohorts.
//!
//! Layout: `b"LCCS"` magic, `u32` version, `u64` header length, a JSON header
//! ([`StoreHeader`]: sample names and per-cluster metadata) and then one contiguous block per
//! cluster holding the counts sample-major (`N_store x K`) as little-endian `u16` or `u32`
//! (chosen per cluster from its maximum count).
//!
//! Sample-major blocks mean that a request for a subset of the cohort touches at most one
//! contiguous `K x width` run per requested sample; a request for the whole cohort streams the
//! block sequentially. Nothing is ever materialised as a dense floating point matrix: each
//! worker thread gathers one cluster at a time.

use crate::ds::Cluster;
use crate::io::CountsTable;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"LCCS";
const VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterMeta {
    pub name: String,
    pub introns: Vec<String>,
    /// Byte offset of the block from the start of the data section.
    pub offset: u64,
    /// 2 or 4 bytes per count.
    pub width: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoreHeader {
    pub samples: Vec<String>,
    pub clusters: Vec<ClusterMeta>,
}

/// Write a store from an in-memory table (clusters grouped by name, sorted).
pub fn build_store(table: &CountsTable, out: &Path) -> Result<(), String> {
    let all: Vec<usize> = (0..table.samples.len()).collect();
    let clusters = crate::io::clusters_from_table(table, &all)?;
    write_store(&table.samples, &clusters, out).map_err(|e| e.to_string())
}

pub fn write_store(samples: &[String], clusters: &[Cluster], out: &Path) -> std::io::Result<()> {
    let mut metas = Vec::with_capacity(clusters.len());
    let mut offset = 0u64;
    for c in clusters {
        let max = c.counts.iter().copied().max().unwrap_or(0);
        let width: u8 = if max <= u16::MAX as u32 { 2 } else { 4 };
        metas.push(ClusterMeta {
            name: c.name.clone(),
            introns: c.introns.clone(),
            offset,
            width,
        });
        offset += (c.counts.len() * width as usize) as u64;
    }
    let header = serde_json::to_vec(&StoreHeader {
        samples: samples.to_vec(),
        clusters: metas.clone(),
    })?;
    let mut w = BufWriter::with_capacity(1 << 20, File::create(out)?);
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(header.len() as u64).to_le_bytes())?;
    w.write_all(&header)?;
    let mut buf: Vec<u8> = Vec::new();
    for (c, m) in clusters.iter().zip(&metas) {
        buf.clear();
        match m.width {
            2 => {
                for &v in &c.counts {
                    buf.extend_from_slice(&(v as u16).to_le_bytes());
                }
            }
            _ => {
                for &v in &c.counts {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        w.write_all(&buf)?;
    }
    w.flush()
}

/// An opened store.
pub struct Store {
    mmap: Mmap,
    data_start: usize,
    pub header: StoreHeader,
}

impl Store {
    pub fn open(path: &Path) -> Result<Store, String> {
        let file =
            File::open(path).map_err(|e| format!("cannot open store {}: {e}", path.display()))?;
        // SAFETY: the store is treated as read-only; concurrent modification of the file is
        // outside this program's control, as for any mmap.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;
        if mmap.len() < 16 || &mmap[0..4] != MAGIC {
            return Err("not a LeafCutter count store".into());
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(format!("unsupported store version {version}"));
        }
        let hlen = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let header: StoreHeader =
            serde_json::from_slice(&mmap[16..16 + hlen]).map_err(|e| e.to_string())?;
        Ok(Store {
            mmap,
            data_start: 16 + hlen,
            header,
        })
    }

    pub fn n_samples(&self) -> usize {
        self.header.samples.len()
    }

    pub fn n_clusters(&self) -> usize {
        self.header.clusters.len()
    }

    /// Gather cluster `i` for the given store sample indices (in request order).
    pub fn cluster(&self, i: usize, sample_idx: &[usize]) -> Cluster {
        let m = &self.header.clusters[i];
        let k = m.introns.len();
        let n_store = self.n_samples();
        let w = m.width as usize;
        let base = self.data_start + m.offset as usize;
        let block = &self.mmap[base..base + n_store * k * w];
        let mut counts = Vec::with_capacity(sample_idx.len() * k);
        for &s in sample_idx {
            let row = &block[s * k * w..(s + 1) * k * w];
            match w {
                2 => counts.extend(
                    row.chunks_exact(2)
                        .map(|b| u16::from_le_bytes([b[0], b[1]]) as u32),
                ),
                _ => counts.extend(
                    row.chunks_exact(4)
                        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
                ),
            }
        }
        let annotations: Vec<String> = {
            let mut a: Vec<String> = m
                .introns
                .iter()
                .filter_map(|s| s.split(':').nth(4).map(String::from))
                .collect();
            a.sort();
            a.dedup();
            a
        };
        Cluster {
            name: m.name.clone(),
            introns: m.introns.clone(),
            annotations,
            n: sample_idx.len(),
            counts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let samples: Vec<String> = (0..5).map(|i| format!("s{i}")).collect();
        let clusters = vec![
            Cluster {
                annotations: vec![],
                name: "c1".into(),
                introns: vec!["a".into(), "b".into()],
                n: 5,
                counts: (0..10).collect(),
            },
            Cluster {
                annotations: vec![],
                name: "c2".into(),
                introns: vec!["c".into(), "d".into(), "e".into()],
                n: 5,
                counts: (0..15).map(|v| v * 10000).collect(),
            },
        ];
        let dir = std::env::temp_dir().join(format!("lcstore_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.lcs");
        write_store(&samples, &clusters, &path).unwrap();
        let st = Store::open(&path).unwrap();
        assert_eq!(st.n_samples(), 5);
        assert_eq!(st.header.clusters[0].width, 2);
        assert_eq!(st.header.clusters[1].width, 4);
        let c = st.cluster(1, &[4, 0]);
        assert_eq!(c.n, 2);
        assert_eq!(c.counts, vec![120000, 130000, 140000, 0, 10000, 20000]);
        let c = st.cluster(0, &[1]);
        assert_eq!(c.counts, vec![2, 3]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
