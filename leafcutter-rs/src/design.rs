//! Design matrices and the collapsed ("sufficient statistic") representation of a cluster.

use std::collections::HashMap;

/// Dense sample-major design matrix: `data[n * p + j]` is covariate `j` of sample `n`.
#[derive(Clone, Debug, PartialEq)]
pub struct Design {
    pub n: usize,
    pub p: usize,
    pub data: Vec<f64>,
}

impl Design {
    pub fn new(n: usize, p: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), n * p);
        Design { n, p, data }
    }

    /// Intercept-only design.
    pub fn intercept(n: usize) -> Self {
        Design {
            n,
            p: 1,
            data: vec![1.0; n],
        }
    }

    /// Build from column vectors (each of length `n`).
    pub fn from_columns(n: usize, cols: &[&[f64]]) -> Self {
        let p = cols.len();
        let mut data = vec![0.0; n * p];
        for (j, c) in cols.iter().enumerate() {
            assert_eq!(c.len(), n);
            for i in 0..n {
                data[i * p + j] = c[i];
            }
        }
        Design { n, p, data }
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.p..(i + 1) * self.p]
    }

    pub fn column(&self, j: usize) -> Vec<f64> {
        (0..self.n).map(|i| self.data[i * self.p + j]).collect()
    }

    /// Keep only the given columns (in the given order).
    pub fn select_columns(&self, cols: &[usize]) -> Design {
        let p = cols.len();
        let mut data = Vec::with_capacity(self.n * p);
        for i in 0..self.n {
            let r = self.row(i);
            for &c in cols {
                data.push(r[c]);
            }
        }
        Design { n: self.n, p, data }
    }

    /// Keep only the given rows (in the given order).
    pub fn select_rows(&self, rows: &[usize]) -> Design {
        let mut data = Vec::with_capacity(rows.len() * self.p);
        for &r in rows {
            data.extend_from_slice(self.row(r));
        }
        Design {
            n: rows.len(),
            p: self.p,
            data,
        }
    }

    /// Standard deviation of a column (denominator n-1, as R's `sd`).
    pub fn column_sd(&self, j: usize) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        let col = self.column(j);
        let mean = col.iter().sum::<f64>() / self.n as f64;
        (col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (self.n as f64 - 1.0)).sqrt()
    }
}

/// A run-length encoded histogram of non-negative integer values: `(value, multiplicity)`.
pub type Hist = Vec<(u32, f64)>;

/// All samples sharing one design row. Only samples with a positive row total are counted:
/// a sample with zero reads in the cluster contributes nothing to the likelihood.
#[derive(Clone, Debug)]
pub struct Cell {
    /// The covariate row shared by the samples in this cell.
    pub x: Vec<f64>,
    /// Number of samples (with positive total) in the cell.
    pub n_pos: f64,
    /// Number of samples in the cell with zero total (they contribute nothing to the
    /// likelihood but matter for the method-of-moments initialisation).
    pub n_zero: f64,
    /// Per intron, histogram of the non-zero counts observed in this cell.
    pub intron_hist: Vec<Hist>,
    /// Histogram of the positive per-sample totals in this cell.
    pub total_hist: Hist,
}

/// A cluster's counts collapsed onto design cells. The Dirichlet-multinomial log likelihood
/// depends on a sample only through its design row and its integer counts, so samples with
/// identical design rows can be pooled and their counts histogrammed. For categorical designs
/// the cost of one likelihood evaluation then stops growing with the number of samples.
/// With continuous covariates every sample is its own cell and this degrades gracefully to the
/// dense computation.
#[derive(Clone, Debug)]
pub struct ClusterData {
    pub k: usize,
    pub p: usize,
    pub n: usize,
    pub cells: Vec<Cell>,
}

/// Convert a bucket array (`buckets[v]` = multiplicity of value `v`) into a sorted
/// run-length histogram of the non-zero values, clearing the buckets.
fn hist_from_buckets(buckets: &mut [f64]) -> Hist {
    let mut h = Hist::new();
    for (v, m) in buckets.iter_mut().enumerate().skip(1) {
        if *m > 0.0 {
            h.push((v as u32, *m));
            *m = 0.0;
        }
    }
    h
}

/// Merge two sorted run-length histograms.
fn merge_hist(a: &Hist, b: &Hist) -> Hist {
    let mut out = Hist::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        if j == b.len() || (i < a.len() && a[i].0 < b[j].0) {
            out.push(a[i]);
            i += 1;
        } else if i == a.len() || b[j].0 < a[i].0 {
            out.push(b[j]);
            j += 1;
        } else {
            out.push((a[i].0, a[i].1 + b[j].1));
            i += 1;
            j += 1;
        }
    }
    out
}

impl ClusterData {
    /// `counts` is sample-major (`counts[n * k + j]`).
    pub fn build(counts: &[u32], n: usize, k: usize, design: &Design) -> ClusterData {
        assert_eq!(counts.len(), n * k);
        assert_eq!(design.n, n);
        let p = design.p;
        // Assign every sample to a cell. Designs have few distinct rows (categorical
        // covariates) so a linear scan over the cells found so far beats hashing; with
        // continuous covariates every sample is its own cell and we switch to a hash map.
        let mut xs: Vec<Vec<f64>> = Vec::new();
        let mut cell_of: Vec<u32> = Vec::with_capacity(n);
        let mut index: Option<HashMap<Vec<u64>, usize>> = None;
        for i in 0..n {
            let row = design.row(i);
            let c = if let Some(map) = index.as_mut() {
                let key: Vec<u64> = row.iter().map(|v| v.to_bits()).collect();
                *map.entry(key).or_insert_with(|| {
                    xs.push(row.to_vec());
                    xs.len() - 1
                })
            } else {
                match xs.iter().position(|x| x.as_slice() == row) {
                    Some(c) => c,
                    None => {
                        xs.push(row.to_vec());
                        if xs.len() > 64 {
                            let mut map = HashMap::with_capacity(n);
                            for (c, x) in xs.iter().enumerate() {
                                map.insert(x.iter().map(|v| v.to_bits()).collect::<Vec<u64>>(), c);
                            }
                            index = Some(map);
                        }
                        xs.len() - 1
                    }
                }
            };
            cell_of.push(c as u32);
        }
        let n_cells = xs.len();
        // Largest value to bucket: counts and totals.
        let mut max_tot = 0usize;
        for i in 0..n {
            let tot: u32 = counts[i * k..(i + 1) * k].iter().sum();
            max_tot = max_tot.max(tot as usize);
        }
        let mut n_pos = vec![0.0; n_cells];
        let mut n_zero = vec![0.0; n_cells];
        let (intron_hist, total_hist): (Vec<Vec<Hist>>, Vec<Hist>) =
            if n_cells * (k + 1) * (max_tot + 1) <= 4 << 20 {
                // counting pass: one bucket array per (cell, intron) and per cell for totals
                let width = max_tot + 1;
                let mut buckets = vec![0.0f64; n_cells * (k + 1) * width];
                for i in 0..n {
                    let y = &counts[i * k..(i + 1) * k];
                    let tot: u32 = y.iter().sum();
                    let c = cell_of[i] as usize;
                    if tot == 0 {
                        n_zero[c] += 1.0;
                        continue;
                    }
                    n_pos[c] += 1.0;
                    let base = c * (k + 1) * width;
                    for j in 0..k {
                        buckets[base + j * width + y[j] as usize] += 1.0;
                    }
                    buckets[base + k * width + tot as usize] += 1.0;
                }
                let mut ih = Vec::with_capacity(n_cells);
                let mut th = Vec::with_capacity(n_cells);
                for c in 0..n_cells {
                    let base = c * (k + 1) * width;
                    ih.push(
                        (0..k)
                            .map(|j| {
                                hist_from_buckets(
                                    &mut buckets[base + j * width..base + (j + 1) * width],
                                )
                            })
                            .collect(),
                    );
                    th.push(hist_from_buckets(
                        &mut buckets[base + k * width..base + (k + 1) * width],
                    ));
                }
                (ih, th)
            } else {
                // very large counts: sort-based histograms
                let mut per_cell_y: Vec<Vec<Vec<u32>>> = vec![vec![Vec::new(); k]; n_cells];
                let mut per_cell_tot: Vec<Vec<u32>> = vec![Vec::new(); n_cells];
                for i in 0..n {
                    let y = &counts[i * k..(i + 1) * k];
                    let tot: u32 = y.iter().sum();
                    let c = cell_of[i] as usize;
                    if tot == 0 {
                        n_zero[c] += 1.0;
                        continue;
                    }
                    n_pos[c] += 1.0;
                    per_cell_tot[c].push(tot);
                    for j in 0..k {
                        if y[j] > 0 {
                            per_cell_y[c][j].push(y[j]);
                        }
                    }
                }
                let sorted_hist = |mut v: Vec<u32>| -> Hist {
                    v.sort_unstable();
                    let mut h = Hist::new();
                    for x in v {
                        match h.last_mut() {
                            Some((val, m)) if *val == x => *m += 1.0,
                            _ => h.push((x, 1.0)),
                        }
                    }
                    h
                };
                (
                    per_cell_y
                        .into_iter()
                        .map(|ys| ys.into_iter().map(sorted_hist).collect())
                        .collect(),
                    per_cell_tot.into_iter().map(sorted_hist).collect(),
                )
            };
        let cells = xs
            .into_iter()
            .zip(intron_hist)
            .zip(total_hist)
            .zip(n_pos.into_iter().zip(n_zero))
            .map(|(((x, intron_hist), total_hist), (n_pos, n_zero))| Cell {
                x,
                n_pos,
                n_zero,
                intron_hist,
                total_hist,
            })
            .collect();
        ClusterData { k, p, n, cells }
    }

    /// The same data under a design made of a subset of the covariate columns: cells whose
    /// rows agree on those columns are merged. Much cheaper than rebuilding from the counts.
    pub fn select_columns(&self, cols: &[usize]) -> ClusterData {
        let mut cells: Vec<Cell> = Vec::new();
        for c in &self.cells {
            let x: Vec<f64> = cols.iter().map(|&q| c.x[q]).collect();
            match cells.iter_mut().find(|m| m.x == x) {
                Some(m) => {
                    m.n_pos += c.n_pos;
                    m.n_zero += c.n_zero;
                    for j in 0..self.k {
                        m.intron_hist[j] = merge_hist(&m.intron_hist[j], &c.intron_hist[j]);
                    }
                    m.total_hist = merge_hist(&m.total_hist, &c.total_hist);
                }
                None => cells.push(Cell {
                    x,
                    n_pos: c.n_pos,
                    n_zero: c.n_zero,
                    intron_hist: c.intron_hist.clone(),
                    total_hist: c.total_hist.clone(),
                }),
            }
        }
        ClusterData {
            k: self.k,
            p: cols.len(),
            n: self.n,
            cells,
        }
    }

    /// Number of (lgamma, digamma) pairs evaluated per objective evaluation; a cost proxy.
    pub fn work_units(&self) -> usize {
        self.cells
            .iter()
            .map(|c| {
                1 + c.total_hist.len() + c.intron_hist.iter().map(|h| 1 + h.len()).sum::<usize>()
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapse_pools_identical_rows() {
        // 4 samples, 2 introns, group design [1,g]
        let counts = vec![3, 1, 3, 1, 0, 5, 0, 0];
        let d = Design::from_columns(4, &[&[1.0; 4], &[0.0, 0.0, 1.0, 1.0]]);
        let cd = ClusterData::build(&counts, 4, 2, &d);
        assert_eq!(cd.cells.len(), 2);
        let c0 = cd.cells.iter().find(|c| c.x[1] == 0.0).unwrap();
        assert_eq!(c0.n_pos, 2.0);
        assert_eq!(c0.intron_hist[0], vec![(3, 2.0)]);
        assert_eq!(c0.intron_hist[1], vec![(1, 2.0)]);
        assert_eq!(c0.total_hist, vec![(4, 2.0)]);
        let c1 = cd.cells.iter().find(|c| c.x[1] == 1.0).unwrap();
        assert_eq!(c1.n_pos, 1.0); // the all-zero sample is dropped
        assert_eq!(c1.n_zero, 1.0);
        assert!(c1.intron_hist[0].is_empty());
        assert_eq!(c1.intron_hist[1], vec![(5, 1.0)]);
        // merging to the intercept-only design equals building it directly
        let merged = cd.select_columns(&[0]);
        let direct = ClusterData::build(&counts, 4, 2, &d.select_columns(&[0]));
        assert_eq!(merged.cells.len(), 1);
        assert_eq!(merged.cells[0].n_pos, direct.cells[0].n_pos);
        assert_eq!(merged.cells[0].n_zero, direct.cells[0].n_zero);
        assert_eq!(merged.cells[0].intron_hist, direct.cells[0].intron_hist);
        assert_eq!(merged.cells[0].total_hist, direct.cells[0].total_hist);
    }
}
