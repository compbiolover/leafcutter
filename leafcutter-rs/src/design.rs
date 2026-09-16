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
        Design { n, p: 1, data: vec![1.0; n] }
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
        Design { n: rows.len(), p: self.p, data }
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

fn hist_from_values(mut v: Vec<u32>) -> Hist {
    v.sort_unstable();
    let mut h: Hist = Vec::new();
    for x in v {
        if x == 0 {
            continue;
        }
        match h.last_mut() {
            Some((val, m)) if *val == x => *m += 1.0,
            _ => h.push((x, 1.0)),
        }
    }
    h
}

impl ClusterData {
    /// `counts` is sample-major (`counts[n * k + j]`).
    pub fn build(counts: &[u32], n: usize, k: usize, design: &Design) -> ClusterData {
        assert_eq!(counts.len(), n * k);
        assert_eq!(design.n, n);
        let p = design.p;
        let mut index: HashMap<Vec<u64>, usize> = HashMap::new();
        let mut xs: Vec<Vec<f64>> = Vec::new();
        let mut per_cell_y: Vec<Vec<Vec<u32>>> = Vec::new();
        let mut per_cell_tot: Vec<Vec<u32>> = Vec::new();
        let mut per_cell_npos: Vec<f64> = Vec::new();
        for i in 0..n {
            let row = design.row(i);
            let key: Vec<u64> = row.iter().map(|v| v.to_bits()).collect();
            let c = match index.get(&key) {
                Some(&c) => c,
                None => {
                    let c = xs.len();
                    index.insert(key, c);
                    xs.push(row.to_vec());
                    per_cell_y.push(vec![Vec::new(); k]);
                    per_cell_tot.push(Vec::new());
                    per_cell_npos.push(0.0);
                    c
                }
            };
            let y = &counts[i * k..(i + 1) * k];
            let tot: u32 = y.iter().sum();
            if tot == 0 {
                continue;
            }
            per_cell_npos[c] += 1.0;
            per_cell_tot[c].push(tot);
            for j in 0..k {
                if y[j] > 0 {
                    per_cell_y[c][j].push(y[j]);
                }
            }
        }
        let cells = xs
            .into_iter()
            .zip(per_cell_y)
            .zip(per_cell_tot)
            .zip(per_cell_npos)
            .map(|(((x, ys), tots), n_pos)| Cell {
                x,
                n_pos,
                intron_hist: ys.into_iter().map(hist_from_values).collect(),
                total_hist: hist_from_values(tots),
            })
            .collect();
        ClusterData { k, p, n, cells }
    }

    /// Number of (lgamma, digamma) pairs evaluated per objective evaluation; a cost proxy.
    pub fn work_units(&self) -> usize {
        self.cells
            .iter()
            .map(|c| 1 + c.total_hist.len() + c.intron_hist.iter().map(|h| 1 + h.len()).sum::<usize>())
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
        assert!(c1.intron_hist[0].is_empty());
        assert_eq!(c1.intron_hist[1], vec![(5, 1.0)]);
    }
}
