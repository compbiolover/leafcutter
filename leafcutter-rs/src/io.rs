//! Reading LeafCutter count and group files and writing the result tables.

use crate::design::Design;
use crate::ds::{Cluster, ClusterResult};
use flate2::read::MultiGzDecoder;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Open a possibly gzip-compressed text file (detected by magic bytes, not the extension).
pub fn open_maybe_gz(path: &Path) -> io::Result<Box<dyn BufRead>> {
    let mut f = File::open(path)?;
    let mut magic = [0u8; 2];
    let n = f.read(&mut magic)?;
    let f = File::open(path)?;
    if n == 2 && magic == [0x1f, 0x8b] {
        Ok(Box::new(BufReader::with_capacity(1 << 20, MultiGzDecoder::new(f))))
    } else {
        Ok(Box::new(BufReader::with_capacity(1 << 20, f)))
    }
}

/// An intron-by-sample count table as produced by the clustering scripts
/// (`*_perind_numers.counts.gz`, or `*_perind.counts.gz` whose `a/b` entries are read as `a`).
pub struct CountsTable {
    pub samples: Vec<String>,
    /// (intron name `chr:start:end:clu_id`, counts over samples)
    pub rows: Vec<(String, Vec<u32>)>,
}

fn parse_count(tok: &str) -> Result<u32, String> {
    let num = tok.split('/').next().unwrap_or(tok);
    num.parse::<u32>().or_else(|_| num.parse::<f64>().map(|v| v.round() as u32)).map_err(|_| format!("bad count '{tok}'"))
}

pub fn read_counts(path: &Path) -> Result<CountsTable, String> {
    let reader = open_maybe_gz(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut lines = reader.lines();
    let header = lines.next().ok_or("empty counts file")?.map_err(|e| e.to_string())?;
    let mut samples: Vec<String> = header.split_whitespace().map(String::from).collect();
    let mut rows = Vec::new();
    let mut header_checked = false;
    for (ln, line) in lines.enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let name = it.next().ok_or_else(|| format!("line {}: empty", ln + 2))?.to_string();
        let vals: Result<Vec<u32>, String> = it.map(parse_count).collect();
        let vals = vals.map_err(|e| format!("line {}: {e}", ln + 2))?;
        if !header_checked {
            // R's read.table: a header one field shorter than the rows means the first column
            // holds row names; a header of equal length means it carried a row-name label.
            if samples.len() == vals.len() + 1 {
                samples.remove(0);
            }
            header_checked = true;
        }
        if vals.len() != samples.len() {
            return Err(format!("line {}: expected {} counts, found {}", ln + 2, samples.len(), vals.len()));
        }
        rows.push((name, vals));
    }
    Ok(CountsTable { samples, rows })
}

/// Cluster id of an intron name `chr:start:end:clu` is `chr:clu` (as `get_intron_meta`).
pub fn cluster_id(intron: &str) -> Result<String, String> {
    let parts: Vec<&str> = intron.split(':').collect();
    if parts.len() < 4 {
        return Err(format!("intron name '{intron}' is not chr:start:end:cluster"));
    }
    Ok(format!("{}:{}", parts[0], parts[parts.len() - 1]))
}

/// Group the rows of a count table into clusters, keeping only the given sample columns
/// (in the given order). Clusters are sorted by name, as R's `table()` does.
pub fn clusters_from_table(table: &CountsTable, sample_cols: &[usize]) -> Result<Vec<Cluster>, String> {
    let mut order: Vec<String> = Vec::new();
    let mut map: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, (name, _)) in table.rows.iter().enumerate() {
        let cid = cluster_id(name)?;
        map.entry(cid.clone()).or_insert_with(|| {
            order.push(cid);
            Vec::new()
        }).push(i);
    }
    order.sort();
    let n = sample_cols.len();
    let mut clusters = Vec::with_capacity(order.len());
    for cid in order {
        let rows = &map[&cid];
        let k = rows.len();
        let mut counts = vec![0u32; n * k];
        let mut introns = Vec::with_capacity(k);
        for (j, &r) in rows.iter().enumerate() {
            let (name, vals) = &table.rows[r];
            introns.push(name.clone());
            for (i, &c) in sample_cols.iter().enumerate() {
                counts[i * k + j] = vals[c];
            }
        }
        clusters.push(Cluster { name: cid, introns, n, counts });
    }
    Ok(clusters)
}

/// Sample metadata from a groups file: `sample group [confounder ...]`, no header.
#[derive(Clone, Debug)]
pub struct Meta {
    pub samples: Vec<String>,
    pub groups: Vec<String>,
    /// Confounder columns as raw strings (column-major).
    pub confounders: Vec<Vec<String>>,
}

pub fn read_groups(path: &Path) -> Result<Meta, String> {
    let reader = open_maybe_gz(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut meta = Meta { samples: Vec::new(), groups: Vec::new(), confounders: Vec::new() };
    for (ln, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.is_empty() {
            continue;
        }
        if toks.len() < 2 {
            return Err(format!("groups file line {}: need at least 2 columns", ln + 1));
        }
        if meta.confounders.is_empty() && meta.samples.is_empty() {
            meta.confounders = vec![Vec::new(); toks.len() - 2];
        }
        if toks.len() - 2 != meta.confounders.len() {
            return Err(format!("groups file line {}: inconsistent number of columns", ln + 1));
        }
        meta.samples.push(toks[0].to_string());
        meta.groups.push(toks[1].to_string());
        for (c, t) in toks[2..].iter().enumerate() {
            meta.confounders[c].push(t.to_string());
        }
    }
    Ok(meta)
}

/// Encoded request design: 0/1 group vector, group names in encoding order, confounder matrix.
#[derive(Clone, Debug)]
pub struct EncodedDesign {
    pub x: Vec<f64>,
    pub group_names: [String; 2],
    pub confounders: Option<Design>,
    pub confounder_names: Vec<String>,
}

fn all_numeric(v: &[String]) -> Option<Vec<f64>> {
    v.iter().map(|s| s.parse::<f64>().ok()).collect()
}

/// Encode groups and confounders exactly as `scripts/leafcutter_ds.R` does: the two group
/// labels in order of first appearance (sorted if numeric) become 0/1; numeric confounders are
/// standardised; categorical confounders become one-of-(L-1) indicator columns with
/// alphabetically sorted levels, the first level being the reference.
pub fn encode_design(groups: &[String], confounders: &[Vec<String>]) -> Result<EncodedDesign, String> {
    let mut names: Vec<String> = Vec::new();
    for g in groups {
        if !names.contains(g) {
            names.push(g.clone());
        }
    }
    if let Some(nums) = all_numeric(&names) {
        let mut idx: Vec<usize> = (0..names.len()).collect();
        idx.sort_by(|&a, &b| nums[a].partial_cmp(&nums[b]).unwrap());
        names = idx.into_iter().map(|i| names[i].clone()).collect();
    }
    if names.len() != 2 {
        return Err(format!("expected exactly 2 groups, found {}: {:?}", names.len(), names));
    }
    let x: Vec<f64> = groups.iter().map(|g| if *g == names[1] { 1.0 } else { 0.0 }).collect();
    let n = groups.len();
    let mut cols: Vec<Vec<f64>> = Vec::new();
    let mut col_names: Vec<String> = Vec::new();
    for (c, col) in confounders.iter().enumerate() {
        if let Some(v) = all_numeric(col) {
            let mean = v.iter().sum::<f64>() / n as f64;
            let sd = if n > 1 { (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt() } else { 0.0 };
            cols.push(v.iter().map(|x| if sd > 0.0 { (x - mean) / sd } else { 0.0 }).collect());
            col_names.push(format!("V{}", c + 3));
        } else {
            let mut levels: Vec<String> = col.clone();
            levels.sort();
            levels.dedup();
            for lvl in levels.iter().skip(1) {
                cols.push(col.iter().map(|v| if v == lvl { 1.0 } else { 0.0 }).collect());
                col_names.push(format!("V{}{}", c + 3, lvl));
            }
        }
    }
    let confounders = if cols.is_empty() {
        None
    } else {
        let refs: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
        Some(Design::from_columns(n, &refs))
    };
    Ok(EncodedDesign { x, group_names: [names[0].clone(), names[1].clone()], confounders, confounder_names: col_names })
}

/// Map requested sample names to column indices of a sample list.
pub fn sample_indices(available: &[String], wanted: &[String]) -> Result<Vec<usize>, String> {
    let idx: HashMap<&str, usize> = available.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    wanted
        .iter()
        .map(|s| idx.get(s.as_str()).copied().ok_or_else(|| format!("sample '{s}' not found in counts")))
        .collect()
}

fn fmt(v: Option<f64>) -> String {
    match v {
        None => "NA".into(),
        Some(x) if x.is_nan() => "NA".into(),
        Some(x) => format!("{x}"),
    }
}

/// `<prefix>_cluster_significance.txt`: cluster, status, loglr, df, p, p.adjust.
pub fn write_cluster_table(path: &Path, results: &[ClusterResult]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "cluster\tstatus\tloglr\tdf\tp\tp.adjust")?;
    for r in results {
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}",
            r.cluster,
            r.status,
            fmt(r.loglr),
            r.df.map(|d| d.to_string()).unwrap_or_else(|| "NA".into()),
            fmt(r.p),
            fmt(r.p_adjust)
        )?;
    }
    w.flush()
}

/// `<prefix>_effect_sizes.txt`: intron, logef, <group0>, <group1>, deltapsi.
pub fn write_effect_sizes(path: &Path, results: &[ClusterResult], group_names: &[String; 2]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "intron\tlogef\t{}\t{}\tdeltapsi", group_names[0], group_names[1])?;
    for r in results {
        for i in &r.introns {
            writeln!(w, "{}\t{}\t{}\t{}\t{}", i.intron, i.logef, i.baseline, i.perturbed, i.deltapsi)?;
        }
    }
    w.flush()
}

/// Write clusters back out in the `perind_numers.counts` format (gzip if the path ends in .gz).
pub fn write_counts(path: &Path, samples: &[String], clusters: &[Cluster]) -> io::Result<()> {
    let file = File::create(path)?;
    let mut w: Box<dyn Write> = if path.extension().map(|e| e == "gz").unwrap_or(false) {
        Box::new(BufWriter::new(flate2::write::GzEncoder::new(file, flate2::Compression::fast())))
    } else {
        Box::new(BufWriter::new(file))
    };
    writeln!(w, "{}", samples.join(" "))?;
    for c in clusters {
        let k = c.k();
        for (j, intron) in c.introns.iter().enumerate() {
            write!(w, "{intron}")?;
            for i in 0..c.n {
                write!(w, " {}", c.counts[i * k + j])?;
            }
            writeln!(w)?;
        }
    }
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_groups_and_confounders() {
        let groups: Vec<String> = ["b", "a", "b", "a"].iter().map(|s| s.to_string()).collect();
        let conf = vec![
            vec!["1", "2", "3", "4"].into_iter().map(String::from).collect(),
            vec!["x", "y", "z", "x"].into_iter().map(String::from).collect(),
        ];
        let e = encode_design(&groups, &conf).unwrap();
        assert_eq!(e.group_names, ["b".to_string(), "a".to_string()]);
        assert_eq!(e.x, vec![0.0, 1.0, 0.0, 1.0]);
        let d = e.confounders.unwrap();
        assert_eq!(d.p, 3); // scaled numeric + 2 indicator columns (levels y, z)
        assert!((d.column_sd(0) - 1.0).abs() < 1e-12);
        assert_eq!(d.column(1), vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(d.column(2), vec![0.0, 0.0, 1.0, 0.0]);
        // numeric groups are sorted
        let groups: Vec<String> = ["1", "0", "1"].iter().map(|s| s.to_string()).collect();
        let e = encode_design(&groups, &[]).unwrap();
        assert_eq!(e.x, vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn cluster_ids() {
        assert_eq!(cluster_id("chr1:100:200:clu_5_NA").unwrap(), "chr1:clu_5_NA");
        assert!(cluster_id("bad").is_err());
    }
}
