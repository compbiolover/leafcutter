//! Reading LeafCutter count and group files and writing the result tables in the layout of
//! leafcutter-ds (`leafcutter/differential_splicing/leafcutter_ds.py`).

use crate::design::Design;
use crate::ds::{Cluster, ClusterResult};
use flate2::read::MultiGzDecoder;
use std::collections::{BTreeSet, HashMap};
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
        Ok(Box::new(BufReader::with_capacity(
            1 << 20,
            MultiGzDecoder::new(f),
        )))
    } else {
        Ok(Box::new(BufReader::with_capacity(1 << 20, f)))
    }
}

/// An intron-by-sample count table as produced by the clustering scripts
/// (`*_perind_numers.counts.gz`, `*.junction_counts.gz`, or `*_perind.counts.gz` whose `a/b`
/// entries are read as `a`).
pub struct CountsTable {
    pub samples: Vec<String>,
    /// (intron name `chr:start:end:clu[:annotation]`, counts over samples)
    pub rows: Vec<(String, Vec<u32>)>,
}

fn parse_count(tok: &str) -> Result<u32, String> {
    let num = tok.split('/').next().unwrap_or(tok);
    num.parse::<u32>()
        .or_else(|_| num.parse::<f64>().map(|v| v.round() as u32))
        .map_err(|_| format!("bad count '{tok}'"))
}

pub fn read_counts(path: &Path) -> Result<CountsTable, String> {
    let reader = open_maybe_gz(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut lines = reader.lines();
    let header = lines
        .next()
        .ok_or("empty counts file")?
        .map_err(|e| e.to_string())?;
    let mut samples: Vec<String> = header.split_whitespace().map(String::from).collect();
    let mut rows = Vec::new();
    let mut header_checked = false;
    for (ln, line) in lines.enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let name = it
            .next()
            .ok_or_else(|| format!("line {}: empty", ln + 2))?
            .to_string();
        let vals: Result<Vec<u32>, String> = it.map(parse_count).collect();
        let vals = vals.map_err(|e| format!("line {}: {e}", ln + 2))?;
        if !header_checked {
            // a header one field longer than the data carries a label for the row-name column
            if samples.len() == vals.len() + 1 {
                samples.remove(0);
            }
            header_checked = true;
        }
        if vals.len() != samples.len() {
            return Err(format!(
                "line {}: expected {} counts, found {}",
                ln + 2,
                samples.len(),
                vals.len()
            ));
        }
        rows.push((name, vals));
    }
    Ok(CountsTable { samples, rows })
}

/// The fields of an intron name `chr:start:end:clu[:annotation]`.
pub struct IntronName<'a> {
    pub chr: &'a str,
    pub start: u64,
    pub end: u64,
    pub clu: &'a str,
    pub annotation: Option<&'a str>,
}

pub fn parse_intron(name: &str) -> Result<IntronName<'_>, String> {
    let parts: Vec<&str> = name.split(':').collect();
    if parts.len() < 4 {
        return Err(format!(
            "intron name '{name}' is not chr:start:end:cluster[:annotation]"
        ));
    }
    let start = parts[1]
        .parse::<u64>()
        .map_err(|_| format!("intron name '{name}': bad start"))?;
    let end = parts[2]
        .parse::<u64>()
        .map_err(|_| format!("intron name '{name}': bad end"))?;
    Ok(IntronName {
        chr: parts[0],
        start,
        end,
        clu: parts[3],
        annotation: parts.get(4).copied(),
    })
}

/// Cluster id of an intron: `chr:clu`.
pub fn cluster_id(intron: &str) -> Result<String, String> {
    let p = parse_intron(intron)?;
    Ok(format!("{}:{}", p.chr, p.clu))
}

/// Group the rows of a count table into clusters, keeping only the given sample columns (in
/// the given order). Clusters keep their order of first appearance, as leafcutter-ds does.
pub fn clusters_from_table(
    table: &CountsTable,
    sample_cols: &[usize],
) -> Result<Vec<Cluster>, String> {
    let mut order: Vec<String> = Vec::new();
    let mut map: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, (name, _)) in table.rows.iter().enumerate() {
        let cid = cluster_id(name)?;
        map.entry(cid.clone())
            .or_insert_with(|| {
                order.push(cid);
                Vec::new()
            })
            .push(i);
    }
    let n = sample_cols.len();
    let mut clusters = Vec::with_capacity(order.len());
    for cid in order {
        let rows = &map[&cid];
        let k = rows.len();
        let mut counts = vec![0u32; n * k];
        let mut introns = Vec::with_capacity(k);
        let mut annotations: BTreeSet<String> = BTreeSet::new();
        for (j, &r) in rows.iter().enumerate() {
            let (name, vals) = &table.rows[r];
            introns.push(name.clone());
            if let Some(a) = parse_intron(name)?.annotation {
                annotations.insert(a.to_string());
            }
            for (i, &c) in sample_cols.iter().enumerate() {
                counts[i * k + j] = vals[c];
            }
        }
        clusters.push(Cluster {
            name: cid,
            introns,
            annotations: annotations.into_iter().collect(),
            n,
            counts,
        });
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
    let mut meta = Meta {
        samples: Vec::new(),
        groups: Vec::new(),
        confounders: Vec::new(),
    };
    for (ln, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.is_empty() {
            continue;
        }
        if toks.len() < 2 {
            return Err(format!(
                "groups file line {}: need at least 2 columns",
                ln + 1
            ));
        }
        if meta.confounders.is_empty() && meta.samples.is_empty() {
            meta.confounders = vec![Vec::new(); toks.len() - 2];
        }
        if toks.len() - 2 != meta.confounders.len() {
            return Err(format!(
                "groups file line {}: inconsistent number of columns",
                ln + 1
            ));
        }
        meta.samples.push(toks[0].to_string());
        meta.groups.push(toks[1].to_string());
        for (c, t) in toks[2..].iter().enumerate() {
            meta.confounders[c].push(t.to_string());
        }
    }
    if meta.samples.is_empty() {
        return Err("groups file is empty".into());
    }
    Ok(meta)
}

/// The phenotype column of the groups file after encoding.
#[derive(Clone, Debug)]
pub enum Phenotype {
    /// Levels with the baseline first; `codes[i]` indexes `levels`.
    Categorical {
        levels: Vec<String>,
        codes: Vec<usize>,
    },
    /// Standardised values (`(x - mean) / sd`, population sd) and the sd used.
    Continuous { values: Vec<f64>, scale_factor: f64 },
}

impl Phenotype {
    pub fn len(&self) -> usize {
        match self {
            Phenotype::Categorical { codes, .. } => codes.len(),
            Phenotype::Continuous { values, .. } => values.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Names of the non-baseline design columns (the `<group>` in `logef_<group>`).
    pub fn group_names(&self) -> Vec<String> {
        match self {
            Phenotype::Categorical { levels, .. } => levels[1..].to_vec(),
            Phenotype::Continuous { .. } => vec!["x".to_string()],
        }
    }
    pub fn is_categorical(&self) -> bool {
        matches!(self, Phenotype::Categorical { .. })
    }
}

/// Encoded request design.
#[derive(Clone, Debug)]
pub struct EncodedDesign {
    pub phenotype: Phenotype,
    pub confounders: Option<Design>,
    pub confounder_names: Vec<String>,
    /// Samples kept (those with complete confounders), by position in the metadata.
    pub sample_rows: Vec<usize>,
    /// Samples dropped for missing confounder values.
    pub dropped_samples: Vec<String>,
    /// Set when the requested baseline is not one of the group labels (the first level in
    /// sorted order is used instead).
    pub baseline_missing: bool,
}

fn is_missing(s: &str) -> bool {
    matches!(
        s,
        "" | "NA" | "N/A" | "n/a" | "NaN" | "nan" | "NULL" | "null" | "None"
    )
}

fn all_numeric(v: &[String]) -> Option<Vec<f64>> {
    v.iter().map(|s| s.parse::<f64>().ok()).collect()
}

fn standardise(v: &[f64]) -> (Vec<f64>, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
    (
        v.iter()
            .map(|x| if sd > 0.0 { (x - mean) / sd } else { 0.0 })
            .collect(),
        sd,
    )
}

/// Encode the phenotype and confounders exactly as `leafcutter_ds.py` does: a numeric
/// phenotype is standardised (continuous); otherwise its labels become ordered levels with
/// `baseline` first; numeric confounders are standardised (population sd, as
/// `StandardScaler`), categorical confounders one-hot encoded with the first sorted level
/// dropped; samples with missing confounder values are removed.
pub fn encode_design(meta: &Meta, baseline: &str) -> Result<EncodedDesign, String> {
    // samples with complete confounders
    let n_all = meta.samples.len();
    let sample_rows: Vec<usize> = (0..n_all)
        .filter(|&i| meta.confounders.iter().all(|c| !is_missing(&c[i])))
        .collect();
    let dropped_samples: Vec<String> = (0..n_all)
        .filter(|i| !sample_rows.contains(i))
        .map(|i| meta.samples[i].clone())
        .collect();
    if sample_rows.is_empty() {
        return Err("no sample has complete confounder values".into());
    }
    let groups: Vec<String> = sample_rows
        .iter()
        .map(|&i| meta.groups[i].clone())
        .collect();
    let n = groups.len();

    let mut baseline_missing = false;
    let phenotype = if let Some(v) = all_numeric(&groups) {
        let (values, scale_factor) = standardise(&v);
        Phenotype::Continuous {
            values,
            scale_factor,
        }
    } else {
        let mut levels: Vec<String> = groups
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if let Some(pos) = levels.iter().position(|l| l == baseline) {
            let b = levels.remove(pos);
            levels.insert(0, b);
        } else {
            baseline_missing = true;
        }
        let codes = groups
            .iter()
            .map(|g| levels.iter().position(|l| l == g).unwrap())
            .collect();
        Phenotype::Categorical { levels, codes }
    };

    let mut cols: Vec<Vec<f64>> = Vec::new();
    let mut col_names: Vec<String> = Vec::new();
    for (c, col) in meta.confounders.iter().enumerate() {
        let sub: Vec<String> = sample_rows.iter().map(|&i| col[i].clone()).collect();
        if let Some(v) = all_numeric(&sub) {
            cols.push(standardise(&v).0);
            col_names.push(format!("conf{}", c + 1));
        } else {
            let levels: Vec<String> = sub
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            for lvl in levels.iter().skip(1) {
                cols.push(
                    sub.iter()
                        .map(|v| if v == lvl { 1.0 } else { 0.0 })
                        .collect(),
                );
                col_names.push(format!("conf{}={}", c + 1, lvl));
            }
        }
    }
    let confounders = if cols.is_empty() {
        None
    } else {
        let refs: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
        Some(Design::from_columns(n, &refs))
    };
    Ok(EncodedDesign {
        phenotype,
        confounders,
        confounder_names: col_names,
        sample_rows,
        dropped_samples,
        baseline_missing,
    })
}

/// Map requested sample names to column indices of a sample list.
pub fn sample_indices(available: &[String], wanted: &[String]) -> Result<Vec<usize>, String> {
    let idx: HashMap<&str, usize> = available
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    wanted
        .iter()
        .map(|s| {
            idx.get(s.as_str())
                .copied()
                .ok_or_else(|| format!("sample '{s}' not found in counts"))
        })
        .collect()
}

fn fmt(v: Option<f64>) -> String {
    match v {
        None => "NA".into(),
        Some(x) if x.is_nan() => "NA".into(),
        Some(x) => format!("{x}"),
    }
}

/// `<prefix>_cluster_significance.txt`: cluster, status, loglr, df, p, p.adjust, genes
/// [, annotations].
pub fn write_cluster_table(
    path: &Path,
    results: &[ClusterResult],
    with_annotations: bool,
) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write!(w, "cluster\tstatus\tloglr\tdf\tp\tp.adjust\tgenes")?;
    if with_annotations {
        write!(w, "\tannotations")?;
    }
    writeln!(w)?;
    for r in results {
        write!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            r.cluster,
            r.status,
            fmt(r.loglr),
            r.df.map(|d| d.to_string()).unwrap_or_else(|| "NA".into()),
            fmt(r.p),
            fmt(r.p_adjust),
            r.genes.as_deref().unwrap_or("NA")
        )?;
        if with_annotations {
            write!(
                w,
                "\t{}",
                if r.annotations.is_empty() {
                    "NA".to_string()
                } else {
                    r.annotations.join(",")
                }
            )?;
        }
        writeln!(w)?;
    }
    w.flush()
}

/// `<prefix>_effect_sizes.txt`: intron, logef_<g>..., psi_<baseline>, psi_<g>..., deltapsi_<g>...
pub fn write_effect_sizes(
    path: &Path,
    results: &[ClusterResult],
    group_names: &[String],
    baseline_label: &str,
) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write!(w, "intron")?;
    for g in group_names {
        write!(w, "\tlogef_{g}")?;
    }
    write!(w, "\tpsi_{baseline_label}")?;
    for g in group_names {
        write!(w, "\tpsi_{g}")?;
    }
    for g in group_names {
        write!(w, "\tdeltapsi_{g}")?;
    }
    writeln!(w)?;
    for r in results {
        for i in &r.introns {
            write!(w, "{}", i.intron)?;
            for g in group_names {
                write!(w, "\t{}", fmt(i.effects.get(g).map(|e| e.logef)))?;
            }
            write!(w, "\t{}", i.psi_baseline)?;
            for g in group_names {
                write!(w, "\t{}", fmt(i.effects.get(g).map(|e| e.psi)))?;
            }
            for g in group_names {
                write!(w, "\t{}", fmt(i.effects.get(g).map(|e| e.deltapsi)))?;
            }
            writeln!(w)?;
        }
    }
    w.flush()
}

/// Write clusters back out in the `perind_numers.counts` format (gzip if the path ends in .gz).
pub fn write_counts(path: &Path, samples: &[String], clusters: &[Cluster]) -> io::Result<()> {
    let file = File::create(path)?;
    let mut w: Box<dyn Write> = if path.extension().map(|e| e == "gz").unwrap_or(false) {
        Box::new(BufWriter::new(flate2::write::GzEncoder::new(
            file,
            flate2::Compression::fast(),
        )))
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

    fn meta(groups: &[&str], conf: &[&[&str]]) -> Meta {
        Meta {
            samples: (0..groups.len()).map(|i| format!("s{i}")).collect(),
            groups: groups.iter().map(|s| s.to_string()).collect(),
            confounders: conf
                .iter()
                .map(|c| c.iter().map(|s| s.to_string()).collect())
                .collect(),
        }
    }

    #[test]
    fn encode_categorical_with_baseline_and_confounders() {
        let m = meta(
            &["b", "a", "c", "a"],
            &[&["1", "2", "3", "4"], &["x", "y", "z", "x"]],
        );
        let e = encode_design(&m, "b").unwrap();
        match &e.phenotype {
            Phenotype::Categorical { levels, codes } => {
                assert_eq!(levels, &["b", "a", "c"]);
                assert_eq!(codes, &[0, 1, 2, 1]);
            }
            _ => panic!(),
        }
        assert_eq!(
            e.phenotype.group_names(),
            vec!["a".to_string(), "c".to_string()]
        );
        let d = e.confounders.unwrap();
        assert_eq!(d.p, 3);
        // population sd standardisation
        let c0 = d.column(0);
        let mean: f64 = c0.iter().sum::<f64>() / 4.0;
        let var: f64 = c0.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-12 && (var - 1.0).abs() < 1e-12);
        assert_eq!(d.column(1), vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(d.column(2), vec![0.0, 0.0, 1.0, 0.0]);
        assert!(!e.baseline_missing);
        // missing baseline
        let e = encode_design(&m, "Control").unwrap();
        assert!(e.baseline_missing);
    }

    #[test]
    fn encode_continuous_and_drop_missing() {
        let m = meta(&["1.5", "2", "3", "10"], &[&["0.1", "NA", "0.3", "0.4"]]);
        let e = encode_design(&m, "Control").unwrap();
        assert_eq!(e.dropped_samples, vec!["s1".to_string()]);
        assert_eq!(e.sample_rows, vec![0, 2, 3]);
        match &e.phenotype {
            Phenotype::Continuous {
                values,
                scale_factor,
            } => {
                assert_eq!(values.len(), 3);
                assert!((values.iter().sum::<f64>()).abs() < 1e-12);
                assert!(*scale_factor > 0.0);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn intron_names() {
        assert_eq!(
            cluster_id("chr1:100:200:clu_5_NA").unwrap(),
            "chr1:clu_5_NA"
        );
        let p = parse_intron("chr10:180128:197910:clu_2_+:NE").unwrap();
        assert_eq!(
            (p.chr, p.start, p.end, p.clu, p.annotation),
            ("chr10", 180128, 197910, "clu_2_+", Some("NE"))
        );
        assert!(cluster_id("bad").is_err());
    }
}
