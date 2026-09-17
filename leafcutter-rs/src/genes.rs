//! Cluster-to-gene labelling from an exon table (`leafcutter.utils.map_clusters_to_genes`).

use crate::ds::Cluster;
use crate::io::{open_maybe_gz, parse_intron};
use std::collections::{BTreeSet, HashMap};
use std::io::BufRead;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Exon {
    pub chr: String,
    pub start: u64,
    pub end: u64,
    pub gene: String,
}

/// Read a tab/space separated exon table with a header naming at least `chr`, `start`,
/// `end` and `gene_name` (as produced by `leafcutter-gtf-to-exons`).
pub fn read_exons(path: &Path) -> Result<Vec<Exon>, String> {
    let reader = open_maybe_gz(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut lines = reader.lines();
    let header = lines
        .next()
        .ok_or("empty exon file")?
        .map_err(|e| e.to_string())?;
    let cols: Vec<&str> = header.split_whitespace().collect();
    let idx = |name: &str| {
        cols.iter()
            .position(|c| *c == name)
            .ok_or_else(|| format!("exon file has no '{name}' column"))
    };
    let (ic, is, ie, ig) = (idx("chr")?, idx("start")?, idx("end")?, idx("gene_name")?);
    let mut exons = Vec::new();
    for (ln, line) in lines.enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() {
            continue;
        }
        let need = ic.max(is).max(ie).max(ig);
        if t.len() <= need {
            return Err(format!("exon file line {}: too few columns", ln + 2));
        }
        let start = t[is]
            .parse::<f64>()
            .map_err(|_| format!("exon file line {}: bad start", ln + 2))?
            as u64;
        let end = t[ie]
            .parse::<f64>()
            .map_err(|_| format!("exon file line {}: bad end", ln + 2))? as u64;
        exons.push(Exon {
            chr: t[ic].to_string(),
            start,
            end,
            gene: t[ig].to_string(),
        });
    }
    Ok(exons)
}

/// `leafcutter.utils.add_chr`: if the first chromosome name lacks a `chr` prefix, prefix all.
fn add_chr_all(first: &str) -> bool {
    !first.contains("chr")
}

/// Genes per cluster: an exon whose start equals an intron end, or whose end equals an intron
/// start, on the same chromosome, labels the cluster. Names are comma-joined, sorted, unique.
pub fn map_clusters_to_genes(clusters: &[Cluster], exons: &[Exon]) -> HashMap<String, String> {
    let mut out = HashMap::new();
    if exons.is_empty() || clusters.is_empty() {
        return out;
    }
    let prefix_exons = add_chr_all(&exons[0].chr);
    let mut by_start: HashMap<(String, u64), Vec<&str>> = HashMap::new();
    let mut by_end: HashMap<(String, u64), Vec<&str>> = HashMap::new();
    for e in exons {
        let chr = if prefix_exons {
            format!("chr{}", e.chr)
        } else {
            e.chr.clone()
        };
        by_start
            .entry((chr.clone(), e.start))
            .or_default()
            .push(&e.gene);
        by_end.entry((chr, e.end)).or_default().push(&e.gene);
    }
    let first_intron = clusters.iter().find_map(|c| c.introns.first().cloned());
    let prefix_introns = first_intron
        .as_deref()
        .and_then(|s| parse_intron(s).ok())
        .map(|p| add_chr_all(p.chr))
        .unwrap_or(false);
    for c in clusters {
        let mut genes: BTreeSet<&str> = BTreeSet::new();
        for name in &c.introns {
            let Ok(p) = parse_intron(name) else { continue };
            let chr = if prefix_introns {
                format!("chr{}", p.chr)
            } else {
                p.chr.to_string()
            };
            if let Some(g) = by_start.get(&(chr.clone(), p.end)) {
                genes.extend(g.iter().copied());
            }
            if let Some(g) = by_end.get(&(chr, p.start)) {
                genes.extend(g.iter().copied());
            }
        }
        if !genes.is_empty() {
            out.insert(
                c.name.clone(),
                genes.into_iter().collect::<Vec<_>>().join(","),
            );
        }
    }
    out
}
