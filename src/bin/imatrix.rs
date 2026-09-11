//! ALICE-Dynamic-v1 tier-decision CLI.
//!
//! Usage:
//!   cargo run --release --features imatrix --bin alice-llm-imatrix -- \
//!     decide-heuristic \
//!     --model models/Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf \
//!     --target-size 3.5G \
//!     --output layer_assignments.json
//!
//! Output is a single JSON document mapping each tensor name to a quant
//! tier string (`"Q4_K"`, `"F32"`, ...) or the sentinel `"DROP"` if the
//! MTP-drop heuristic fires for the given size budget.
//!
//! Phase I.0 + I.3 only. The `decide-heuristic` subcommand does NOT require
//! calibration data — it applies the Unsloth-baseline heuristics extracted
//! by `memory/success_unsloth_v3_sensitivity_oracle_2026_09_12.md` directly
//! to the tensor layout of any input GGUF. Later phases add activation
//! capture (`collect`), imatrix-tuned assignment (`decide`), and repack.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use alice_llm::gguf::GgufFile;
use alice_llm::imatrix::{
    assign_tier, classify_layers, Assignment, BlockKind, LayerClassification, SizeBudget, TierMenu,
};

use memmap2::Mmap;
use serde::Serialize;

const HEURISTIC_REV: &str = "unsloth-baseline-v1";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<(), CliError> {
    let sub = args.get(1).map(String::as_str);
    match sub {
        Some("decide-heuristic") => run_decide_heuristic(&args[2..]),
        Some("--help" | "-h") | None => {
            print_help();
            Ok(())
        }
        Some(other) => Err(CliError::UnknownSubcommand(other.to_string())),
    }
}

fn print_help() {
    eprintln!(
        r#"alice-llm-imatrix — ALICE-Dynamic-v1 imatrix pipeline

Subcommands:
  decide-heuristic  Emit a layer-assignment JSON for the input GGUF using
                    the Unsloth-baseline heuristics (no calibration data).

decide-heuristic flags:
  --model PATH       Input GGUF file (required)
  --target-size N[K|M|G]
                     Target output size, e.g. "3.5G", "9800M" (required)
  --output PATH      Output JSON path (required)
"#
    );
}

fn run_decide_heuristic(args: &[String]) -> Result<(), CliError> {
    let params = DecideParams::from_args(args)?;
    let file = File::open(&params.model)
        .map_err(|e| CliError::Io(format!("open {}: {e}", params.model.display())))?;
    // SAFETY: memmap2::Mmap::map requires the caller to ensure the file is
    // not mutated concurrently. This CLI opens the file read-only and holds
    // the borrow for the duration of parsing.
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| CliError::Io(format!("mmap {}: {e}", params.model.display())))?;
    let gguf = GgufFile::parse(&mmap).ok_or(CliError::GgufParse)?;

    let arch = gguf
        .meta_str("general.architecture")
        .unwrap_or("<unknown>")
        .to_string();
    let block_count = detect_block_count(&gguf, &arch);
    let classification = classify_layers(gguf.tensors.values());
    let menu = TierMenu::for_budget(params.target_size);

    let mut assignments: Vec<(String, &'static str)> = gguf
        .tensors
        .keys()
        .map(|name| {
            let tag = match assign_tier(name, &classification, &menu, params.target_size) {
                Assignment::Keep(tier) => tier.as_str(),
                Assignment::Drop => "DROP",
            };
            (name.clone(), tag)
        })
        .collect();
    assignments.sort_by(|a, b| tensor_name_key(&a.0).cmp(&tensor_name_key(&b.0)));

    let out = LayerAssignmentsOutput {
        model: file_stem(&params.model),
        target_size_gb: params.target_size.total_gb(),
        target_size_bytes: params.target_size.total_bytes(),
        heuristic_rev: HEURISTIC_REV,
        arch,
        block_count,
        max_observed_block: classification.max_block,
        late_layer_threshold: classification.late_layer_threshold(),
        total_tensors: gguf.tensors.len(),
        tier_menu: TierMenuOutput {
            high: menu.high.as_str(),
            mid: menu.mid.as_str(),
            mid_low: menu.mid_low.as_str(),
            low: menu.low.as_str(),
        },
        layer_assignments: assignments
            .into_iter()
            .map(|(n, t)| (n, t.to_string()))
            .collect(),
    };
    let json = serde_json::to_string_pretty(&out).map_err(|e| CliError::Json(e.to_string()))?;
    let mut sink = File::create(&params.output)
        .map_err(|e| CliError::Io(format!("create {}: {e}", params.output.display())))?;
    sink.write_all(json.as_bytes())
        .map_err(|e| CliError::Io(format!("write {}: {e}", params.output.display())))?;
    sink.write_all(b"\n")
        .map_err(|e| CliError::Io(format!("write {}: {e}", params.output.display())))?;

    // Human-readable summary to stderr so JSON stdout stays clean (should
    // the user pipe --output /dev/stdout later).
    print_summary(&out, &classification);
    Ok(())
}

fn print_summary(out: &LayerAssignmentsOutput, classification: &LayerClassification) {
    eprintln!("=== alice-llm-imatrix decide-heuristic ===");
    eprintln!("  model:        {}", out.model);
    eprintln!("  arch:         {}", out.arch);
    eprintln!(
        "  blocks:       {} declared / {} observed",
        out.block_count.unwrap_or(0),
        out.max_observed_block + 1
    );
    eprintln!(
        "  budget:       {:.2} GB ({} bytes)",
        out.target_size_gb, out.target_size_bytes
    );
    eprintln!(
        "  tier menu:    high={} mid={} mid_low={} low={}",
        out.tier_menu.high, out.tier_menu.mid, out.tier_menu.mid_low, out.tier_menu.low
    );
    eprintln!("  heuristic:    {}", out.heuristic_rev);
    eprintln!(
        "  tensors:      {} assignments emitted",
        out.layer_assignments.len()
    );

    let pure = classification
        .kinds
        .values()
        .filter(|k| matches!(k, BlockKind::PureAttention))
        .count();
    let delta = classification
        .kinds
        .values()
        .filter(|k| matches!(k, BlockKind::DeltaNetHybrid))
        .count();
    eprintln!("  hybrid split: {pure} pure-attention + {delta} DeltaNet blocks");
}

// Sort key: for `blk.N.<part>.weight` return `(N, part)` for numerically
// stable ordering (`blk.10` after `blk.9`). Non-block tensors sort first
// by name.
fn tensor_name_key(name: &str) -> (u32, &str) {
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(dot) = rest.find('.') {
            let (idx_str, tail) = rest.split_at(dot);
            if let Ok(idx) = idx_str.parse::<u32>() {
                return (idx + 1, tail);
            }
        }
    }
    (0, name)
}

fn detect_block_count(gguf: &GgufFile<'_>, arch: &str) -> Option<u64> {
    let key = format!("{arch}.block_count");
    if let Some(v) = gguf.meta_u32(&key) {
        return Some(u64::from(v));
    }
    // Fallback: try a couple of common architecture-agnostic keys.
    for k in ["llama.block_count", "qwen35.block_count", "llm.block_count"] {
        if let Some(v) = gguf.meta_u32(k) {
            return Some(u64::from(v));
        }
    }
    None
}

fn file_stem(path: &std::path::Path) -> String {
    path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("<unknown>")
        .to_string()
}

#[derive(Debug)]
struct DecideParams {
    model: PathBuf,
    target_size: SizeBudget,
    output: PathBuf,
}

impl DecideParams {
    fn from_args(args: &[String]) -> Result<Self, CliError> {
        let mut model: Option<PathBuf> = None;
        let mut target: Option<SizeBudget> = None;
        let mut output: Option<PathBuf> = None;
        let mut iter = args.iter();
        while let Some(flag) = iter.next() {
            match flag.as_str() {
                "--model" => {
                    let v = iter.next().ok_or(CliError::MissingValue("--model"))?;
                    model = Some(PathBuf::from(v));
                }
                "--target-size" => {
                    let v = iter.next().ok_or(CliError::MissingValue("--target-size"))?;
                    target = Some(parse_size(v)?);
                }
                "--output" => {
                    let v = iter.next().ok_or(CliError::MissingValue("--output"))?;
                    output = Some(PathBuf::from(v));
                }
                other => return Err(CliError::UnknownFlag(other.to_string())),
            }
        }
        Ok(Self {
            model: model.ok_or(CliError::MissingValue("--model"))?,
            target_size: target.ok_or(CliError::MissingValue("--target-size"))?,
            output: output.ok_or(CliError::MissingValue("--output"))?,
        })
    }
}

fn parse_size(input: &str) -> Result<SizeBudget, CliError> {
    if input.is_empty() {
        return Err(CliError::BadSize(input.to_string()));
    }
    let (num_str, mul): (&str, u64) = match input.chars().last() {
        Some('G' | 'g') => (&input[..input.len() - 1], 1_000_000_000),
        Some('M' | 'm') => (&input[..input.len() - 1], 1_000_000),
        Some('K' | 'k') => (&input[..input.len() - 1], 1_000),
        Some(c) if c.is_ascii_digit() => (input, 1),
        _ => return Err(CliError::BadSize(input.to_string())),
    };
    let value: f64 = num_str
        .parse()
        .map_err(|_| CliError::BadSize(input.to_string()))?;
    if value <= 0.0 || !value.is_finite() {
        return Err(CliError::BadSize(input.to_string()));
    }
    let bytes = (value * (mul as f64)).round() as u64;
    SizeBudget::from_bytes(bytes).ok_or_else(|| CliError::BadSize(input.to_string()))
}

#[derive(Debug, Serialize)]
struct LayerAssignmentsOutput {
    model: String,
    target_size_gb: f32,
    target_size_bytes: u64,
    heuristic_rev: &'static str,
    arch: String,
    block_count: Option<u64>,
    max_observed_block: u32,
    late_layer_threshold: u32,
    total_tensors: usize,
    tier_menu: TierMenuOutput,
    // BTreeMap so JSON output is stable across runs.
    layer_assignments: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
struct TierMenuOutput {
    high: &'static str,
    mid: &'static str,
    mid_low: &'static str,
    low: &'static str,
}

#[derive(Debug)]
enum CliError {
    Io(String),
    GgufParse,
    Json(String),
    MissingValue(&'static str),
    UnknownFlag(String),
    UnknownSubcommand(String),
    BadSize(String),
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "io: {msg}"),
            Self::GgufParse => write!(
                f,
                "failed to parse GGUF (bad magic, truncated file, or unsupported version)"
            ),
            Self::Json(msg) => write!(f, "json: {msg}"),
            Self::MissingValue(flag) => write!(f, "missing value for {flag}"),
            Self::UnknownFlag(flag) => write!(f, "unknown flag: {flag}"),
            Self::UnknownSubcommand(sub) => write!(f, "unknown subcommand: {sub}"),
            Self::BadSize(s) => write!(
                f,
                "invalid size: {s} (expected e.g. 3.5G, 9800M, 1500K, or bytes)"
            ),
        }
    }
}
