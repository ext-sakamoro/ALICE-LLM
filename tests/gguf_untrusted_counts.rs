//! A count read from a GGUF header must not decide how much memory the parser reserves, and a file's
//! nesting must not decide how deep the parser recurses: the metadata count, the tensor count, a tensor's
//! dimension count and a metadata array's length are checked against the bytes that remain before
//! anything is reserved for them, and arrays nested inside arrays are refused past a fixed depth, so the
//! parser's recursion is bounded by the parser and not by the file.
//! Measured with a capped global allocator: the largest single request may not exceed 16 MiB (a
//! request above it is refused, not made) and the peak of live bytes stays a small multiple of the file.

use alice_llm::gguf::GgufFile;
use std::alloc::{GlobalAlloc, Layout, System};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Largest single request the parser may make in this binary. A request above it is refused (null),
/// so a parser that sizes a table by a claimed count stops the process with "memory allocation of N
/// bytes failed" before anything that large is reserved, the same on every OS.
const CAP: usize = 16 * 1024 * 1024;

struct Capped;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

fn admit(size: usize) -> bool {
    LARGEST.fetch_max(size, Ordering::SeqCst);
    size <= CAP
}

fn note(size: usize) {
    let now = LIVE.fetch_add(size, Ordering::SeqCst) + size;
    PEAK.fetch_max(now, Ordering::SeqCst);
}

unsafe impl GlobalAlloc for Capped {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !admit(layout.size()) {
            return std::ptr::null_mut();
        }
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            note(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !admit(layout.size()) {
            return std::ptr::null_mut();
        }
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            note(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static ALLOC: Capped = Capped;

/// The tests of this binary measure one process-wide counter, so they run one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

/// Fixed cost of the parser apart from what it decodes.
const SLACK: usize = 64 * 1024;

/// Parses `bytes`; returns whether it parsed, after checking the request cap and the peak.
fn parses(bytes: &[u8], what: &str) -> bool {
    let base = LIVE.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    LARGEST.store(0, Ordering::SeqCst);
    let out = catch_unwind(AssertUnwindSafe(|| GgufFile::parse(bytes).is_some()));
    let largest = LARGEST.load(Ordering::SeqCst);
    let peak = PEAK.load(Ordering::SeqCst) - base;
    assert!(
        largest <= CAP,
        "{what}: a single request of {largest} bytes"
    );
    assert!(
        peak <= 64 * bytes.len() + SLACK,
        "{what}: parsing reserved {peak} bytes for a {}-byte file",
        bytes.len()
    );
    out.unwrap_or_else(|_| panic!("{what}: the parser panicked instead of returning None"))
}

const MAGIC: u32 = 0x4655_4747;

fn header(tensors: u64, metadata: u64) -> Vec<u8> {
    let mut b = MAGIC.to_le_bytes().to_vec();
    b.extend_from_slice(&3_u32.to_le_bytes());
    b.extend_from_slice(&tensors.to_le_bytes());
    b.extend_from_slice(&metadata.to_le_bytes());
    b
}

fn string(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

#[test]
fn a_well_formed_small_file_parses() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // one u32 metadata value and one 1-d tensor; the parser accepts it
    let mut b = header(1, 1);
    string(&mut b, "general.alignment");
    b.extend_from_slice(&4_u32.to_le_bytes());
    b.extend_from_slice(&32_u32.to_le_bytes());
    string(&mut b, "t");
    b.extend_from_slice(&1_u32.to_le_bytes());
    b.extend_from_slice(&4_u64.to_le_bytes());
    b.extend_from_slice(&0_u32.to_le_bytes()); // F32
    b.extend_from_slice(&0_u64.to_le_bytes());
    assert!(
        parses(&b, "well formed"),
        "the probe format matches the parser"
    );
}

#[test]
fn claimed_counts_are_not_reserved() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut compared = 0;

    let b = header(0, 1 << 40);
    assert!(!parses(&b, "metadata count"));
    compared += 1;

    let b = header(1 << 40, 0);
    assert!(!parses(&b, "tensor count"));
    compared += 1;

    let mut b = header(1, 0);
    string(&mut b, "t");
    b.extend_from_slice(&u32::MAX.to_le_bytes()); // dimension count
    assert!(!parses(&b, "dimension count"));
    compared += 1;

    // the same claim in a file that holds one whole tensor entry after it, so the tensor count fits
    // and only the bound on the dimension count can refuse it
    let mut b = header(1, 0);
    string(&mut b, "t");
    b.extend_from_slice(&u32::MAX.to_le_bytes()); // dimension count
    b.extend_from_slice(&[0; 16]);
    assert!(!parses(&b, "dimension count, room for one tensor"));
    compared += 1;

    let mut b = header(0, 1);
    string(&mut b, "k");
    b.extend_from_slice(&9_u32.to_le_bytes()); // array
    b.extend_from_slice(&4_u32.to_le_bytes()); // of u32
    b.extend_from_slice(&(1_u64 << 40).to_le_bytes());
    assert!(!parses(&b, "array length"));
    compared += 1;

    assert_eq!(compared, 5);
}

/// A complete file with one metadata array nested `depth` levels (arrays of one array, ending in an
/// array of one u32).
fn nested(depth: usize) -> Vec<u8> {
    let mut b = header(0, 1);
    string(&mut b, "k");
    b.extend_from_slice(&9_u32.to_le_bytes()); // the value is an array
    for _ in 1..depth {
        b.extend_from_slice(&9_u32.to_le_bytes()); // of arrays
        b.extend_from_slice(&1_u64.to_le_bytes()); // of length 1
    }
    b.extend_from_slice(&4_u32.to_le_bytes()); // the innermost array holds u32
    b.extend_from_slice(&1_u64.to_le_bytes());
    b.extend_from_slice(&7_u32.to_le_bytes());
    b
}

#[test]
fn array_nesting_is_limited_to_a_fixed_depth() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // shallow nesting parses; nesting far beyond any real model's metadata is refused, so the
    // recursion depth is bounded by the parser and not by the file (1024 levels fit the default test
    // stack, so this refusal is a limit, not a stack overflow)
    assert!(
        parses(&nested(4), "nesting 4"),
        "a shallow nested array parses"
    );
    assert!(
        !parses(&nested(1024), "nesting 1024"),
        "an array nested 1024 levels deep is refused by a depth limit"
    );
}

#[test]
fn each_refusal_says_what_was_refused() {
    use alice_llm::gguf::{GgufError, MAX_ARRAY_DEPTH};
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let err = |b: &[u8]| GgufFile::parse_checked(b).err();
    let count = |e: Option<GgufError>, what: &str, claimed: u64| matches!(e, Some(GgufError::CountExceedsFile { what: w, claimed: c, .. }) if w == what && c == claimed);
    let mut compared = 0;

    assert!(count(err(&header(0, 1 << 40)), "metadata count", 1 << 40));
    compared += 1;
    assert!(count(err(&header(1 << 40, 0)), "tensor count", 1 << 40));
    compared += 1;

    let mut b = header(1, 0);
    string(&mut b, "t");
    b.extend_from_slice(&u32::MAX.to_le_bytes());
    b.extend_from_slice(&[0; 16]);
    assert!(count(err(&b), "dimension count", u64::from(u32::MAX)));
    compared += 1;

    let mut b = header(0, 1);
    string(&mut b, "k");
    b.extend_from_slice(&9_u32.to_le_bytes());
    b.extend_from_slice(&4_u32.to_le_bytes());
    b.extend_from_slice(&(1_u64 << 40).to_le_bytes());
    assert!(count(err(&b), "metadata array length", 1 << 40));
    compared += 1;

    assert_eq!(
        err(&nested(1024)),
        Some(GgufError::NestingTooDeep {
            max: MAX_ARRAY_DEPTH
        })
    );
    compared += 1;

    // general.alignment 0 would divide by zero when the data section is aligned
    let mut b = header(0, 1);
    string(&mut b, "general.alignment");
    b.extend_from_slice(&4_u32.to_le_bytes());
    b.extend_from_slice(&0_u32.to_le_bytes());
    assert_eq!(err(&b), Some(GgufError::ZeroAlignment));
    compared += 1;

    let mut b = header(0, 1);
    string(&mut b, "k");
    b.extend_from_slice(&99_u32.to_le_bytes());
    assert_eq!(err(&b), Some(GgufError::UnknownType { vtype: 99 }));
    compared += 1;

    let mut b = header(0, 0);
    b[0] ^= 1;
    assert!(matches!(err(&b), Some(GgufError::BadMagic { .. })));
    let mut b = header(0, 0);
    b[4] = 9;
    assert_eq!(err(&b), Some(GgufError::UnsupportedVersion { got: 9 }));
    assert_eq!(err(&header(0, 0)[..10]), Some(GgufError::Truncated));
    compared += 3;

    // parse is parse_checked without the reason
    assert!(GgufFile::parse(&header(0, 1 << 40)).is_none());
    assert_eq!(compared, 10);
}
