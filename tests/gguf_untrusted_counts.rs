//! A count read from a GGUF header must not decide how much memory the parser reserves, and a file's
//! nesting must not decide how deep the parser recurses: the metadata count, the tensor count, a tensor's
//! dimension count and a metadata array's length are checked against the bytes that remain before
//! anything is reserved for them, and arrays nested inside arrays end in `None` for a file that is cut,
//! however deep the file nests them.
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

    let mut b = header(0, 1);
    string(&mut b, "k");
    b.extend_from_slice(&9_u32.to_le_bytes()); // array
    b.extend_from_slice(&4_u32.to_le_bytes()); // of u32
    b.extend_from_slice(&(1_u64 << 40).to_le_bytes());
    assert!(!parses(&b, "array length"));
    compared += 1;

    assert_eq!(compared, 4);
}

#[test]
fn deeply_nested_arrays_end_without_exhausting_the_stack() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // 200 000 levels of a one-element array of arrays, then the file ends: a cut file, so `None`
    let mut b = header(0, 1);
    string(&mut b, "k");
    b.extend_from_slice(&9_u32.to_le_bytes());
    for _ in 0..200_000 {
        b.extend_from_slice(&9_u32.to_le_bytes());
        b.extend_from_slice(&1_u64.to_le_bytes());
    }
    // the default test thread stack (2 MiB): the depth of the file must not decide the recursion
    let parsed = std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(move || parses(&b, "nested arrays"))
        .unwrap()
        .join()
        .expect("the parser thread finished");
    assert!(!parsed);
}
