//! ALICE-Law G13: outputs that must not depend on `HashMap` / `HashSet`
//! iteration order or on the number of threads
//!
//! `HashMap` uses a randomly seeded hasher, so every map instance iterates in
//! its own order; building the same content many times exercises many orders.
// The tests build HashMaps on purpose, to feed the crate inputs in many orders
#![allow(clippy::disallowed_types)]

use std::collections::HashMap;

use alice_llm::tokenizer::BpeTokenizer;

/// Several strings share an id: the decoded string is the smallest of them,
/// whatever order the vocabulary map iterates in
#[test]
fn a_shared_token_id_decodes_to_the_smallest_string() {
    for _ in 0..32 {
        let mut vocab = HashMap::new();
        for (s, id) in [("b", 7_u32), ("a", 7), ("c", 7), ("x", 1), ("y", 2)] {
            vocab.insert(s.to_string(), id);
        }
        let t = BpeTokenizer::new(Vec::new(), vocab);
        assert_eq!(t.decode(&[7, 1, 2]), "axy");
    }
}

/// The bigram bias ranks by count, then by token id: the logits are the same
/// bits whatever order the counts were observed in
#[cfg(feature = "dspark")]
#[test]
fn bigram_bias_does_not_depend_on_observation_order() {
    use alice_llm::speculative_dspark::FullCountBigramBias;
    // 5 is seen twice and 2, 3, 4, 6 once each: the rank-3 cut falls inside the
    // tie, so which two of the four are kept depends on the tie-break alone
    let pairs: Vec<(u32, u32)> = vec![(0, 5), (0, 3), (0, 4), (0, 5), (0, 6), (0, 2)];
    let apply = |order: &[(u32, u32)]| {
        let mut b = FullCountBigramBias::new(8, 3).unwrap();
        for &(p, n) in order {
            b.observe(p, n).unwrap();
        }
        let mut logits = vec![0.0_f32; 8];
        b.apply(0, &mut logits, 0.5).unwrap();
        logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    };
    let want = apply(&pairs);
    // ranked by count, then token id: 5, then 2 and 3
    assert!(want[5] != 0 && want[2] != 0 && want[3] != 0, "{want:?}");
    assert!(want[4] == 0 && want[6] == 0, "{want:?}");
    let mut rev = pairs.clone();
    rev.reverse();
    assert_eq!(apply(&rev), want);
    for k in 1..pairs.len() {
        let mut rot = pairs.clone();
        rot.rotate_left(k);
        assert_eq!(apply(&rot), want, "rotation {k}");
    }
}

/// A row-parallel kernel gives the same bits with 1 and with several
/// threads (each output row is computed by one thread, in a fixed order)
#[cfg(all(feature = "gguf", feature = "parallel"))]
#[test]
fn ternary_matvec_is_the_same_on_one_and_four_threads() {
    use alice_llm::gguf::{ternary_matvec, TernaryMatrix, TernaryRow};
    let num_rows = 97;
    let num_cols = 256;
    let rows: Vec<TernaryRow> = (0..num_rows)
        .map(|r| {
            let w: Vec<f32> = (0..num_cols)
                .map(|c| {
                    let k = u16::try_from((r * 31 + c * 17) % 101).unwrap();
                    f32::from(k) / 50.0 - 1.0
                })
                .collect();
            TernaryRow::from_f32(&w, 0.7)
        })
        .collect();
    let m = TernaryMatrix {
        rows,
        num_rows,
        num_cols,
    };
    let input: Vec<f32> = (0..num_cols)
        .map(|c| f32::from(u16::try_from(c % 13).unwrap()) / 6.0 - 1.0)
        .collect();
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut out = vec![0.0_f32; num_rows];
        pool.install(|| ternary_matvec(&m, &input, &mut out));
        out.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    };
    let one = run(1);
    assert!(one.iter().any(|&b| b != 0), "the kernel computed nothing");
    assert_eq!(run(4), one);
    assert_eq!(run(7), one);
}

/// A string given twice keeps its smallest id, in either order of the pairs
#[test]
fn a_repeated_token_string_keeps_its_smallest_id() {
    let pairs = vec![
        ("a".to_string(), 5_u32),
        ("b".to_string(), 1),
        ("a".to_string(), 3),
    ];
    let mut rev = pairs.clone();
    rev.reverse();
    for input in [pairs, rev] {
        let t = BpeTokenizer::new(Vec::new(), input);
        assert_eq!(t.encode("ab"), vec![3, 1]);
        assert_eq!(t.decode(&[3, 5]), "a", "id 5 is no longer a's");
        assert_eq!(t.vocab_size(), 2);
    }
}

/// With two undefined rules, the error names the same one every time (rules
/// are validated in name order)
#[cfg(feature = "grammar")]
#[test]
fn the_undefined_rule_error_is_the_same_on_every_parse() {
    use alice_llm::grammar::{parse_gbnf, GbnfError};
    let src = "root ::= zeta | alpha\nzeta ::= missingz\nalpha ::= missinga\n";
    for _ in 0..32 {
        match parse_gbnf(src) {
            Err(GbnfError::UndefinedRule { name }) => assert_eq!(name, "missinga"),
            other => panic!("expected UndefinedRule, got {other:?}"),
        }
    }
}

/// The serialized bias table is the same bytes whatever order the pairs were
/// observed in (the counts are written in token order)
#[cfg(feature = "dspark-serde")]
#[test]
fn the_serialized_bigram_table_does_not_depend_on_observation_order() {
    use alice_llm::speculative_dspark::FullCountBigramBias;
    let pairs: Vec<(u32, u32)> = (0..40_u32).map(|i| ((i * 7) % 13, (i * 5) % 11)).collect();
    let bytes = |order: &[(u32, u32)]| {
        let mut b = FullCountBigramBias::new(16, 4).unwrap();
        for &(p, n) in order {
            b.observe(p, n).unwrap();
        }
        bincode::serialize(&b).unwrap()
    };
    let want = bytes(&pairs);
    for k in 0..pairs.len() {
        let mut rot = pairs.clone();
        rot.rotate_left(k);
        if k % 2 == 1 {
            rot.reverse();
        }
        assert_eq!(bytes(&rot), want, "order {k}");
    }
}

/// The public metadata and tensor maps of a parsed GGUF iterate in key order
#[test]
fn gguf_metadata_and_tensors_iterate_in_key_order() {
    use alice_llm::gguf::GgufFile;
    fn s(b: &mut Vec<u8>, x: &str) {
        b.extend_from_slice(&(x.len() as u64).to_le_bytes());
        b.extend_from_slice(x.as_bytes());
    }
    let keys = ["m.key", "z.key", "a.key", "q.key", "b.key"];
    let names = ["w.weight", "c.weight", "k.weight"];
    let mut b = Vec::new();
    b.extend_from_slice(&0x4655_4747_u32.to_le_bytes());
    b.extend_from_slice(&3_u32.to_le_bytes());
    b.extend_from_slice(&(names.len() as u64).to_le_bytes());
    b.extend_from_slice(&(keys.len() as u64).to_le_bytes());
    for k in keys {
        s(&mut b, k);
        b.extend_from_slice(&8_u32.to_le_bytes()); // string
        s(&mut b, "v");
    }
    for (i, n) in names.iter().enumerate() {
        s(&mut b, n);
        b.extend_from_slice(&1_u32.to_le_bytes());
        b.extend_from_slice(&4_u64.to_le_bytes());
        b.extend_from_slice(&0_u32.to_le_bytes()); // F32
        b.extend_from_slice(&((i * 32) as u64).to_le_bytes());
    }
    b.resize(b.len().next_multiple_of(32) + names.len() * 32, 0);
    let g = GgufFile::parse(&b).expect("parse");
    let got: Vec<&str> = g.metadata.keys().map(String::as_str).collect();
    assert_eq!(got, ["a.key", "b.key", "m.key", "q.key", "z.key"]);
    let got: Vec<&str> = g.tensors.keys().map(String::as_str).collect();
    assert_eq!(got, ["c.weight", "k.weight", "w.weight"]);
}

/// With a bad page id in every sequence but the first, the error names the
/// second sequence's id on any number of threads (the first error in row
/// order, as the single-threaded build reports it). Row 0 is valid and long
/// (256 blocks), so while it runs the other threads take later rows, fail on
/// their first block and stop the parallel walk; a run that returned the first
/// failure to happen instead of the first in row order would skip row 1 and
/// name a later row's id
#[cfg(feature = "parallel")]
#[test]
fn the_proxy_score_error_is_the_first_in_row_order() {
    use alice_llm::sparse_attention::proxy::compute_proxy_block_max_scores;
    use alice_llm::sparse_attention::types::{BlockTables, CuSeqlensQ, SparseAttentionError};
    let (batch, pages, page, dim) = (16_usize, 256_usize, 64_usize, 128_usize);
    let q: Vec<f32> = (0..batch * dim)
        .map(|i| f32::from(u8::try_from(i % 7).unwrap()) / 7.0)
        .collect();
    let k: Vec<f32> = (0..pages * page * dim)
        .map(|i| f32::from(u8::try_from(i % 11).unwrap()) / 11.0)
        .collect();
    let table: Vec<i32> = (0..batch * pages)
        .map(|n| {
            let (b, p) = (n / pages, n % pages);
            if b == 0 {
                i32::try_from(p).unwrap()
            } else if p == 0 {
                1000 + i32::try_from(b).unwrap()
            } else {
                0
            }
        })
        .collect();
    let tbl = BlockTables::new(table, batch, pages).unwrap();
    let cu = CuSeqlensQ::new((0..=batch).map(|b| i64::try_from(b).unwrap()).collect()).unwrap();
    let used = vec![i32::try_from(pages * page).unwrap(); batch];
    for threads in [1, 3, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for _ in 0..20 {
            let err = pool
                .install(|| {
                    compute_proxy_block_max_scores(
                        &q,
                        &k,
                        &tbl,
                        &cu,
                        Some(&used),
                        1,
                        dim,
                        page,
                        page,
                    )
                })
                .unwrap_err();
            match err {
                SparseAttentionError::ShapeMismatch { got, .. } => {
                    assert_eq!(got, 1001, "{threads} threads")
                }
                other => panic!("expected ShapeMismatch, got {other:?}"),
            }
        }
    }
}

/// The sparse attention forward pass reports the first failing work unit's
/// page id on any number of threads: one sequence, one work unit per selected
/// block; unit 0 is valid and long (256 pages), every later unit fails on its
/// first page, so a run that returned the first failure to happen would skip
/// unit 1 and name a later unit's id
#[cfg(feature = "parallel")]
#[test]
fn the_sparse_forward_error_is_the_first_in_unit_order() {
    use alice_llm::sparse_attention::forward::kvouter_forward;
    use alice_llm::sparse_attention::index::build_kvouter_index;
    use alice_llm::sparse_attention::types::{
        BlockTables, CuSeqlensQ, SparseAttentionError, SparseSelection,
    };
    let (blocks, ppb, page, dim) = (16_usize, 256_usize, 64_usize, 128_usize);
    let q: Vec<f32> = (0..blocks * dim)
        .map(|i| f32::from(u8::try_from(i % 7).unwrap()) / 7.0)
        .collect();
    let kv: Vec<f32> = (0..ppb * page * dim)
        .map(|i| f32::from(u8::try_from(i % 11).unwrap()) / 11.0)
        .collect();
    let table: Vec<i32> = (0..blocks * ppb)
        .map(|n| {
            let (j, p) = (n / ppb, n % ppb);
            if j == 0 {
                i32::try_from(p).unwrap()
            } else if p == 0 {
                1000 + i32::try_from(j).unwrap()
            } else {
                0
            }
        })
        .collect();
    let tbl = BlockTables::new(table, 1, blocks * ppb).unwrap();
    let cu = CuSeqlensQ::new(vec![0, i64::try_from(blocks).unwrap()]).unwrap();
    let used = vec![i32::try_from(blocks * ppb * page).unwrap()];
    // query token j selects block j
    let sel = SparseSelection::new(
        (0..blocks).map(|j| i32::try_from(j).unwrap()).collect(),
        blocks,
        1,
        1,
    )
    .unwrap();
    let block = ppb * page;
    let idx = build_kvouter_index(&sel, &tbl, &cu, Some(&used), block, page).unwrap();
    for threads in [1, 3, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for _ in 0..20 {
            let err = pool
                .install(|| {
                    kvouter_forward(
                        &q,
                        &kv,
                        &kv,
                        &idx,
                        &tbl,
                        &cu,
                        Some(&used),
                        1,
                        1,
                        dim,
                        block,
                        page,
                        1.0,
                        false,
                    )
                })
                .unwrap_err();
            match err {
                SparseAttentionError::ShapeMismatch { got, .. } => {
                    assert_eq!(got, 1001, "{threads} threads")
                }
                other => panic!("expected ShapeMismatch, got {other:?}"),
            }
        }
    }
}
