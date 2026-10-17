//! ALICE-Law G13: outputs that must not depend on `HashMap` / `HashSet`
//! iteration order or on the number of threads
//!
//! `HashMap` uses a randomly seeded hasher, so every map instance iterates in
//! its own order; building the same content many times exercises many orders.

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
