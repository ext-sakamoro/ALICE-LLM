//! `RUNNING_SEMANTICS_ID` is a cfg alias of one of the two identifiers the G9 check reads
//! (`scripts/id_move.toml` lists `LLM_SEMANTICS_ID` and `LLM_PLATFORM_SEMANTICS_ID` and exempts
//! this alias): in every build it equals exactly one of them, the Law one without
//! `platform-math` and the platform one with it. CI runs this file in both builds.

use alice_llm::semantics::{LLM_PLATFORM_SEMANTICS_ID, LLM_SEMANTICS_ID, RUNNING_SEMANTICS_ID};

#[test]
fn the_running_identifier_is_exactly_one_of_the_two_checked_identifiers() {
    assert_ne!(LLM_SEMANTICS_ID, LLM_PLATFORM_SEMANTICS_ID);
    let matches = [LLM_SEMANTICS_ID, LLM_PLATFORM_SEMANTICS_ID]
        .iter()
        .filter(|id| **id == RUNNING_SEMANTICS_ID)
        .count();
    assert_eq!(
        matches, 1,
        "RUNNING_SEMANTICS_ID is neither or both of the checked identifiers"
    );
    let expected = if cfg!(feature = "platform-math") {
        LLM_PLATFORM_SEMANTICS_ID
    } else {
        LLM_SEMANTICS_ID
    };
    assert_eq!(RUNNING_SEMANTICS_ID, expected);
}
