import pandas as pd

BEH_KEYS_NEUTRAL = ["target_behavior", "alternative_behavior"]
BEH_KEYS_LEGACY = ["avoid_conflict", "approach_conflict_care"]


def _rename_behavior_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy behavior-keyed columns to neutral behavior-keyed columns."""
    renamed = frame.copy()
    prefixes = [
        "instinct_",
        "enjoyment_",
        "utility_",
        "suggest_term_",
        "momentary_urge_",
        "momentary_enjoyment_",
        "momentary_utility_",
        "report_urge_",
        "report_enjoyment_",
        "report_utility_",
    ]

    col_map = {}
    for old_key, new_key in zip(BEH_KEYS_LEGACY, BEH_KEYS_NEUTRAL):
        for prefix in prefixes:
            old_exact = f"{prefix}{old_key}"
            new_exact = f"{prefix}{new_key}"
            if old_exact in renamed.columns and new_exact not in renamed.columns:
                col_map[old_exact] = new_exact

            old_init = f"{prefix}{old_key}_0"
            new_init = f"{prefix}{new_key}_0"
            if old_init in renamed.columns and new_init not in renamed.columns:
                col_map[old_init] = new_init

            old_prefix = f"{prefix}{old_key}_"
            new_prefix = f"{prefix}{new_key}_"
            for col in renamed.columns:
                if col.startswith(old_prefix):
                    mapped = new_prefix + col[len(old_prefix):]
                    if mapped not in renamed.columns:
                        col_map[col] = mapped

    if col_map:
        renamed = renamed.rename(columns=col_map)
    return renamed


def ensure_neutral_behavior_schema(
    events_df: pd.DataFrame,
    people_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize events/people frames to neutral behavior keys.

    This function is intentionally idempotent: if data is already neutral, no-op.
    """
    ev = _rename_behavior_columns(events_df)
    ppl = _rename_behavior_columns(people_df)

    # Choice labels in events may still use legacy behavior names.
    if "choice_behavior" in ev.columns:
        ev = ev.copy()
        ev["choice_behavior"] = ev["choice_behavior"].replace(
            {
                BEH_KEYS_LEGACY[0]: BEH_KEYS_NEUTRAL[0],
                BEH_KEYS_LEGACY[1]: BEH_KEYS_NEUTRAL[1],
            }
        )

    if "observed_behavior" in ev.columns:
        ev = ev.copy()
        ev["observed_behavior"] = ev["observed_behavior"].replace(
            {
                BEH_KEYS_LEGACY[0]: BEH_KEYS_NEUTRAL[0],
                BEH_KEYS_LEGACY[1]: BEH_KEYS_NEUTRAL[1],
            }
        )

    if "learning_behavior" in ev.columns:
        ev = ev.copy()
        ev["learning_behavior"] = ev["learning_behavior"].replace(
            {
                BEH_KEYS_LEGACY[0]: BEH_KEYS_NEUTRAL[0],
                BEH_KEYS_LEGACY[1]: BEH_KEYS_NEUTRAL[1],
            }
        )

    return ev, ppl
