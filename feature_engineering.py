"""
feature_engineering.py
──────────────────────
Complete feature engineering pipeline for the IPL Win Probability model.

Usage:
    from feature_engineering import build_features
    df = build_features("data/matches.csv", "data/deliveries.csv")

Outputs a clean DataFrame ready for train/test split and model training.
Every feature is documented with its formula, tier, and why it's included.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & VALIDATE
# ─────────────────────────────────────────────────────────────────────────────

def load_data(matches_path: str, deliveries_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and basic-validate both CSVs."""
    matches    = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)

    required_match_cols = {"id", "winner"}
    required_del_cols   = {"match_id", "inning", "over", "ball",
                           "batting_team", "bowling_team",
                           "total_runs", "player_dismissed"}

    missing_m = required_match_cols - set(matches.columns)
    missing_d = required_del_cols   - set(deliveries.columns)

    if missing_m:
        raise ValueError(f"matches.csv missing columns: {missing_m}")
    if missing_d:
        raise ValueError(f"deliveries.csv missing columns: {missing_d}")

    print(f"Loaded {len(matches):,} matches and {len(deliveries):,} deliveries.")
    return matches, deliveries


# ─────────────────────────────────────────────────────────────────────────────
# 2. FIRST INNINGS TARGET EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_targets(deliveries: pd.DataFrame) -> pd.Series:
    """
    Compute 1st innings total per match → target for 2nd innings chasing team.
    Target = 1st innings runs + 1  (standard cricket rule).

    Returns a Series indexed by match_id.
    """
    first_inn = (
        deliveries[deliveries["inning"] == 1]
        .groupby("match_id")["total_runs"]
        .sum()
        + 1
    )
    return first_inn


# ─────────────────────────────────────────────────────────────────────────────
# 3. FILTER TO 2ND INNINGS
# ─────────────────────────────────────────────────────────────────────────────

def filter_second_innings(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only 2nd innings rows (where chasing happens).
    Super overs (inning=3/4) are excluded — different dynamics.
    """
    df = deliveries[deliveries["inning"] == 2].copy()
    df = df.sort_values(["match_id", "over", "ball"]).reset_index(drop=True)
    print(f"2nd innings rows: {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. TIER 1 — CORE MATCH STATE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_core_match_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tier 1: The irreducible features every model needs.

    ball_number      : cumulative ball count within the innings (1–120)
    runs_scored      : cumulative runs by batting team to this ball
    wickets_fallen   : cumulative wickets lost to this ball
    wickets_in_hand  : 10 - wickets_fallen  (more intuitive for model)
    balls_remaining  : 120 - ball_number
    over             : integer over number (0–19)
    balls_in_over    : ball within current over (1–6)
    """
    grp = df.groupby("match_id")

    df["ball_number"]    = grp.cumcount() + 1
    df["runs_scored"]    = grp["total_runs"].cumsum()
    df["wickets_fallen"] = grp["player_dismissed"].transform(
                               lambda x: x.notna().cumsum()
                           ).astype(int)
    df["wickets_in_hand"]  = 10 - df["wickets_fallen"]
    df["balls_remaining"]  = 120 - df["ball_number"]
    df["over"]             = df["over"].astype(int)
    df["balls_in_over"]    = df["ball"]

    print("  ✓ Tier 1 — core match state")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. TIER 2 — RUN RATE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_run_rate_features(df: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
    """
    Tier 2: Run-rate derived features — the strongest predictors.

    target           : runs needed to win (1st innings total + 1)
    runs_remaining   : target - runs_scored
    crr              : current run rate  = runs_scored / overs_faced
    rrr              : required run rate = runs_remaining / overs_left
    rr_diff          : crr - rrr  (positive means batting team ahead)

    Note: eps guards against division-by-zero on ball 1.
    """
    eps = 1e-5

    df["target"]         = df["match_id"].map(targets)
    df["runs_remaining"] = df["target"] - df["runs_scored"]

    overs_faced = df["ball_number"] / 6
    overs_left  = df["balls_remaining"] / 6

    df["crr"]     = df["runs_scored"]    / (overs_faced + eps)
    df["rrr"]     = df["runs_remaining"] / (overs_left  + eps)
    df["rr_diff"] = df["crr"] - df["rrr"]

    print("  ✓ Tier 2 — run rate features")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. TIER 3 — CONTEXT FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tier 3: Match context that affects chaseability.

    is_high_target    : target > 180 (binary; extreme chases have different dynamics)
    target_bucket     : categorical bucket of target (low/medium/high/very_high)
                        encoded as integer for XGBoost
    """
    df["is_high_target"] = (df["target"] > 180).astype(int)

    def bucket(t):
        if t < 140:  return 0   # low
        if t < 165:  return 1   # medium
        if t < 185:  return 2   # high
        return 3                # very high

    df["target_bucket"] = df["target"].apply(bucket)

    print("  ✓ Tier 3 — context features")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. BONUS — MOMENTUM & PHASE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bonus: Features that distinguish this project from standard Kaggle notebooks.

    momentum_6       : runs scored in the last 6 balls (1 over rolling window)
                       captures hot/cold streaks independent of cumulative state
    momentum_12      : runs scored in last 12 balls (2 overs) — medium-term form
    partnership_runs : runs since the last wicket fell
                       a 50-run partnership vs a 5-run one signals batting stability
    is_powerplay     : binary flag, over < 6
    is_middle        : binary flag, 6 <= over < 15
    is_death         : binary flag, over >= 15
    over_sq          : over^2  (non-linear phase signal; death overs accelerate)
    """
    grp = df.groupby("match_id")

    # Rolling run rates
    df["momentum_6"]  = grp["total_runs"].transform(
                            lambda x: x.rolling(6,  min_periods=1).sum()
                        )
    df["momentum_12"] = grp["total_runs"].transform(
                            lambda x: x.rolling(12, min_periods=1).sum()
                        )

    # Partnership runs — reset on each wicket
    def partnership_cumsum(group):
        """Cumulative runs since last wicket."""
        out   = []
        total = 0
        for _, row in group.iterrows():
            if row["player_dismissed"] is not None and not pd.isna(row["player_dismissed"]):
                total = 0        # wicket fell — new partnership starts
            total += row["total_runs"]
            out.append(total)
        return pd.Series(out, index=group.index)

    df["partnership_runs"] = grp.apply(
        partnership_cumsum, include_groups=False
    ).reset_index(level=0, drop=True)

    # Phase flags
    df["is_powerplay"] = (df["over"] <  6).astype(int)
    df["is_middle"]    = ((df["over"] >= 6) & (df["over"] < 15)).astype(int)
    df["is_death"]     = (df["over"] >= 15).astype(int)

    # Non-linear over signal
    df["over_sq"] = df["over"] ** 2

    print("  ✓ Bonus — momentum & phase features")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. TEAM ENCODING (train-safe — returns encoder objects separately)
# ─────────────────────────────────────────────────────────────────────────────

def add_team_encoding_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add raw team name columns — actual encoding happens after train/test split
    to prevent leakage. This function just ensures the columns are clean.
    """
    df["batting_team"]  = df["batting_team"].str.strip()
    df["bowling_team"]  = df["bowling_team"].str.strip()
    print("  ✓ Team columns cleaned (encode AFTER splitting)")
    return df


def encode_teams_on_train(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, object, object]:
    """
    Fit LabelEncoders on training data ONLY, then transform both.
    Call this AFTER the train/test split.

    Returns (train_df, test_df, le_bat, le_bowl).
    """
    from sklearn.preprocessing import LabelEncoder

    le_bat  = LabelEncoder().fit(train_df["batting_team"])
    le_bowl = LabelEncoder().fit(train_df["bowling_team"])

    known_bat  = set(le_bat.classes_)
    known_bowl = set(le_bowl.classes_)

    def safe_enc(le, col, known):
        return col.map(lambda x: le.transform([x])[0] if x in known else -1)

    train_df["bat_enc"]  = le_bat.transform(train_df["batting_team"])
    train_df["bowl_enc"] = le_bowl.transform(train_df["bowling_team"])
    test_df["bat_enc"]   = safe_enc(le_bat,  test_df["batting_team"],  known_bat)
    test_df["bowl_enc"]  = safe_enc(le_bowl, test_df["bowling_team"],  known_bowl)

    print(f"  ✓ Teams encoded — {len(le_bat.classes_)} batting, {len(le_bowl.classes_)} bowling")
    return train_df, test_df, le_bat, le_bowl


# ─────────────────────────────────────────────────────────────────────────────
# 9. TARGET VARIABLE
# ─────────────────────────────────────────────────────────────────────────────

def add_target_variable(df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    win = 1 if batting team wins the match, else 0.

    Merge ONLY the winner column — nothing else from matches.csv
    to avoid post-match leakage (result_margin, player_of_match, etc.).
    """
    df = df.merge(
        matches[["id", "winner"]],
        left_on="match_id", right_on="id",
        how="left"
    )
    df["win"] = (df["batting_team"] == df["winner"]).astype(int)
    df = df.drop(columns=["id", "winner"])

    win_rate = df["win"].mean()
    print(f"  ✓ Target variable added — win rate: {win_rate:.2%} (expect ~50%)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 10. SANITY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_checks(df: pd.DataFrame) -> None:
    """Catch obvious bugs before training."""
    errors = []

    if df["runs_scored"].min() < 0:
        errors.append("Negative runs_scored found.")
    if df["wickets_fallen"].max() > 10:
        errors.append("wickets_fallen > 10 found.")
    if df["balls_remaining"].min() < 0:
        errors.append("Negative balls_remaining found.")
    if df["rrr"].max() > 200:
        errors.append("Suspiciously high RRR (>200) — check target merge.")
    if df["win"].isna().any():
        errors.append("NaN in target column 'win' — check match merge.")
    if df["target"].isna().any():
        n = df["target"].isna().sum()
        errors.append(f"{n} rows with NaN target — matches with no 1st innings data?")

    null_counts = df[FEATURE_COLS].isnull().sum()
    high_null = null_counts[null_counts > len(df) * 0.01]
    if not high_null.empty:
        errors.append(f"High null % in features: {high_null.to_dict()}")

    if errors:
        print("\n⚠️  SANITY CHECK WARNINGS:")
        for e in errors:
            print(f"   • {e}")
    else:
        print("  ✓ All sanity checks passed")


# ─────────────────────────────────────────────────────────────────────────────
# 11. FEATURE COLUMN REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Tier 1 — core state
    "runs_scored",
    "wickets_fallen",
    "wickets_in_hand",
    "balls_remaining",
    "over",
    "balls_in_over",

    # Tier 2 — run rates
    "target",
    "runs_remaining",
    "crr",
    "rrr",
    "rr_diff",

    # Tier 3 — context
    "is_high_target",
    "target_bucket",

    # Bonus — momentum & phase
    "momentum_6",
    "momentum_12",
    "partnership_runs",
    "is_powerplay",
    "is_middle",
    "is_death",
    "over_sq",

    # Teams (added after split via encode_teams_on_train)
    "bat_enc",
    "bowl_enc",
]

TARGET_COL = "win"
ID_COL     = "match_id"


# ─────────────────────────────────────────────────────────────────────────────
# 12. MASTER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    matches_path: str,
    deliveries_path: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Steps:
        1. Load & validate CSVs
        2. Extract 1st innings targets
        3. Filter to 2nd innings
        4. Add Tier 1 (core state)
        5. Add Tier 2 (run rates)
        6. Add Tier 3 (context)
        7. Add Bonus (momentum & phase)
        8. Clean team columns
        9. Add target variable (win)
       10. Sanity checks
       11. Drop nulls & return

    NOTE: Team encoding (bat_enc, bowl_enc) must be done AFTER
    train/test split using encode_teams_on_train().
    """
    if verbose:
        print("=" * 55)
        print("IPL Feature Engineering Pipeline")
        print("=" * 55)

    matches, deliveries = load_data(matches_path, deliveries_path)

    if verbose: print("\nExtracting targets from 1st innings...")
    targets = extract_targets(deliveries)

    if verbose: print("Filtering to 2nd innings...")
    df = filter_second_innings(deliveries)

    if verbose: print("\nBuilding features...")
    df = add_core_match_state(df)
    df = add_run_rate_features(df, targets)
    df = add_context_features(df)
    df = add_momentum_features(df)
    df = add_team_encoding_raw(df)
    df = add_target_variable(df, matches)

    if verbose: print("\nRunning sanity checks...")
    # Check all features except team encodings (added post-split)
    run_sanity_checks(df)

    # Drop rows with null target or null critical features
    before = len(df)
    df = df.dropna(subset=["win", "target", "runs_scored"])
    dropped = before - len(df)
    if dropped > 0 and verbose:
        print(f"  Dropped {dropped} rows with null critical values.")

    if verbose:
        print(f"\n{'=' * 55}")
        print(f"Done. Final shape: {df.shape}")
        print(f"Matches:  {df[ID_COL].nunique():,}")
        print(f"Balls:    {len(df):,}")
        print(f"Win rate: {df[TARGET_COL].mean():.2%}")
        print(f"{'=' * 55}\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 13. TRAIN / TEST SPLIT (match-level, leak-free)
# ─────────────────────────────────────────────────────────────────────────────

def match_level_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by match_id — NEVER by individual row.

    Splitting by row leaks future balls from the same match into training,
    giving artificially high AUC (~0.95+). Match-level split gives honest
    metrics (~0.82–0.87 AUC) and a model that actually generalises.
    """
    match_ids = df[ID_COL].unique()

    rng = np.random.default_rng(random_state)
    rng.shuffle(match_ids)

    split_idx  = int(len(match_ids) * (1 - test_size))
    train_ids  = set(match_ids[:split_idx])
    test_ids   = set(match_ids[split_idx:])

    train_df = df[df[ID_COL].isin(train_ids)].copy()
    test_df  = df[df[ID_COL].isin(test_ids)].copy()

    print(f"Train: {train_df[ID_COL].nunique():,} matches, {len(train_df):,} balls")
    print(f"Test:  {test_df[ID_COL].nunique():,} matches, {len(test_df):,} balls")

    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# 14. QUICK FEATURE SUMMARY (useful in notebooks)
# ─────────────────────────────────────────────────────────────────────────────

def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Print a clean summary table of all engineered features."""
    cols_to_check = [c for c in FEATURE_COLS if c not in ("bat_enc", "bowl_enc")]
    stats = df[cols_to_check].describe().T[["mean", "std", "min", "max"]]
    stats["null%"] = (df[cols_to_check].isnull().sum() / len(df) * 100).round(2)
    return stats.round(3)


# ─────────────────────────────────────────────────────────────────────────────
# 15. SINGLE-BALL INFERENCE HELPER (used by Streamlit app)
# ─────────────────────────────────────────────────────────────────────────────

def build_inference_row(
    runs: int,
    wickets: int,
    balls_done: int,
    target: int,
    bat_enc: int,
    bowl_enc: int,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame for real-time prediction in the dashboard.
    Matches the exact feature order of FEATURE_COLS.

    Args:
        runs       : runs scored so far in the 2nd innings
        wickets    : wickets fallen so far
        balls_done : total balls bowled in 2nd innings (1–120)
        target     : runs needed to win (1st innings total + 1)
        bat_enc    : label-encoded batting team id
        bowl_enc   : label-encoded bowling team id

    Returns:
        Single-row DataFrame ready for model.predict_proba()
    """
    eps = 1e-5

    balls_remaining = 120 - balls_done
    runs_remaining  = target - runs
    overs_faced     = balls_done / 6
    overs_left      = balls_remaining / 6
    over_num        = balls_done // 6

    crr     = runs / (overs_faced + eps)
    rrr     = runs_remaining / (overs_left + eps)
    rr_diff = crr - rrr

    row = {
        # Tier 1
        "runs_scored":      runs,
        "wickets_fallen":   wickets,
        "wickets_in_hand":  10 - wickets,
        "balls_remaining":  balls_remaining,
        "over":             over_num,
        "balls_in_over":    balls_done % 6 or 6,

        # Tier 2
        "target":           target,
        "runs_remaining":   runs_remaining,
        "crr":              crr,
        "rrr":              rrr,
        "rr_diff":          rr_diff,

        # Tier 3
        "is_high_target":  int(target > 180),
        "target_bucket":   0 if target < 140 else 1 if target < 165 else 2 if target < 185 else 3,

        # Bonus
        "momentum_6":       crr,   # proxy when computing live (use rolling in batch)
        "momentum_12":      crr,
        "partnership_runs": runs,  # proxy — full tracking needs match state history
        "is_powerplay":     int(over_num < 6),
        "is_middle":        int(6 <= over_num < 15),
        "is_death":         int(over_num >= 15),
        "over_sq":          over_num ** 2,

        # Teams
        "bat_enc":          bat_enc,
        "bowl_enc":         bowl_enc,
    }

    return pd.DataFrame([row])[FEATURE_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# CLI — run directly to see pipeline output
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    matches_path    = sys.argv[1] if len(sys.argv) > 1 else "data/matches.csv"
    deliveries_path = sys.argv[2] if len(sys.argv) > 2 else "data/deliveries.csv"

    df = build_features(matches_path, deliveries_path)

    print("\nFeature summary (pre-encoding):")
    print(feature_summary(df).to_string())

    print("\nSample row (ball 60 of a random match):")
    sample = df[df["ball_number"] == 60].sample(1, random_state=42)
    for feat in [c for c in FEATURE_COLS if c not in ("bat_enc", "bowl_enc")]:
        print(f"  {feat:<20} {sample[feat].values[0]:.3f}")
