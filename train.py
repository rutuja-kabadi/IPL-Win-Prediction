import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import pickle

# ── Load ──────────────────────────────────────────────────
matches    = pd.read_csv("data/matches.csv")
deliveries = pd.read_csv("data/deliveries.csv")

# ── Merge ONLY the winner column — nothing else ───────────
df = deliveries.merge(
    matches[["id", "winner"]],   # <-- critical: don't pull in margin, POTM etc.
    left_on="match_id", right_on="id"
)

# ── 2nd innings only ──────────────────────────────────────
df = df[df["inning"] == 2].copy()
df = df.sort_values(["match_id", "over", "ball"]).reset_index(drop=True)

# ── Cumulative features (computed per ball WITHIN each match)
df["ball_number"]     = df.groupby("match_id").cumcount() + 1
df["runs_scored"]     = df.groupby("match_id")["total_runs"].cumsum()
df["wickets_fallen"]  = (
    df.groupby("match_id")["player_dismissed"]
      .transform(lambda x: x.notna().cumsum())
)

# ── Target from 1st innings ───────────────────────────────
first_inn_totals = (
    deliveries[deliveries["inning"] == 1]
    .groupby("match_id")["total_runs"].sum() + 1
)
df["target"] = df["match_id"].map(first_inn_totals)

# ── Derived features ──────────────────────────────────────
df["balls_remaining"] = 120 - df["ball_number"]
df["runs_remaining"]  = df["target"] - df["runs_scored"]
df["wickets_in_hand"] = 10 - df["wickets_fallen"]
df["over"]            = df["over"].astype(int)

eps = 1e-5
df["crr"] = df["runs_scored"] / (df["ball_number"] / 6 + eps)
df["rrr"] = df["runs_remaining"] / (df["balls_remaining"] / 6 + eps)
df["rr_diff"] = df["crr"] - df["rrr"]

# ── Momentum: runs in last 6 balls ───────────────────────
df["momentum"] = (
    df.groupby("match_id")["total_runs"]
      .transform(lambda x: x.rolling(6, min_periods=1).sum())
)

# ── Phase dummies ─────────────────────────────────────────
df["is_powerplay"] = (df["over"] < 6).astype(int)
df["is_death"]     = (df["over"] >= 15).astype(int)

# ── Target variable ───────────────────────────────────────
df["win"] = (df["batting_team"] == df["winner"]).astype(int)

# ════════════════════════════════════════════════════════
# SPLIT BY MATCH ID — the critical step
# ════════════════════════════════════════════════════════
match_ids = df["match_id"].unique()
np.random.seed(42)
np.random.shuffle(match_ids)

split      = int(0.8 * len(match_ids))
train_ids  = set(match_ids[:split])
test_ids   = set(match_ids[split:])

train_df = df[df["match_id"].isin(train_ids)].copy()
test_df  = df[df["match_id"].isin(test_ids)].copy()

# ── Encode teams on TRAIN only, transform both ────────────
le_bat  = LabelEncoder().fit(train_df["batting_team"])
le_bowl = LabelEncoder().fit(train_df["bowling_team"])

# handle unseen teams in test gracefully
def safe_transform(le, col):
    known = set(le.classes_)
    return col.map(lambda x: le.transform([x])[0] if x in known else -1)

train_df["bat_enc"]  = le_bat.transform(train_df["batting_team"])
train_df["bowl_enc"] = le_bowl.transform(train_df["bowling_team"])
test_df["bat_enc"]   = safe_transform(le_bat,  test_df["batting_team"])
test_df["bowl_enc"]  = safe_transform(le_bowl, test_df["bowling_team"])

# ── Feature matrix ────────────────────────────────────────
FEATURES = [
    "runs_scored", "wickets_fallen", "balls_remaining",
    "runs_remaining", "wickets_in_hand", "over",
    "crr", "rrr", "rr_diff", "target",
    "momentum", "is_powerplay", "is_death",
    "bat_enc", "bowl_enc"
]

X_train, y_train = train_df[FEATURES].fillna(0), train_df["win"]
X_test,  y_test  = test_df[FEATURES].fillna(0),  test_df["win"]

# ── Train ─────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    early_stopping_rounds=20,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# ── Evaluate — three metrics, not just AUC ────────────────
preds = model.predict_proba(X_test)[:, 1]
print(f"ROC-AUC      : {roc_auc_score(y_test, preds):.4f}")
print(f"Log Loss     : {log_loss(y_test, preds):.4f}")
print(f"Brier Score  : {brier_score_loss(y_test, preds):.4f}")

# ── Save ──────────────────────────────────────────────────
pickle.dump(model,   open("model/ipl_model.pkl",  "wb"))
pickle.dump(le_bat,  open("model/le_bat.pkl",     "wb"))
pickle.dump(le_bowl, open("model/le_bowl.pkl",    "wb"))
