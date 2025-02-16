import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

###############################################################################
# 1) Load or create your dataset
###############################################################################
# Let's assume you already have a pandas DataFrame with columns:
#   "text", "CF", "intervention", "skill"
# and you've defined label mappings:
CF_label2id = {"B": 0, "GA": 1, "TA": 2, "": 3}
INT_label2id = {"EAR": 0, "CP": 1, "": 2}  # blank => 2
skill_label2id = {
   "RL": 0,  # Reflective Listening
    "G": 1,   # Genuineness
    "V": 2,   # Validation
    "A": 3,   # Affirmation
    "RA": 4,  # Respect for Autonomy
    "AP": 5,  # Asking for Permission
    "OQ": 6,  # Open-ended Question
    "": 7    # Neutral
}
NUM_SKILLS = len(skill_label2id)  # 8

df = pd.read_csv("data/htc_examples_ids_with_pure_skills.csv")

###############################################################################
# 2) Encode the labels numerically
###############################################################################

# 2.1 CF -> integer
df["CF"].fillna("", inplace=True)
df["CF_id"] = df["CF"].apply(lambda x: CF_label2id[x])

# 2.2 Intervention -> integer
df["IC"].fillna("", inplace=True)
df["intervention_id"] = df["IC"].apply(lambda x: INT_label2id[x])

# 2.3 skills -> int
def encode_skill(skill_label):
  return skill_label2id[skill_label]

df["skill"].fillna("", inplace=True)
df["skill"] = df["skill"].apply(encode_skill)

# ------------------------------------------------------------------
# 1) Suppose you already have df with "text", "CF_id", "intervention_id", "skill"
#    Combine them into a single multi-output array Y (shape: Nx3)
# ------------------------------------------------------------------
df["combined_labels"] = df.apply(lambda row: [row["CF_id"], row["intervention_id"], row["skill"]], axis=1)
Y = np.vstack(df["combined_labels"].values)

X = df["text"].values

# ------------------------------------------------------------------
# 2) Define pipeline
# ------------------------------------------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))
])

# ------------------------------------------------------------------
# 3) K-Fold Cross Validation
# ------------------------------------------------------------------
k = 3
kfold = KFold(n_splits=k, shuffle=True, random_state=42)

cf_f1_micro_scores, cf_f1_macro_scores = [], []
int_f1_micro_scores, int_f1_macro_scores = [], []
skill_f1_micro_scores, skill_f1_macro_scores = [], []

fold_idx = 1
for train_index, test_index in kfold.split(X):
    print(f"\n===== Fold {fold_idx} / {k} =====")
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    pipeline.fit(X_train, Y_train)
    Y_pred = pipeline.predict(X_test)

    true_cf, pred_cf = Y_test[:, 0], Y_pred[:, 0]
    true_int, pred_int = Y_test[:, 1], Y_pred[:, 1]
    true_skill, pred_skill = Y_test[:, 2], Y_pred[:, 2]

    # CF
    cf_f1_micro = f1_score(true_cf, pred_cf, average="micro", zero_division=0)
    cf_f1_macro = f1_score(true_cf, pred_cf, average="macro", zero_division=0)
    cf_f1_micro_scores.append(cf_f1_micro)
    cf_f1_macro_scores.append(cf_f1_macro)
    print(f"  CF F1 - micro: {cf_f1_micro:.4f}, macro: {cf_f1_macro:.4f}")

    # Intervention
    int_f1_micro = f1_score(true_int, pred_int, average="micro", zero_division=0)
    int_f1_macro = f1_score(true_int, pred_int, average="macro", zero_division=0)
    int_f1_micro_scores.append(int_f1_micro)
    int_f1_macro_scores.append(int_f1_macro)
    print(f"  Intervention F1 - micro: {int_f1_micro:.4f}, macro: {int_f1_macro:.4f}")

    # Skill
    skill_f1_micro = f1_score(true_skill, pred_skill, average="micro", zero_division=0)
    skill_f1_macro = f1_score(true_skill, pred_skill, average="macro", zero_division=0)
    skill_f1_micro_scores.append(skill_f1_micro)
    skill_f1_macro_scores.append(skill_f1_macro)
    print(f"  Skill F1 - micro: {skill_f1_micro:.4f}, macro: {skill_f1_macro:.4f}")

    fold_idx += 1

# ------------------------------------------------------------------
# 4) Aggregate Results
# ------------------------------------------------------------------
print("\n\n===== FINAL CROSS-VALIDATION RESULTS (3-FOLD) =====")

print("--- CF ---")
print(f"CF F1-micro: {np.mean(cf_f1_micro_scores):.4f} ± {np.std(cf_f1_micro_scores):.4f}")
print(f"CF F1-macro: {np.mean(cf_f1_macro_scores):.4f} ± {np.std(cf_f1_macro_scores):.4f}")

print("\n--- Intervention ---")
print(f"Intervention F1-micro: {np.mean(int_f1_micro_scores):.4f} ± {np.std(int_f1_micro_scores):.4f}")
print(f"Intervention F1-macro: {np.mean(int_f1_macro_scores):.4f} ± {np.std(int_f1_macro_scores):.4f}")

print("\n--- Skill ---")
print(f"Skill F1-micro: {np.mean(skill_f1_micro_scores):.4f} ± {np.std(skill_f1_micro_scores):.4f}")
print(f"Skill F1-macro: {np.mean(skill_f1_macro_scores):.4f} ± {np.std(skill_f1_macro_scores):.4f}")