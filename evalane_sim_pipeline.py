"""
╔══════════════════════════════════════════════════════════════════╗
║      Evalane — Simulation-Aligned Chained Pipeline              ║
║                                                                  ║
║  Features match exactly what the simulation provides:           ║
║    Lane counts, Congestion %, Current green lane,               ║
║    Green timer remaining, Event type & lane                     ║
║                                                                  ║
║  Chain:                                                          ║
║   Simulation Features                                            ║
║       │                                                          ║
║       ▼                                                          ║
║  [Model 1] XGBClassifier  →  Congestion_Level                   ║
║       │         (Low / Moderate / High / Severe)                 ║
║       ▼                                                          ║
║  [Model 2] XGBClassifier  →  Optimal_Action      ← primary      ║
║       │         (GREEN_N / GREEN_S / GREEN_E / GREEN_W)          ║
║       ▼                                                          ║
║  [Model 3] XGBRegressor   →  Signal_Duration     ← secondary    ║
║                               seconds, clipped [12, 83]         ║
║                                                                  ║
║  predict() returns:                                              ║
║    congestion_level      str    Low|Moderate|High|Severe         ║
║    congestion_confidence float  0.0 – 1.0                       ║
║    optimal_action        str    GREEN_N|S|E|W                    ║
║    action_confidence     float  0.0 – 1.0                       ║
║    action_probabilities  dict   {GREEN_N: p, …}                 ║
║    signal_duration       float  seconds                         ║
║    signal_duration_range list   [low, high] ±1 RMSE             ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import warnings
import pickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score

from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────
# 1.  Constants
# ─────────────────────────────────────────────────────────────────
DATA_PATH = "/home/a7madmj/Documents/Graduation/NEWWfiles/evalane_sim_dataset_v2.csv"

DURATION_MIN     = 12
DURATION_MAX     = 83
AMBULANCE_FLOOR  = 35     # minimum green time guaranteed for ambulance

CONGESTION_ORDER = ["Low", "Moderate", "High", "Severe"]

# Categorical columns in the dataset
CAT_COLS = ["Current_Green_Lane", "Event_Type", "Event_Lane"]

# The three targets — excluded from input features
_TARGETS = {"Congestion_Level", "Optimal_Action", "Signal_Duration"}

TARGET_CL  = "Congestion_Level"
TARGET_ACT = "Optimal_Action"
TARGET_DUR = "Signal_Duration"


# ─────────────────────────────────────────────────────────────────
# 2.  Preprocessing helper
# ─────────────────────────────────────────────────────────────────
def build_preprocessor(feature_cols: list) -> ColumnTransformer:
    cat_present = [c for c in CAT_COLS if c in feature_cols]
    num_cols    = [c for c in feature_cols if c not in CAT_COLS]

    transformers = []
    if cat_present:
        transformers.append((
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            cat_present,
        ))
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ─────────────────────────────────────────────────────────────────
# 3.  Chained Pipeline class
# ─────────────────────────────────────────────────────────────────
class EvalaneSimPipeline:
    """
    Simulation-aligned three-model chained pipeline.

    Input features (all available in the simulation):
        Lane_N/S/E/W_Count, Congestion_N/S/E/W,
        Current_Green_Lane, Green_Timer_Remaining,
        Event_Type, Event_Lane

    Chain:
        features → Model1 → Congestion_Level
        features + CL → Model2 → Optimal_Action   (primary output)
        features + CL + OA → Model3 → Signal_Duration (secondary)
    """

    def __init__(self):
        self.le_act        = LabelEncoder()
        self.pipe1         = None
        self.pipe2         = None
        self.pipe3         = None
        self.base_features = None
        self.feat2         = None
        self.feat3         = None
        self.rmse3         = None

    def _feature_cols(self, df: pd.DataFrame) -> list:
        return [c for c in df.columns if c not in _TARGETS]

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> "EvalaneSimPipeline":
        # ── Encode targets ────────────────────────────────────
        cong_cat = pd.Categorical(df[TARGET_CL],
                                  categories=CONGESTION_ORDER, ordered=True)
        y_cong = cong_cat.codes.astype(int)
        y_act  = self.le_act.fit_transform(df[TARGET_ACT])
        y_dur  = df[TARGET_DUR].values.astype(float)

        self.base_features = self._feature_cols(df)
        self.feat2 = self.base_features + ["__cong_enc__"]
        self.feat3 = self.base_features + ["__cong_enc__", "__act_enc__"]

        X_base = df[self.base_features]

        idx = np.arange(len(df))
        tr, te = train_test_split(idx, test_size=0.2,
                                  random_state=42, stratify=y_cong)

        # ══════════════════════════════════════════════════════
        # MODEL 1 — Congestion_Level
        # ══════════════════════════════════════════════════════
        if verbose:
            print("\n" + "═" * 60)
            print("  Model 1 — Congestion_Level  (Ordinal Classifier)")
            print("═" * 60)

        self.pipe1 = Pipeline([
            ("pre", build_preprocessor(self.base_features)),
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85,
                eval_metric="mlogloss", random_state=42, n_jobs=-1,
            )),
        ])
        self.pipe1.fit(X_base.iloc[tr], y_cong[tr])

        p1   = self.pipe1.predict(X_base.iloc[te])
        if verbose:
            print(classification_report(
                [CONGESTION_ORDER[v] for v in y_cong[te]],
                [CONGESTION_ORDER[v] for v in p1],
                target_names=CONGESTION_ORDER
            ))

        cong_tr = self.pipe1.predict(X_base.iloc[tr])
        cong_te = self.pipe1.predict(X_base.iloc[te])

        def _aug2(base_df, cp):
            a = base_df.copy().reset_index(drop=True)
            a["__cong_enc__"] = cp
            return a[self.feat2]

        X2_tr = _aug2(X_base.iloc[tr], cong_tr)
        X2_te = _aug2(X_base.iloc[te], cong_te)

        # ══════════════════════════════════════════════════════
        # MODEL 2 — Optimal_Action
        # ══════════════════════════════════════════════════════
        if verbose:
            print("═" * 60)
            print("  Model 2 — Optimal_Action  (Multi-class Classifier)")
            print("═" * 60)

        self.pipe2 = Pipeline([
            ("pre", build_preprocessor(self.feat2)),
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85,
                eval_metric="mlogloss", random_state=42, n_jobs=-1,
            )),
        ])
        self.pipe2.fit(X2_tr, y_act[tr])

        p2 = self.pipe2.predict(X2_te)
        if verbose:
            print(classification_report(
                self.le_act.inverse_transform(y_act[te]),
                self.le_act.inverse_transform(p2),
                target_names=self.le_act.classes_
            ))

        act_tr = self.pipe2.predict(X2_tr)
        act_te = self.pipe2.predict(X2_te)

        def _aug3(aug2_df, ap):
            a = aug2_df.copy().reset_index(drop=True)
            a["__act_enc__"] = ap
            return a[self.feat3]

        X3_tr = _aug3(X2_tr, act_tr)
        X3_te = _aug3(X2_te, act_te)

        # ══════════════════════════════════════════════════════
        # MODEL 3 — Signal_Duration
        # ══════════════════════════════════════════════════════
        if verbose:
            print("═" * 60)
            print("  Model 3 — Signal_Duration  (Regression)")
            print("═" * 60)

        self.pipe3 = Pipeline([
            ("pre", build_preprocessor(self.feat3)),
            ("reg", XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.04,
                subsample=0.85, colsample_bytree=0.85,
                random_state=42, n_jobs=-1,
            )),
        ])
        self.pipe3.fit(X3_tr, y_dur[tr])

        p3         = np.clip(self.pipe3.predict(X3_te), DURATION_MIN, DURATION_MAX)
        self.rmse3 = float(mean_squared_error(y_dur[te], p3) ** 0.5)
        r2         = r2_score(y_dur[te], p3)

        if verbose:
            print(f"  RMSE : {self.rmse3:.3f} s")
            print(f"  R²   : {r2:.4f}")
            print("═" * 60)
            print("  ✓  All three models trained successfully.")
            print("═" * 60 + "\n")

        return self

    def predict(self, raw_obs: dict) -> dict:
        """
        Parameters — keys matching simulation data:
            Lane_N_Count, Lane_S_Count, Lane_E_Count, Lane_W_Count
            Congestion_N, Congestion_S, Congestion_E, Congestion_W
            Current_Green_Lane   (e.g. "N")
            Green_Timer_Remaining (e.g. 18)
            Event_Type           ("Normal" | "Ambulance" | "Breakdown")
            Event_Lane           ("N"|"S"|"E"|"W"|"None")

        Returns:
            congestion_level, congestion_confidence,
            optimal_action, action_confidence, action_probabilities,
            signal_duration, signal_duration_range
        """
        row = {}
        for col in self.base_features:
            if col == "Event_Type":
                row[col] = raw_obs.get(col, "Normal") or "Normal"
            elif col in ("Current_Green_Lane", "Event_Lane"):
                row[col] = raw_obs.get(col, "None") or "None"
            else:
                row[col] = raw_obs.get(col, 0)

        X_base = pd.DataFrame([row])[self.base_features]

        # Model 1 → Congestion_Level
        cong_enc         = int(self.pipe1.predict(X_base)[0])
        congestion_level = CONGESTION_ORDER[cong_enc]
        cong_proba       = self.pipe1.predict_proba(X_base)[0]
        cong_confidence  = round(float(cong_proba[cong_enc]), 3)

        # Model 2 → Optimal_Action
        X2 = X_base.copy()
        X2["__cong_enc__"] = cong_enc
        act_enc        = int(self.pipe2.predict(X2[self.feat2])[0])
        optimal_action = self.le_act.inverse_transform([act_enc])[0]
        act_proba      = self.pipe2.predict_proba(X2[self.feat2])[0]
        act_confidence = round(float(act_proba[act_enc]), 3)
        action_probabilities = {
            str(label): round(float(prob), 3)
            for label, prob in zip(self.le_act.classes_, act_proba)
        }

        # Model 3 → Signal_Duration
        X3 = X2.copy()
        X3["__act_enc__"] = act_enc
        raw_dur         = float(self.pipe3.predict(X3[self.feat3])[0])
        signal_duration = round(float(np.clip(raw_dur, DURATION_MIN, DURATION_MAX)), 1)
        rmse   = self.rmse3 or 0.0
        dur_lo = round(float(np.clip(raw_dur - rmse, DURATION_MIN, DURATION_MAX)), 1)
        dur_hi = round(float(np.clip(raw_dur + rmse, DURATION_MIN, DURATION_MAX)), 1)

        # ── Ambulance hard override ───────────────────────────
        # Ambulance always gets its lane — non-negotiable.
        # Duration is model-driven but floored at AMBULANCE_FLOOR.
        lane_to_action = {"N":"GREEN_N","S":"GREEN_S","E":"GREEN_E","W":"GREEN_W"}
        event_type = row.get("Event_Type", "Normal")
        event_lane = row.get("Event_Lane", "None")

        if event_type == "Ambulance" and event_lane in lane_to_action:
            forced_action  = lane_to_action[event_lane]
            forced_act_enc = int(self.le_act.transform([forced_action])[0])
            optimal_action = forced_action
            act_confidence = 1.0
            action_probabilities = {
                k: (1.0 if k == forced_action else 0.0)
                for k in action_probabilities
            }
            X3_f = X2.copy()
            X3_f["__act_enc__"] = forced_act_enc
            raw_dur_f       = float(self.pipe3.predict(X3_f[self.feat3])[0])
            signal_duration = round(float(np.clip(
                max(raw_dur_f, AMBULANCE_FLOOR), DURATION_MIN, DURATION_MAX)), 1)
            dur_lo = round(float(np.clip(signal_duration - rmse, DURATION_MIN, DURATION_MAX)), 1)
            dur_hi = round(float(np.clip(signal_duration + rmse, DURATION_MIN, DURATION_MAX)), 1)

        return {
            "congestion_level":      congestion_level,
            "congestion_confidence": cong_confidence,
            "optimal_action":        optimal_action,
            "action_confidence":     act_confidence,
            "action_probabilities":  action_probabilities,
            "signal_duration":       signal_duration,
            "signal_duration_range": [dur_lo, dur_hi],
        }

    def save(self, path: str = "evalane_sim_pipeline.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Pipeline saved → {path}")

    @classmethod
    def load(cls, path: str = "evalane_sim_pipeline.pkl") -> "EvalaneSimPipeline":
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────
# 4.  Standalone predict function
# ─────────────────────────────────────────────────────────────────
def predict_junction_state(raw_observation: dict,
                            pipeline_path: str = "evalane_sim_pipeline.pkl") -> dict:
    return EvalaneSimPipeline.load(pipeline_path).predict(raw_observation)


# ─────────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────────
def main():
    print(f"\nLoading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape:   {df.shape}")

    pipeline = EvalaneSimPipeline()
    pipeline.fit(df, verbose=True)
    pipeline.save("evalane_sim_pipeline.pkl")

    print("═" * 60)
    print("  DEMO 1 — Evening rush, West lane congested (normal)")
    print("═" * 60)
    sample1 = {
        "Lane_N_Count": 44, "Lane_S_Count": 22,
        "Lane_E_Count": 15, "Lane_W_Count": 47,
        "Congestion_N": 68.0, "Congestion_S": 36.0,
        "Congestion_E": 24.0, "Congestion_W": 74.0,
        "Current_Green_Lane": "S",
        "Green_Timer_Remaining": 8,
        "Event_Type": "Normal", "Event_Lane": "None",
    }

    print("═" * 60)
    print("  DEMO 2 — Ambulance in North lane")
    print("═" * 60)
    sample2 = {
        "Lane_N_Count": 21, "Lane_S_Count": 38,
        "Lane_E_Count": 29, "Lane_W_Count": 44,
        "Congestion_N": 31.0, "Congestion_S": 55.0,
        "Congestion_E": 42.0, "Congestion_W": 68.0,
        "Current_Green_Lane": "W",
        "Green_Timer_Remaining": 14,
        "Event_Type": "Ambulance", "Event_Lane": "N",
    }

    print("═" * 60)
    print("  DEMO 3 — Breakdown in East lane, South dominant")
    print("═" * 60)
    sample3 = {
        "Lane_N_Count": 24, "Lane_S_Count": 51,
        "Lane_E_Count": 9,  "Lane_W_Count": 31,
        "Congestion_N": 35.0, "Congestion_S": 78.0,
        "Congestion_E": 12.0, "Congestion_W": 45.0,
        "Current_Green_Lane": "N",
        "Green_Timer_Remaining": 5,
        "Event_Type": "Breakdown", "Event_Lane": "E",
    }

    for label, sample in [
        ("Evening rush — West congested", sample1),
        ("Ambulance in North lane",       sample2),
        ("Breakdown in East — South dom", sample3),
    ]:
        r = pipeline.predict(sample)
        print(f"\n  [{label}]")
        print(f"  ┌──────────────────────────────────────────────────────────────────┐")
        print(f"  │  Congestion Level   : {r['congestion_level']:<43}│")
        print(f"  │  Congestion Conf.   : {r['congestion_confidence']:<43}│")
        print(f"  │  Optimal Action     : {r['optimal_action']:<43}│")
        print(f"  │  Action Confidence  : {r['action_confidence']:<43}│")
        print(f"  │  Action Probs       : {str(r['action_probabilities']):<43}│")
        print(f"  │  Signal Duration    : {str(r['signal_duration']) + ' s':<43}│")
        print(f"  │  Duration Range     : {str(r['signal_duration_range']):<43}│")
        print(f"  └──────────────────────────────────────────────────────────────────┘")

    return pipeline

if __name__ == "__main__":
    main()
