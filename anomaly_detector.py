"""
SpaceXAI — Rocket Telemetry Anomaly Detector
=============================================
Simulates Falcon 9 Merlin engine sensor streams and detects anomalies
using Isolation Forest — an unsupervised ML algorithm ideal for
high-dimensional telemetry data where labeled failures are rare.

Sensors modeled:
  - Chamber pressure (psi)
  - LOX flow rate (kg/s)
  - RP-1 flow rate (kg/s)
  - Turbopump vibration (g-force)
  - Nozzle throat temperature (K)
  - Engine gimbal angle (deg)

Author: github.com/yourhandle
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. SENSOR CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class SensorSpec:
    name: str
    unit: str
    nominal: float
    noise: float
    min_val: float
    max_val: float


SENSORS = [
    SensorSpec("chamber_pressure",  "psi",   980.0,  12.0,  800.0, 1200.0),
    SensorSpec("lox_flow_rate",     "kg/s",  2.350,  0.030,  1.5,   3.5),
    SensorSpec("rp1_flow_rate",     "kg/s",  1.000,  0.015,  0.5,   1.8),
    SensorSpec("turbopump_vib",     "g",     1.200,  0.200,  0.0,   8.0),
    SensorSpec("nozzle_throat_temp","K",  1850.0,  25.0, 1500.0, 2400.0),
    SensorSpec("gimbal_angle",      "deg",   0.050,  0.050, -5.0,   5.0),
]

SENSOR_NAMES = [s.name for s in SENSORS]


# ─────────────────────────────────────────────
# 2. TELEMETRY SIMULATOR
# ─────────────────────────────────────────────

class TelemetrySimulator:
    """
    Generates realistic rocket engine telemetry with optional
    injected faults. Faults model real failure modes:
      - pressure_spike    → rapid combustion instability
      - lox_loss          → oxidizer feed system failure
      - turbopump_failure → bearing degradation / cavitation
      - thermal_runaway   → regenerative cooling failure
    """

    FAULT_MODES = {
        "pressure_spike":    {"chamber_pressure": +200,  "turbopump_vib": +1.5},
        "lox_loss":          {"lox_flow_rate":    -0.8,  "chamber_pressure": -120},
        "turbopump_failure": {"turbopump_vib":    +4.0,  "lox_flow_rate": -0.3},
        "thermal_runaway":   {"nozzle_throat_temp": +400, "chamber_pressure": +80},
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        n_samples: int = 500,
        fault_type: Optional[str] = None,
        fault_start: int = 350,
        fault_duration: int = 60,
    ) -> pd.DataFrame:
        """Generate n_samples rows of telemetry, optionally injecting a fault."""

        data = {}
        for spec in SENSORS:
            nominal = self.rng.normal(spec.nominal, spec.noise, n_samples)
            # add slow drift to simulate thermal soak
            drift = np.linspace(0, spec.noise * 0.5, n_samples)
            data[spec.name] = np.clip(nominal + drift, spec.min_val, spec.max_val)

        # inject fault window
        if fault_type and fault_type in self.FAULT_MODES:
            deltas = self.FAULT_MODES[fault_type]
            fault_end = min(fault_start + fault_duration, n_samples)
            ramp = np.linspace(0, 1, fault_end - fault_start)  # smooth onset
            for sensor, delta in deltas.items():
                data[sensor][fault_start:fault_end] += ramp * delta
                # clip to physical bounds
                spec = next(s for s in SENSORS if s.name == sensor)
                data[sensor] = np.clip(data[sensor], spec.min_val, spec.max_val)

        df = pd.DataFrame(data)
        df["timestamp"] = pd.date_range("2026-01-01", periods=n_samples, freq="100ms")
        df["fault"] = fault_type or "none"
        df["is_fault"] = False
        if fault_type:
            df.loc[fault_start : fault_start + fault_duration - 1, "is_fault"] = True
        return df


# ─────────────────────────────────────────────
# 3. ANOMALY DETECTION MODEL
# ─────────────────────────────────────────────

class TelemetryAnomalyDetector:
    """
    Isolation Forest wrapped in a sklearn Pipeline with StandardScaler.
    Trained on nominal data; anomaly score reflects deviation from
    learned nominal distribution — no labelled failures needed.
    """

    def __init__(self, contamination: float = 0.03, n_estimators: int = 200):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("isoforest", IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=42,
                n_jobs=-1,
            )),
        ])
        self.trained = False

    def fit(self, df: pd.DataFrame) -> "TelemetryAnomalyDetector":
        X = df[SENSOR_NAMES].values
        self.pipeline.fit(X)
        self.trained = True
        print(f"[Model] Trained on {len(df)} nominal samples · "
              f"{len(SENSOR_NAMES)} sensors · IsolationForest n={200}")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.trained, "Call .fit() before .predict()"
        X = df[SENSOR_NAMES].values
        # sklearn: -1 = anomaly, +1 = normal → convert to 0/1
        labels = self.pipeline.predict(X)
        # raw decision scores (lower = more anomalous)
        raw_scores = self.pipeline.decision_function(X)
        # normalize to [0,1] where 1 = most anomalous
        norm_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)

        result = df.copy()
        result["anomaly_label"] = (labels == -1).astype(int)
        result["anomaly_score"] = np.round(norm_scores, 4)
        return result


# ─────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────

def evaluate(result: pd.DataFrame) -> dict:
    if "is_fault" not in result.columns:
        return {}
    tp = ((result["anomaly_label"] == 1) & result["is_fault"]).sum()
    fp = ((result["anomaly_label"] == 1) & ~result["is_fault"]).sum()
    fn = ((result["anomaly_label"] == 0) & result["is_fault"]).sum()
    tn = ((result["anomaly_label"] == 0) & ~result["is_fault"]).sum()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return dict(TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
                precision=round(precision, 3), recall=round(recall, 3), f1=round(f1, 3))


# ─────────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────────

COLORS = {
    "nominal":  "#185FA5",
    "anomaly":  "#E24B4A",
    "fault_bg": "#FCEBEB",
    "score":    "#854F0B",
    "grid":     "#e8e8e8",
}

def plot_results(result: pd.DataFrame, fault_type: str, metrics: dict, save_path: str = None):
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    fig.suptitle(
        f"Rocket Telemetry Anomaly Detection  ·  Fault: {fault_type.replace('_', ' ').title()}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(4, 2, hspace=0.55, wspace=0.3)
    sensor_axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(len(SENSORS))]
    score_ax    = fig.add_subplot(gs[3, :])

    t = np.arange(len(result)) * 0.1  # seconds

    # fault window shading
    fault_mask = result["is_fault"].values
    def shade_fault(ax):
        if fault_mask.any():
            start = t[fault_mask][0]
            end   = t[fault_mask][-1]
            ax.axvspan(start, end, color=COLORS["fault_bg"], alpha=0.6, label="Injected fault")

    for ax, spec in zip(sensor_axes, SENSORS):
        vals = result[spec.name].values
        shade_fault(ax)
        ax.plot(t, vals, lw=1.0, color=COLORS["nominal"], label="Sensor reading")
        # mark detected anomalies
        anom_idx = result["anomaly_label"].values == 1
        ax.scatter(t[anom_idx], vals[anom_idx], color=COLORS["anomaly"],
                   s=14, zorder=5, label="Detected anomaly")
        ax.axhline(spec.nominal, color="#aaa", lw=0.7, ls="--")
        ax.set_title(f"{spec.name.replace('_', ' ').title()} ({spec.unit})",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(color=COLORS["grid"], lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    # anomaly score timeline
    shade_fault(score_ax)
    score_ax.fill_between(t, result["anomaly_score"], alpha=0.3, color=COLORS["score"])
    score_ax.plot(t, result["anomaly_score"], lw=1.2, color=COLORS["score"], label="Anomaly score")
    score_ax.axhline(0.5, color=COLORS["anomaly"], lw=1.0, ls="--", label="Decision threshold")
    score_ax.set_ylim(0, 1)
    score_ax.set_title("Composite Anomaly Score (Isolation Forest)", fontsize=9, fontweight="bold")
    score_ax.set_xlabel("Time (s)", fontsize=8)
    score_ax.set_ylabel("Score (0=nominal, 1=anomaly)", fontsize=8)
    score_ax.tick_params(labelsize=8)
    score_ax.grid(color=COLORS["grid"], lw=0.5)
    score_ax.spines[["top", "right"]].set_visible(False)
    score_ax.legend(fontsize=8, loc="upper left")

    # metrics box
    if metrics:
        txt = (f"Precision {metrics['precision']:.1%}  ·  "
               f"Recall {metrics['recall']:.1%}  ·  "
               f"F1 {metrics['f1']:.1%}  ·  "
               f"TP {metrics['TP']}  FP {metrics['FP']}  FN {metrics['FN']}")
        fig.text(0.5, 0.01, txt, ha="center", fontsize=9,
                 color="#333",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#EAF3DE", ec="#C0DD97", lw=0.8))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(fault_type: str = "turbopump_failure", save_plot: str = None):
    print("\n" + "="*55)
    print(" SpaceXAI Telemetry Anomaly Detector")
    print("="*55)

    sim   = TelemetrySimulator(seed=7)
    model = TelemetryAnomalyDetector(contamination=0.03)

    # train on clean nominal data
    print("\n[1/4] Generating nominal training data …")
    train_df = sim.generate(n_samples=1000)
    model.fit(train_df)

    # generate test data with injected fault
    print(f"\n[2/4] Simulating test flight with fault: {fault_type} …")
    test_df = sim.generate(n_samples=500, fault_type=fault_type,
                           fault_start=300, fault_duration=80)

    # detect
    print("\n[3/4] Running anomaly detection …")
    result = model.predict(test_df)

    # evaluate
    print("\n[4/4] Evaluation:")
    metrics = evaluate(result)
    for k, v in metrics.items():
        print(f"       {k:<12} {v}")

    # plot
    plot_results(result, fault_type, metrics, save_path=save_plot)
    return result, metrics


if __name__ == "__main__":
    import sys
    fault = sys.argv[1] if len(sys.argv) > 1 else "turbopump_failure"
    valid = list(TelemetrySimulator.FAULT_MODES.keys())
    if fault not in valid:
        print(f"Unknown fault. Choose from: {valid}")
        sys.exit(1)
    run_pipeline(fault_type=fault, save_plot=f"anomaly_{fault}.png")
