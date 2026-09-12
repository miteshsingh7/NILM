# ─── CELL: Threshold Sweep on Transfer Checkpoint (with fridge event audit) ──
# Paste this cell into the Kaggle notebook after the evaluation cells.

import sys, os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, "/kaggle/working/nilm")
os.chdir("/kaggle/working/nilm")

from src.config import NILMConfig
from src.data_pipeline import NILMDataset, NormalizationParams
from src.model import MultiApplianceNILM
from src.utils import compute_nde, compute_mae

# ─── Config ──────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "/kaggle/working/checkpoints_transfer/fold_2/best_model.pt"
REDD_DIR = "/kaggle/working/nilm/data/raw/redd"
DATA_DIR = "/kaggle/working/nilm/data/processed"
HELD_OUT_HOUSE = 2
TARGET_APPS = ["fridge", "dishwasher"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128

# ─── Threshold grid: 0.02..0.08 step 0.02, 0.10..0.80 step 0.05 ─────────────
def build_threshold_grid():
    thresholds = []
    t = 0.02
    while t < 0.10 - 1e-9:
        thresholds.append(round(t, 4))
        t += 0.02
    t = 0.10
    while t <= 0.80 + 1e-9:
        thresholds.append(round(t, 4))
        t += 0.05
    return thresholds

def compute_fbeta(precision, recall, beta):
    if precision + recall < 1e-12:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall)

def compute_event_recall_raw(o_true_bin, o_pred_bin):
    """Raw event recall: NO minimum gap/duration merging.
    Every 0->1 transition in ground truth starts a new event."""
    true_flat = o_true_bin.flatten().astype(int)
    pred_flat = o_pred_bin.flatten().astype(int)
    n_events = 0
    n_detected = 0
    in_event = False
    event_detected = False
    for i in range(len(true_flat)):
        if true_flat[i] == 1:
            if not in_event:
                in_event = True
                event_detected = False
                n_events += 1
            if pred_flat[i] == 1:
                event_detected = True
        else:
            if in_event:
                if event_detected:
                    n_detected += 1
                in_event = False
    if in_event and event_detected:
        n_detected += 1
    if n_events == 0:
        return float("nan"), 0, 0
    return n_detected / n_events, n_detected, n_events

def compute_event_recall_merged(o_true_bin, o_pred_bin, min_gap_samples=10, min_dur_samples=5):
    """Merged event recall: bridges short gaps and drops short spikes.
    
    1. Bridge gaps <= min_gap_samples (default 10 = 60s at 6s sampling)
       in ground truth before segmenting events.
    2. Drop events shorter than min_dur_samples (default 5 = 30s at 6s).
    3. Then count how many merged events overlap with at least one predicted ON.
    """
    true_flat = o_true_bin.flatten().astype(int).copy()
    pred_flat = o_pred_bin.flatten().astype(int)
    
    # Step 1: Bridge short gaps in ground truth
    in_gap = False
    gap_start = 0
    for i in range(len(true_flat)):
        if true_flat[i] == 0:
            if not in_gap:
                in_gap = True
                gap_start = i
        else:
            if in_gap:
                gap_len = i - gap_start
                if gap_len <= min_gap_samples:
                    true_flat[gap_start:i] = 1  # bridge the gap
                in_gap = False
    
    # Step 2: Segment events and filter by minimum duration
    events = []  # list of (start, end) indices
    in_event = False
    evt_start = 0
    for i in range(len(true_flat)):
        if true_flat[i] == 1:
            if not in_event:
                in_event = True
                evt_start = i
        else:
            if in_event:
                evt_end = i
                if (evt_end - evt_start) >= min_dur_samples:
                    events.append((evt_start, evt_end))
                in_event = False
    if in_event:
        evt_end = len(true_flat)
        if (evt_end - evt_start) >= min_dur_samples:
            events.append((evt_start, evt_end))
    
    # Step 3: Count detections
    n_events = len(events)
    n_detected = 0
    for (s, e) in events:
        if np.any(pred_flat[s:e] == 1):
            n_detected += 1
    
    if n_events == 0:
        return float("nan"), 0, 0
    return n_detected / n_events, n_detected, n_events

def fridge_event_audit(o_true_bin, step_seconds=6):
    """Detailed audit of fridge ground-truth event structure."""
    true_flat = o_true_bin.flatten().astype(int)
    
    # Raw events (no merging)
    raw_events = []
    in_event = False
    evt_start = 0
    for i in range(len(true_flat)):
        if true_flat[i] == 1:
            if not in_event:
                in_event = True
                evt_start = i
        else:
            if in_event:
                raw_events.append(i - evt_start)
                in_event = False
    if in_event:
        raw_events.append(len(true_flat) - evt_start)
    
    raw_events = np.array(raw_events)
    raw_durations_s = raw_events * step_seconds
    
    # Raw gaps
    raw_gaps = []
    in_gap = False
    gap_start = 0
    for i in range(len(true_flat)):
        if true_flat[i] == 0:
            if not in_gap:
                in_gap = True
                gap_start = i
        else:
            if in_gap:
                raw_gaps.append(i - gap_start)
                in_gap = False
    raw_gaps = np.array(raw_gaps) if raw_gaps else np.array([0])
    raw_gap_durations_s = raw_gaps * step_seconds
    
    print(f"\n{'='*80}")
    print(f"FRIDGE EVENT SEGMENTATION AUDIT")
    print(f"{'='*80}")
    print(f"  ⚠ FLAG: Current event-recall logic has NO minimum-gap merging or")
    print(f"    minimum-duration filtering. Every single 0→1 transition in the")
    print(f"    ground-truth on/off label counts as a new event.")
    print(f"")
    print(f"  Raw (unmerged) event count: {len(raw_events)}")
    total_seconds = len(true_flat) * step_seconds
    total_days = total_seconds / 86400
    print(f"  Total observation: {len(true_flat)} samples = {total_days:.1f} days")
    print(f"  Events/day (raw): {len(raw_events) / total_days:.1f}")
    print(f"")
    print(f"  --- Raw event duration distribution ---")
    print(f"    Min:    {raw_durations_s.min():>8.0f}s ({raw_durations_s.min()/60:.1f} min)")
    print(f"    P5:     {np.percentile(raw_durations_s, 5):>8.0f}s ({np.percentile(raw_durations_s, 5)/60:.1f} min)")
    print(f"    P25:    {np.percentile(raw_durations_s, 25):>8.0f}s ({np.percentile(raw_durations_s, 25)/60:.1f} min)")
    print(f"    Median: {np.median(raw_durations_s):>8.0f}s ({np.median(raw_durations_s)/60:.1f} min)")
    print(f"    P75:    {np.percentile(raw_durations_s, 75):>8.0f}s ({np.percentile(raw_durations_s, 75)/60:.1f} min)")
    print(f"    P95:    {np.percentile(raw_durations_s, 95):>8.0f}s ({np.percentile(raw_durations_s, 95)/60:.1f} min)")
    print(f"    Max:    {raw_durations_s.max():>8.0f}s ({raw_durations_s.max()/60:.1f} min)")
    print(f"    Mean:   {raw_durations_s.mean():>8.1f}s ({raw_durations_s.mean()/60:.1f} min)")
    print(f"")
    print(f"  --- Raw gap (OFF period) duration distribution ---")
    print(f"    Min:    {raw_gap_durations_s.min():>8.0f}s ({raw_gap_durations_s.min()/60:.1f} min)")
    print(f"    P5:     {np.percentile(raw_gap_durations_s, 5):>8.0f}s ({np.percentile(raw_gap_durations_s, 5)/60:.1f} min)")
    print(f"    P25:    {np.percentile(raw_gap_durations_s, 25):>8.0f}s ({np.percentile(raw_gap_durations_s, 25)/60:.1f} min)")
    print(f"    Median: {np.median(raw_gap_durations_s):>8.0f}s ({np.median(raw_gap_durations_s)/60:.1f} min)")
    print(f"    P75:    {np.percentile(raw_gap_durations_s, 75):>8.0f}s ({np.percentile(raw_gap_durations_s, 75)/60:.1f} min)")
    print(f"    Max:    {raw_gap_durations_s.max():>8.0f}s ({raw_gap_durations_s.max()/60:.1f} min)")
    print(f"")
    
    # Counts by duration bucket
    n_under_6s = int(np.sum(raw_durations_s <= 6))
    n_under_30s = int(np.sum(raw_durations_s <= 30))
    n_under_60s = int(np.sum(raw_durations_s <= 60))
    n_under_5m = int(np.sum(raw_durations_s <= 300))
    n_over_5m = int(np.sum(raw_durations_s > 300))
    print(f"  --- Event count by duration bucket ---")
    print(f"    ≤6s (1 sample):  {n_under_6s:>6d}  ({100*n_under_6s/len(raw_events):.1f}%)")
    print(f"    ≤30s (≤5 samp):  {n_under_30s:>6d}  ({100*n_under_30s/len(raw_events):.1f}%)")
    print(f"    ≤60s (≤10 samp): {n_under_60s:>6d}  ({100*n_under_60s/len(raw_events):.1f}%)")
    print(f"    ≤5min:           {n_under_5m:>6d}  ({100*n_under_5m/len(raw_events):.1f}%)")
    print(f"    >5min:           {n_over_5m:>6d}  ({100*n_over_5m/len(raw_events):.1f}%)")
    print(f"")
    
    # Gap counts
    n_gap_under_6s = int(np.sum(raw_gap_durations_s <= 6))
    n_gap_under_30s = int(np.sum(raw_gap_durations_s <= 30))
    n_gap_under_60s = int(np.sum(raw_gap_durations_s <= 60))
    print(f"  --- Gap count by duration bucket ---")
    print(f"    ≤6s (1 sample):  {n_gap_under_6s:>6d}  ({100*n_gap_under_6s/len(raw_gaps):.1f}%)")
    print(f"    ≤30s (≤5 samp):  {n_gap_under_30s:>6d}  ({100*n_gap_under_30s/len(raw_gaps):.1f}%)")
    print(f"    ≤60s (≤10 samp): {n_gap_under_60s:>6d}  ({100*n_gap_under_60s/len(raw_gaps):.1f}%)")
    
    # Merged count at different settings
    print(f"\n  --- Merged event counts (bridging short gaps, dropping short spikes) ---")
    for min_gap, min_dur, label in [
        (5, 3, "gap≤30s, dur≥18s"),
        (10, 5, "gap≤60s, dur≥30s"),
        (20, 10, "gap≤120s, dur≥60s"),
        (50, 50, "gap≤5min, dur≥5min"),
        (100, 100, "gap≤10min, dur≥10min"),
    ]:
        merged = true_flat.copy()
        # Bridge gaps
        in_g = False
        gs = 0
        for i in range(len(merged)):
            if merged[i] == 0:
                if not in_g:
                    in_g = True
                    gs = i
            else:
                if in_g:
                    if (i - gs) <= min_gap:
                        merged[gs:i] = 1
                    in_g = False
        # Count events with min duration
        count = 0
        in_e = False
        es = 0
        for i in range(len(merged)):
            if merged[i] == 1:
                if not in_e:
                    in_e = True
                    es = i
            else:
                if in_e:
                    if (i - es) >= min_dur:
                        count += 1
                    in_e = False
        if in_e and (len(merged) - es) >= min_dur:
            count += 1
        print(f"    {label:>30s}: {count:>6d} events ({count/total_days:.1f}/day)")
    
    print(f"{'='*80}")

# ─── Load checkpoint ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"THRESHOLD SWEEP — TRANSFER CHECKPOINT")
print(f"  {CHECKPOINT_PATH}")
print(f"{'='*80}")

ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
appliances = ckpt.get("appliances", ["fridge", "microwave", "dishwasher", "washing_machine"])
print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}")
print(f"Checkpoint val_loss: {ckpt.get('val_loss', '?')}")
print(f"Appliances: {appliances}")
print(f"Target sweep: {TARGET_APPS}")
print(f"Held-out house: {HELD_OUT_HOUSE}")

norm_dict = ckpt["norm_params"]
norm_params = NormalizationParams(
    mains_mean=norm_dict["mains_mean"],
    mains_std=norm_dict["mains_std"],
    appliance_stats=norm_dict["appliance_stats"],
)
print(f"Norm params: mains mean={norm_params.mains_mean:.2f} W, std={norm_params.mains_std:.2f} W")
for app_name in appliances:
    s = norm_params.appliance_stats.get(app_name, {})
    print(f"  {app_name:15s}: active_mean={s.get('active_mean',0):.2f} W, "
          f"active_std={s.get('active_std',1):.2f} W, threshold={s.get('threshold',20):.1f} W")

# ─── Load model ──────────────────────────────────────────────────────────────
model = MultiApplianceNILM(appliances=appliances)
state_dict = ckpt["model_state_dict"]
clean_sd = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
model.load_state_dict(clean_sd)
model.to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}")

# ─── Prepare test dataset ────────────────────────────────────────────────────
config = NILMConfig(appliances=appliances, device=DEVICE, held_out_house=HELD_OUT_HOUSE)
from src.train import prepare_datasets
_, _, test_ds, _ = prepare_datasets(config, data_dir=DATA_DIR, redd_dir=REDD_DIR)
print(f"Test dataset: {len(test_ds)} windows from House {HELD_OUT_HOUSE}")

# ─── Inference ────────────────────────────────────────────────────────────────
print("\nRunning inference on test set...")
loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

p_pred_list, p_true_list, o_pred_list, o_true_list = [], [], [], []
with torch.no_grad():
    for batch in loader:
        x_b = batch[0].to(DEVICE)
        preds = model(x_b)
        p_pred_list.append(preds["power"].cpu().numpy())
        p_true_list.append(batch[1].numpy())
        o_pred_list.append(preds["on_off"].cpu().numpy())
        o_true_list.append(batch[2].numpy())

p_pred = np.concatenate(p_pred_list, axis=0)
p_true = np.concatenate(p_true_list, axis=0)
o_pred = np.concatenate(o_pred_list, axis=0)
o_true = np.concatenate(o_true_list, axis=0)
print(f"Inference complete: {p_pred.shape[0]} windows, {p_pred.shape[1]} timesteps/window")

# ─── Sweep ────────────────────────────────────────────────────────────────────
thresholds = build_threshold_grid()
print(f"Threshold grid ({len(thresholds)} values): {thresholds}")

for app in TARGET_APPS:
    if app not in appliances:
        print(f"\n⚠ '{app}' not in checkpoint — skipping")
        continue
    
    i = appliances.index(app)
    app_stats = norm_params.appliance_stats.get(app, {})
    app_threshold_watts = app_stats.get("threshold", 20.0)

    p_pred_watts = norm_params.denormalize_appliance(p_pred[..., i], app)
    p_true_watts = norm_params.denormalize_appliance(p_true[..., i], app)
    o_pred_prob = o_pred[..., i]
    o_true_bin = o_true[..., i]

    active_count = int(np.sum(p_true_watts >= app_threshold_watts))

    # ─── Fridge event audit ───────────────────────────────────────────
    if app == "fridge":
        fridge_event_audit(o_true_bin, step_seconds=6)

    _, _, n_events_raw = compute_event_recall_raw(o_true_bin, np.ones_like(o_true_bin))
    
    # For fridge, also compute merged event counts
    if app == "fridge":
        _, _, n_events_merged = compute_event_recall_merged(
            o_true_bin, np.ones_like(o_true_bin), min_gap_samples=10, min_dur_samples=5)
        merged_label = f"(raw: {n_events_raw}, merged[gap≤60s,dur≥30s]: {n_events_merged})"
    else:
        merged_label = f"(raw: {n_events_raw})"

    print(f"\n{'='*80}")
    print(f"APPLIANCE: {app.upper()}")
    print(f"  Power threshold: {app_threshold_watts} W")
    print(f"  Active samples (>= threshold): {active_count}")
    print(f"  Sigmoid stats: min={o_pred_prob.min():.4f}, max={o_pred_prob.max():.4f}, "
          f"mean={o_pred_prob.mean():.4f}, median={np.median(o_pred_prob):.4f}")
    print(f"  GT on-fraction: {o_true_bin.mean():.4f}")
    print(f"  GT contiguous ON events {merged_label}")
    print(f"{'='*80}")

    # Header: raw event recall + merged event recall for fridge
    if app == "fridge":
        print(f"\n{'Thresh':>6s} | {'Prec':>7s} {'Recall':>7s} {'F1':>7s} "
              f"{'F2':>7s} {'F3':>7s} | {'RawEvtRec':>9s} {'det':>5s}/{'tot':<5s} "
              f"{'MrgEvtRec':>9s} {'det':>5s}/{'tot':<5s} "
              f"| {'NDE':>7s} {'MAE(W)':>7s} | {'TP':>8s} {'FP':>8s} {'FN':>8s}")
        print("-" * 140)
    else:
        print(f"\n{'Thresh':>6s} | {'Prec':>7s} {'Recall':>7s} {'F1':>7s} "
              f"{'F2':>7s} {'F3':>7s} | {'EvtRecall':>9s} {'det':>5s}/{'tot':<5s} "
              f"| {'NDE':>7s} {'MAE(W)':>7s} | {'TP':>8s} {'FP':>8s} {'FN':>8s}")
        print("-" * 115)

    best_f1_r = {"f1": -1}
    best_f2_r = {"f2": -1}
    best_f3_r = {"f3": -1}
    best_nde_r = {"nde": 1e9}

    o_true_flat = (o_true_bin.flatten() >= 0.5).astype(int)
    o_pred_flat = o_pred_prob.flatten()
    p_true_flat = p_true_watts.flatten()
    p_pred_flat = p_pred_watts.flatten()

    for thr in thresholds:
        pred_bin = (o_pred_flat >= thr).astype(int)
        tp = int(np.sum((pred_bin == 1) & (o_true_flat == 1)))
        fp = int(np.sum((pred_bin == 1) & (o_true_flat == 0)))
        fn = int(np.sum((pred_bin == 0) & (o_true_flat == 1)))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = compute_fbeta(prec, rec, 1.0)
        f2 = compute_fbeta(prec, rec, 2.0)
        f3 = compute_fbeta(prec, rec, 3.0)
        
        # Raw event recall
        raw_er, raw_nd, raw_ne = compute_event_recall_raw(
            o_true_bin, pred_bin.reshape(o_pred_prob.shape))
        
        # NDE with hard-gated power
        p_gated = p_pred_flat * pred_bin
        nde = compute_nde(p_true_flat, p_gated)
        mae = float(np.mean(np.abs(p_true_flat - p_gated)))
        
        raw_er_s = f"{raw_er:.4f}" if not np.isnan(raw_er) else "  NaN  "
        
        if app == "fridge":
            # Also compute merged event recall
            mrg_er, mrg_nd, mrg_ne = compute_event_recall_merged(
                o_true_bin, pred_bin.reshape(o_pred_prob.shape),
                min_gap_samples=10, min_dur_samples=5)
            mrg_er_s = f"{mrg_er:.4f}" if not np.isnan(mrg_er) else "  NaN  "
            print(f"  {thr:>4.2f} | {prec:>7.4f} {rec:>7.4f} {f1:>7.4f} "
                  f"{f2:>7.4f} {f3:>7.4f} | {raw_er_s:>9s} {raw_nd:>5d}/{raw_ne:<5d} "
                  f"{mrg_er_s:>9s} {mrg_nd:>5d}/{mrg_ne:<5d} "
                  f"| {nde:>7.4f} {mae:>7.2f} | {tp:>8d} {fp:>8d} {fn:>8d}")
        else:
            print(f"  {thr:>4.2f} | {prec:>7.4f} {rec:>7.4f} {f1:>7.4f} "
                  f"{f2:>7.4f} {f3:>7.4f} | {raw_er_s:>9s} {raw_nd:>5d}/{raw_ne:<5d} "
                  f"| {nde:>7.4f} {mae:>7.2f} | {tp:>8d} {fp:>8d} {fn:>8d}")
        
        r = {"threshold": thr, "precision": prec, "recall": rec, 
             "f1": f1, "f2": f2, "f3": f3, "nde": nde}
        if f1 > best_f1_r["f1"]: best_f1_r = r
        if f2 > best_f2_r["f2"]: best_f2_r = r
        if f3 > best_f3_r["f3"]: best_f3_r = r
        if nde < best_nde_r["nde"]: best_nde_r = r

    print(f"\n--- {app.upper()} OPTIMAL THRESHOLDS ---")
    print(f"  Best F1 = {best_f1_r['f1']:.4f} at threshold = {best_f1_r['threshold']:.2f} "
          f"(P={best_f1_r['precision']:.4f}, R={best_f1_r['recall']:.4f})")
    print(f"  Best F2 = {best_f2_r['f2']:.4f} at threshold = {best_f2_r['threshold']:.2f} "
          f"(P={best_f2_r['precision']:.4f}, R={best_f2_r['recall']:.4f})")
    print(f"  Best F3 = {best_f3_r['f3']:.4f} at threshold = {best_f3_r['threshold']:.2f} "
          f"(P={best_f3_r['precision']:.4f}, R={best_f3_r['recall']:.4f})")
    print(f"  Best NDE = {best_nde_r['nde']:.4f} at threshold = {best_nde_r['threshold']:.2f}")

    if app == "dishwasher":
        print(f"\n  ⚠ NOTE: Optimal thresholds are selected on the same House 2 test")
        print(f"    slice — NOT a separate calibration split. Known limitation.")

print(f"\n{'='*80}")
print("THRESHOLD SWEEP COMPLETE")
print(f"{'='*80}")
