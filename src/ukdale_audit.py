"""Coverage and active-sample auditing script for UK-DALE dataset."""

from pathlib import Path
import pandas as pd

from src.config import NILMConfig
from src.data_ukdale import (
    build_ukdale_coverage_table,
    load_ukdale_house,
    resample_and_clean,
    print_ukdale_active_sample_audit,
)


def main():
    ukdale_dir = Path("data/raw/ukdale")
    data_processed = Path("data/processed")
    data_processed.mkdir(parents=True, exist_ok=True)

    cfg = NILMConfig()

    # 1. Print Coverage Table
    coverage_df = build_ukdale_coverage_table(ukdale_dir, appliances=cfg.appliances)
    print("\n================ UK-DALE APPLIANCE / HOUSE COVERAGE TABLE ================")
    print(coverage_df.to_string(index=False))
    print("==========================================================================\n")

    # 2. Load and cache cleaned houses
    house_dfs = {}
    for h in range(1, 6):
        h_dir = ukdale_dir / f"house_{h}"
        if not h_dir.exists():
            continue
        cached_parquet = data_processed / f"ukdale_real_house_{h}.parquet"
        cached_csv = data_processed / f"ukdale_real_house_{h}.csv"
        if cached_parquet.exists():
            df_clean = pd.read_parquet(cached_parquet)
        elif cached_csv.exists():
            df_clean = pd.read_csv(cached_csv, index_col=0, parse_dates=True)
        else:
            df_clean, presence = load_ukdale_house(
                h_dir,
                h,
                appliances=cfg.appliances,
                sample_period_seconds=cfg.sample_period_seconds,
                max_gap_fill_samples=cfg.max_gap_fill_samples,
            )
            df_clean.to_parquet(cached_parquet)
        house_dfs[h] = df_clean
        print(f"Loaded UK-DALE House {h}: {len(df_clean):,} samples @ 6s ({len(df_clean)*6/3600/24:.1f} days)")

    # 3. Overall active sample audit across all 5 houses
    print("\n================ UK-DALE ACTIVE SAMPLES PER HOUSE AUDIT ================")
    rows = []
    for app in cfg.appliances:
        thresh = cfg.get_threshold(app)
        row = {"Appliance": app, "Threshold": f"{thresh:.0f} W"}
        for h in range(1, 6):
            if h in house_dfs and app in house_dfs[h].columns:
                cnt = int((house_dfs[h][app] >= thresh).sum())
                pct = cnt / len(house_dfs[h]) * 100 if len(house_dfs[h]) > 0 else 0.0
                row[f"House {h}"] = f"{cnt:,} ({pct:.2f}%)"
            else:
                row[f"House {h}"] = "0 (0.00%) / Missing"
        rows.append(row)
    active_df = pd.DataFrame(rows)
    print(active_df.to_string(index=False))
    print("=========================================================================\n")


if __name__ == "__main__":
    main()
