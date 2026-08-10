#!/usr/bin/env python
"""
Point-based center-keep evaluation of the nested network-size experiment
models against each fold's FIXED validation holdout. Reuses the exact
center-keep stitching machinery from exp_point_centerkeep_eval.py (already
validated against the gridded production pipeline, r=0.91 per-site agreement).

Run: python py/exp_networksize_eval.py
"""
import os, sys, glob
import numpy as np, pandas as pd, torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_config import *
from define_hyperparams import *
import exp_point_centerkeep_eval as ck   # reuse build_site_daily, normalize_site, make_windows, stitch_site, kge

BASE = "/data/ssd2/soil-moisture-ml"
REPO = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
OUT = f"{REPO}/cache/networksize-pointeval"
os.makedirs(OUT, exist_ok=True)
SIZES = [100, 200, 300, 400, 500, 600]
DEVICE = "cuda"

def eval_one(depth, fold, size, model_path, sites, obs, template_input_size):
    ck.DEPTH = depth   # ck.build_site_daily/normalize_site use module-level seq_mn etc already loaded for this depth at import
    model = uNet(num_classes, template_input_size, max_channels, dropout).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    rows = []
    for site in sites:
        d = ck.build_site_daily(site)
        if d is None or len(d) < 250:
            continue
        d = ck.normalize_site(d)
        windows = ck.make_windows(d)
        if not windows:
            continue
        ml = ck.stitch_site(windows, model, DEVICE)
        ml["date"] = pd.to_datetime(ml["date"])
        o = obs[obs.site_id == site][["date", "soil_moisture"]].rename(columns={"soil_moisture": "obs"})
        mt = ml.merge(o, on="date", how="inner").dropna()
        k, r, pb = ck.kge(mt["ml"].to_numpy(), mt["obs"].to_numpy())
        rows.append((depth, fold, size, site, len(mt), k, r, pb))
    del model; torch.cuda.empty_cache()
    return rows

def main():
    all_rows = []
    for depth in ["shallow", "middle"]:
        # reload the correct depth's min-max scalers into the ck module (set at import for DEPTH_FLAG env-default;
        # re-load explicitly here since we loop both depths in one process)
        suffix = "-shallow" if depth == "shallow" else ""
        ck.seq_mn, ck.seq_mx = ck.load_minmax(f"{BASE}/min-max-definitions/seq-min-max-definitions-pretrain{suffix}.csv")
        ck.stat_mn, ck.stat_mx = ck.load_minmax(f"{BASE}/min-max-definitions/static-min-max-definitions-pretrain{suffix}.csv")

        depth_label = {"shallow": "Shallow", "middle": "Middle"}[depth]
        obs = pd.read_csv(f"{BASE}/observations/final-soil-moisture-data-generalized-no-frozen.csv",
                          dtype={"site_id": str}, low_memory=False)
        obs = obs[obs.generalized_depth == depth_label][["site_id", "date", "soil_moisture"]]
        obs["date"] = pd.to_datetime(obs["date"])

        tmpl = ck.StreamflowTrainDatasetTemplate(csv_file=glob.glob(f"{BASE}/full-dataloader-{depth}/*.csv")[0])
        input_size = tmpl.x_tensor.shape[2]

        for fold in range(1, 11):
            val = pd.read_csv(f"{BASE}/split-definitions-kfold-{depth}/validation_split_fold_{fold}.csv", dtype={"site_id": str})
            sites = sorted(val.site_id.unique())
            for size in SIZES:
                mpath = f"{BASE}/results-networksize-{depth}/fold{fold}_size{size}/model.pt"
                if not os.path.exists(mpath):
                    print(f"[skip] missing {mpath}"); continue
                rows = eval_one(depth, fold, size, mpath, sites, obs, input_size)
                all_rows.extend(rows)
                ks = [r[5] for r in rows if np.isfinite(r[5])]
                print(f"[{depth} fold {fold} size {size}] n_sites={len(rows)} median KGE={np.median(ks) if ks else float('nan'):.3f}", flush=True)

    R = pd.DataFrame(all_rows, columns=["depth", "fold", "size", "site_id", "n_obs", "KGE", "r", "pbias"])
    R.to_csv(f"{OUT}/networksize_point_eval.csv", index=False)
    print(f"\nWrote {OUT}/networksize_point_eval.csv ({len(R)} rows)")

if __name__ == "__main__":
    main()
