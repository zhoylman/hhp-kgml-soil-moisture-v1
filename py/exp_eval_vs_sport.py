#!/usr/bin/env python
"""
Network-value experiment EVAL: held-out SCAN 2005-2015, per-site KGE/r/|%bias|
for the fine-tuned KGML model vs the SPoRT-LIS baseline. Station point readout
(run model on the window CSVs) -- no gridded inference.

DEPTH/FROZEN CONVENTION (matches R/6_1-build-kfold-dataset.R + 1_1):
  KGML  : model prediction vs generalized-NO-FROZEN obs   (Middle = 10-50 cm)
  SPoRT : SPoRT_raw_10-40cm sim vs sport-specific-NO-FROZEN obs (Middle = 10-40 cm)
Each model is scored against the obs aggregation matched to its native depth;
both exclude frozen-soil days. Metrics: KGE (Gupta 2009), Pearson r,
pbias = 100*sum(sim-obs)/sum(obs)  -- table reports |% bias|.

Run: CUDA_DEVICE_ORDER=PCI_BUS_ID DEPTH_FLAG=middle CUDA_VISIBLE_DEVICES=0 python py/exp_eval_vs_sport.py
"""
import sys, os, re
from collections import defaultdict
import numpy as np, pandas as pd, torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_config import *
from define_hyperparams import *      # max_channels, dropout, num_classes

DEPTH       = os.environ.get("DEPTH_FLAG", "middle")
TAG         = os.environ.get("EXP_TAG", "")
BASE        = "/data/ssd2/soil-moisture-ml"
SPLIT       = f"{BASE}/split-definitions-exp-{DEPTH}{TAG}/eval_scan_pre2010.csv"
MODEL       = f"{BASE}/results-exp-finetune-{DEPTH}{TAG}/model.pt"
OUT         = f"{BASE}/results-exp-finetune-{DEPTH}{TAG}/eval_scan_vs_sport.csv"
KGML_OBS    = f"{BASE}/observations/final-soil-moisture-data-generalized-no-frozen.csv"            # 10-50 Middle
SPORT_OBS   = f"{BASE}/observations/final-soil-moisture-data-generalized-sport-specific-no-frozen.csv"  # 10-40 Middle
SPORT_SIM   = f"{BASE}/observations/observational-sites-raw-sport.csv"
DEPTH_LABEL = {"middle": "Middle", "shallow": "Shallow"}[DEPTH]
SPORT_VAR   = {"middle": "SPoRT_raw_10-40cm", "shallow": "SPoRT_raw_0-10cm"}[DEPTH]
YEAR_CH, YEAR_FROZEN = 9, (2013 - 1979) / (2023 - 1979)
MIN_N = 10
SITE  = re.compile(r"basin-group-([^_]+)_")

def per_site_metrics(df, simcol, obscol, tag):
    """df: site_id, <simcol>, <obscol> -> per-site KGE/r/pbias."""
    rows = []
    for site, g in df.groupby("site_id"):
        s = g[simcol].to_numpy(float); o = g[obscol].to_numpy(float)
        m = np.isfinite(s) & np.isfinite(o)
        if m.sum() < MIN_N:
            continue
        s, o = s[m], o[m]
        if s.std() == 0 or o.std() == 0 or o.mean() == 0:
            continue
        r = float(np.corrcoef(s, o)[0, 1])
        kge = 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)
        pbias = 100 * (s.sum() - o.sum()) / o.sum()
        rows.append((site, int(m.sum()), float(kge), r, float(pbias)))
    return pd.DataFrame(rows, columns=["site_id", f"n_{tag}", f"KGE_{tag}", f"r_{tag}", f"pbias_{tag}"])

def main():
    # ---- 1. run model on eval windows -> per (site,date) mean prediction ----
    tmpl = StreamflowTrainDatasetTemplate(csv_file=get_first_file_path(f"{BASE}/full-dataloader-{DEPTH}"))
    input_size = tmpl.x_tensor.shape[2]
    model = uNet(num_classes, input_size, max_channels, dropout).to('cuda')
    model.load_state_dict(torch.load(MODEL, map_location='cuda')); model.eval()
    ev = pd.read_csv(SPLIT)
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))   # site -> date -> [sum, n]
    with torch.no_grad():
        for path in ev['path']:
            d = pd.read_csv(path)
            site = SITE.search(os.path.basename(path)).group(1)
            X = torch.tensor(d.drop(['time', 'soil_moisture'], axis=1).values).float().unsqueeze(0).to('cuda')
            X[:, :, YEAR_CH] = YEAR_FROZEN
            yh = model(X).cpu().view(-1).numpy()
            for t, p in zip(d['time'].astype(str).str.slice(0, 10), yh):
                c = acc[site][t]; c[0] += float(p); c[1] += 1
    ml = pd.DataFrame([(s, t, v[0] / v[1]) for s, dd in acc.items() for t, v in dd.items()],
                      columns=["site_id", "date", "ml"])
    sites = set(ml.site_id.unique())
    EVAL_LO, EVAL_HI = ml.date.min(), ml.date.max()   # eval window = actual prediction-date span
    print(f"[eval] {DEPTH}: model ran {len(ev)} windows over {len(sites)} SCAN sites, "
          f"dates {EVAL_LO}..{EVAL_HI} (year frozen ch{YEAR_CH}={YEAR_FROZEN:.4f})")

    # ---- 2. KGML: prediction vs generalized-no-frozen obs (10-50) ----
    ko = pd.read_csv(KGML_OBS, low_memory=False)
    ko = ko[(ko.generalized_depth == DEPTH_LABEL)].copy()
    ko["site_id"] = ko.site_id.astype(str).str.strip()
    ko["date"] = ko.date.astype(str).str.slice(0, 10)
    ko = ko[ko.site_id.isin(sites)][["site_id", "date", "soil_moisture"]].rename(columns={"soil_moisture": "obs"})
    kgml = ml.merge(ko, on=["site_id", "date"], how="inner")
    kgml_m = per_site_metrics(kgml, "ml", "obs", "KGML")

    # ---- 3. SPoRT: 10-40 sim vs sport-specific-no-frozen obs (10-40) ----
    so = pd.read_csv(SPORT_OBS, low_memory=False)
    so = so[so.generalized_depth == DEPTH_LABEL].copy()
    so["site_id"] = so.site_id.astype(str).str.strip()
    so["date"] = so.date.astype(str).str.slice(0, 10)
    so = so[so.site_id.isin(sites) & (so.date >= EVAL_LO) & (so.date <= EVAL_HI)][["site_id", "date", "soil_moisture"]].rename(columns={"soil_moisture": "obs"})
    ss = pd.read_csv(SPORT_SIM, low_memory=False)
    ss = ss[ss["var"] == SPORT_VAR].drop(columns=["var"])
    ss["date"] = pd.to_datetime(ss["time"], errors="coerce").dt.strftime("%Y-%m-%d")
    keep = [c for c in ss.columns if c in sites]
    ss = ss[["date"] + keep].melt(id_vars="date", var_name="site_id", value_name="sport")
    sport = ss.merge(so, on=["site_id", "date"], how="inner")
    sport_m = per_site_metrics(sport, "sport", "obs", "SPoRT")

    # ---- 4. combine + report ----
    R = kgml_m.merge(sport_m, on="site_id", how="outer").sort_values("site_id")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    R.to_csv(OUT, index=False)

    v = R.dropna(subset=["KGE_KGML", "KGE_SPoRT"])
    med = lambda c: float(np.nanmedian(v[c]))
    imp = lambda a, b: (a - b) / abs(b) * 100
    mk_kge, sk_kge = med("KGE_KGML"), med("KGE_SPoRT")
    mk_r,   sk_r   = med("r_KGML"),   med("r_SPoRT")
    mk_b,   sk_b   = float(np.nanmedian(v.pbias_KGML.abs())), float(np.nanmedian(v.pbias_SPoRT.abs()))
    print(f"\n[eval] {DEPTH.upper()}  held-out SCAN {EVAL_LO[:4]}-{EVAL_HI[:4]}  |  {len(v)} paired sites")
    kgml_depth = {"middle": "10-50", "shallow": "0-10"}[DEPTH]
    print(f"  KGML obs=generalized-no-frozen ({DEPTH_LABEL} {kgml_depth}) ; SPoRT={SPORT_VAR} vs sport-specific-no-frozen")
    print(f"  {'':10s}{'KGML':>8s}{'SPoRT':>9s}{'%Imp':>8s}")
    print(f"  {'KGE':10s}{mk_kge:8.2f}{sk_kge:9.2f}{imp(mk_kge,sk_kge):+7.0f}%")
    print(f"  {'Pearson r':10s}{mk_r:8.2f}{sk_r:9.2f}{imp(mk_r,sk_r):+7.0f}%")
    print(f"  {'|% Bias|':10s}{mk_b:8.1f}{sk_b:9.1f}{imp(mk_b,sk_b):+7.0f}%")
    print(f"  model > SPoRT KGE at {(v.KGE_KGML > v.KGE_SPoRT).mean()*100:.0f}% of sites; saved -> {OUT}")

if __name__ == "__main__":
    main()
