#!/usr/bin/env python
"""
Build split-definition CSVs for the network-value fine-tune experiment.

Design (middle): fine-tune on everything-but-SCAN with windows fully in 2020+;
evaluate on SCAN with windows fully in 2005-2009 (SPoRT-comparable "pre-2010").
SCAN is held out of fine-tune entirely (spatial holdout) + a ~10yr temporal gap.

Output = three {path, site_id} CSVs the existing dataloader reads directly
(StreamflowTrainDatasetXY only uses the `path` column). No window files are
rewritten; the `year` freeze happens at load time in the training script.

Reusable for shallow: pass --depth shallow.
"""
import os, re, glob, argparse, random
import pandas as pd
from multiprocessing import Pool

BASE = "/data/ssd2/soil-moisture-ml"
META = f"{BASE}/observations/final-soil-moisture-data-generalized-meta.csv"
SITE_RE = re.compile(r"basin-group-([^_]+)_")

def file_daterange(path):
    """Return (site_id, dmin, dmax) as ISO-date strings; reads only the time col."""
    site = SITE_RE.search(os.path.basename(path))
    site = site.group(1) if site else None
    t = pd.read_csv(path, usecols=["time"], dtype=str)["time"]
    s = t.str.slice(0, 10)
    return path, site, s.min(), s.max()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", default="middle")
    ap.add_argument("--eval_network", default="SCAN")
    ap.add_argument("--ft_start", default="2020-01-01")   # fine-tune: windows fully within [ft_start, ft_end]
    ap.add_argument("--ft_end", default="9999-12-31")
    ap.add_argument("--eval_start", default="2005-01-01")  # eval: windows fully within [eval_start, eval_end]
    ap.add_argument("--eval_end", default="2009-12-31")
    ap.add_argument("--tag", default="")                   # output dir suffix (keep variants separate)
    ap.add_argument("--holdout", choices=["network", "random"], default="network")
    ap.add_argument("--holdout_frac", type=float, default=0.30)  # random mode: fraction of eval-capable sites held out
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=69)
    a = ap.parse_args()

    loader_dir = f"{BASE}/full-dataloader-{a.depth}"
    out_dir = f"{BASE}/split-definitions-exp-{a.depth}{a.tag}"
    os.makedirs(out_dir, exist_ok=True)

    meta = pd.read_csv(META, dtype={"site_id": str})
    net = dict(zip(meta["site_id"].astype(str).str.strip(), meta["network"]))

    files = sorted(glob.glob(f"{loader_dir}/*.csv"))
    print(f"[splits] scanning {len(files)} window files in {loader_dir}")
    with Pool(a.workers) as p:
        rows = p.map(file_daterange, files, chunksize=64)

    df = pd.DataFrame(rows, columns=["path", "site_id", "dmin", "dmax"])
    df["site_id"] = df["site_id"].astype(str).str.strip()
    df["network"] = df["site_id"].map(net)
    n_nonet = df["network"].isna().sum()
    if n_nonet:
        print(f"[splits] WARNING: {n_nonet} files had no network match (dropped from both splits)")

    in_eval_window = (df["dmin"] >= a.eval_start) & (df["dmax"] <= a.eval_end)
    if a.holdout == "random":
        # randomly hold out a fraction of sites that HAVE eval-window data; train on the rest (any network)
        evaluable = sorted(df.loc[in_eval_window & df["network"].notna(), "site_id"].unique())
        random.seed(a.seed); random.shuffle(evaluable)
        n_h = max(1, int(round(len(evaluable) * a.holdout_frac)))
        held = set(evaluable[:n_h])
        ev = df[df["site_id"].isin(held) & in_eval_window].copy()
        ft = df[(~df["site_id"].isin(held)) & df["network"].notna()
                & (df["dmin"] >= a.ft_start) & (df["dmax"] <= a.ft_end)].copy()
        print(f"[splits] RANDOM holdout: {len(held)}/{len(evaluable)} eval-capable sites held out (seed {a.seed})")
    else:
        is_eval = df["network"].eq(a.eval_network)
        ft = df[(~is_eval) & df["network"].notna() & (df["dmin"] >= a.ft_start) & (df["dmax"] <= a.ft_end)].copy()
        ev = df[is_eval & in_eval_window].copy()

    # train/val split of fine-tune pool BY SITE
    random.seed(a.seed)
    ft_sites = sorted(ft["site_id"].unique())
    random.shuffle(ft_sites)
    n_val = max(1, int(round(len(ft_sites) * a.val_frac)))
    val_sites = set(ft_sites[:n_val])
    tr = ft[~ft["site_id"].isin(val_sites)]
    vl = ft[ft["site_id"].isin(val_sites)]

    def write(d, name):
        d[["path", "site_id"]].to_csv(f"{out_dir}/{name}.csv", index=False)

    write(tr, "finetune_train")
    write(vl, "finetune_val")
    write(ev, "eval_scan_pre2010")

    held_desc = f"random {a.holdout_frac:.0%}" if a.holdout == "random" else f"network!={a.eval_network}"
    print(f"\n[splits] depth={a.depth}  holdout={held_desc}  ->  {out_dir}")
    print(f"  fine-tune pool : {len(ft):6d} windows, {ft.site_id.nunique():4d} sites ({a.ft_start}..{a.ft_end})")
    print("    by network   :", ft.groupby('network').agg(windows=('path', 'size'),
                                                           sites=('site_id', 'nunique')).to_dict('index'))
    print(f"    train        : {len(tr):6d} windows, {tr.site_id.nunique():4d} sites")
    print(f"    val          : {len(vl):6d} windows, {vl.site_id.nunique():4d} sites")
    print(f"  eval ({a.eval_start[:4]}-{a.eval_end[:4]}) : {len(ev):6d} windows, {ev.site_id.nunique():4d} sites")
    print("    by network   :", ev.groupby('network').agg(windows=('path', 'size'),
                                                          sites=('site_id', 'nunique')).to_dict('index'))

if __name__ == "__main__":
    main()
