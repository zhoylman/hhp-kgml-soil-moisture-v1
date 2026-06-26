#!/usr/bin/env python
"""
Improved fine-tune recipe (A/B test on the mini subset before the HPC retrain).
Changes vs 4_2b (everything else identical: architecture, loss, 180-day windows,
year freeze ch9, depths, batch size, 30 epochs):
  - best-val checkpoint  : save the epoch with best val KGE, not the last
  - warmup + cosine LR   : 2-epoch warmup, then cosine decay PEAK_LR -> PEAK_LR*FLOOR
                           (replaces the near-dead 1e-5 -> 1e-7 step)
  - discriminative LR    : encoder layers at ENC_MULT x the decoder LR
  - AdamW + weight decay : small-data regularization

Run: CUDA_DEVICE_ORDER=PCI_BUS_ID DEPTH_FLAG=middle EXP_TAG=... CUDA_VISIBLE_DEVICES=0 NW=12 python py/4_2c-exp-finetune.py
"""
import sys, os, time, math, copy
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_config import *
from define_hyperparams import *

DEPTH       = os.environ.get("DEPTH_FLAG", "middle")
TAG         = os.environ.get("EXP_TAG", "")
BASE        = "/data/ssd2/soil-moisture-ml"
SPLIT_DIR   = f"{BASE}/split-definitions-exp-{DEPTH}{TAG}"
LOADER_DIR  = f"{BASE}/full-dataloader-{DEPTH}"
OUT_DIR     = f"{BASE}/results-exp-finetune-{DEPTH}{TAG}"
PRETRAINED  = {
    "middle":  f"{BASE}/results-pretrain/uNET_middle_18339_sites_08_09_2025-23_18_00_709581/model.pt",
    "shallow": f"{BASE}/results-pretrain/uNET-shallow_18893_sites_08_11_2025-11_06_05_942687/model.pt",
}[DEPTH]

YEAR_CH, YEAR_FROZEN = 9, (2013 - 1979) / (2023 - 1979)
NUM_EPOCHS = 30
PEAK_LR  = 1e-5     # decoder peak (proven); encoder gets ENC_MULT x this
ENC_MULT = 0.3
WARMUP   = 2
FLOOR    = 0.1      # cosine floor as fraction of peak  -> 1e-6
WD       = 1e-4

def compute_kge(yhat, yobs):
    m = ~np.isnan(yhat) & ~np.isnan(yobs)
    if m.sum() < 2: return np.nan
    s, o = yhat[m], yobs[m]
    if s.std() == 0 or o.std() == 0: return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (s.std()/o.std()-1)**2 + (s.mean()/o.mean()-1)**2)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tmpl = StreamflowTrainDatasetTemplate(csv_file=get_first_file_path(LOADER_DIR))
    dim_1, dim_2 = tmpl.x_tensor.shape[1], tmpl.x_tensor.shape[2]
    nw = int(os.environ.get("NW", "12"))
    train_loader = DataLoader(StreamflowTrainDatasetXY(f"{SPLIT_DIR}/finetune_train.csv", dim_1, dim_2),
                              batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader   = DataLoader(StreamflowTrainDatasetXY_Test(f"{SPLIT_DIR}/finetune_val.csv", dim_2),
                              batch_size=64, shuffle=False, num_workers=8)
    print(f"[ftc] depth={DEPTH}{TAG}  train={len(train_loader.dataset)} val={len(val_loader.dataset)} windows")

    model = uNet(num_classes, dim_2, max_channels, dropout).to('cuda')
    model.load_state_dict(torch.load(PRETRAINED, map_location='cuda'))

    enc = [p for n, p in model.named_parameters() if n.startswith("encoder")]
    dec = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
    opt = torch.optim.AdamW([{"params": enc, "lr": PEAK_LR * ENC_MULT},
                             {"params": dec, "lr": PEAK_LR}], weight_decay=WD)
    base = [PEAK_LR * ENC_MULT, PEAK_LR]

    def lr_mult(ep):
        if ep < WARMUP: return (ep + 1) / WARMUP
        prog = (ep - WARMUP) / max(1, NUM_EPOCHS - WARMUP)
        return FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * prog))

    def val_kge():
        model.eval(); ys, os_ = [], []
        with torch.no_grad():
            for X, y, _ in val_loader:
                X = X.to('cuda'); X[:, :, YEAR_CH] = YEAR_FROZEN
                ys.append(model(X).cpu().view(-1).numpy()); os_.append(y.view(-1).numpy())
        return compute_kge(np.concatenate(ys), np.concatenate(os_))   # pooled (selection signal)

    t0 = time.time(); best_kge, best_state, best_ep, hist = -1e9, None, -1, []
    for epoch in range(NUM_EPOCHS):
        m = lr_mult(epoch)
        for g, b in zip(opt.param_groups, base): g["lr"] = b * m
        model.train(); bl = []
        for X, y in train_loader:
            X = X.to('cuda'); X[:, :, YEAR_CH] = YEAR_FROZEN
            opt.zero_grad(); loss = criterion(model(X), y.to('cuda')); loss.backward(); opt.step()
            bl.append(loss.item())
        vk = val_kge(); hist.append((epoch, float(np.mean(bl)), float(vk)))
        if vk > best_kge: best_kge, best_ep = vk, epoch        # tracked for reference only
        print(f"[ftc] ep {epoch:2d} loss {np.mean(bl):.5f} val_KGE(pooled) {vk:.4f} lr {opt.param_groups[1]['lr']:.2e}", flush=True)

    torch.save(model.state_dict(), f"{OUT_DIR}/model.pt")      # LAST epoch (no early stopping)
    pd.DataFrame(hist, columns=["epoch", "train_loss", "val_kge_pooled"]).to_csv(f"{OUT_DIR}/val_kge.csv", index=False)
    print(f"[ftc] DONE ({(time.time()-t0)/60:.1f} min) saved LAST epoch -> {OUT_DIR}/model.pt "
          f"(peak pooled val {best_kge:.4f} @ ep {best_ep}, reference only)")

if __name__ == "__main__":
    main()
