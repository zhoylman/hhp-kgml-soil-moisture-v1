#!/usr/bin/env python
"""
Network-value experiment fine-tune.

Start from the SPoRT-PRETRAINED checkpoint (NOT the production obs-tuned model,
which saw SCAN), fine-tune on the non-SCAN 2020+ split (year frozen), single
train/val run (no k-fold). SCAN is held out entirely for the pre-period eval
(see exp_eval_vs_sport.py). Recipe matches production 4_2: 30 epochs, Adam,
LR 1e-5 -> 1e-7 at epoch 5, L1+derivative+bias loss, encoder unfrozen, batch 32.

year freeze: feature channel 9 -> 0.7727 in BOTH train and val loops.

Run:  DEPTH_FLAG=middle CUDA_VISIBLE_DEVICES=0 python py/4_2b-exp-finetune.py
"""
import sys, os, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_config import *           # uNet, dataset classes, get_first_file_path
from define_hyperparams import *     # batch_size, max_channels, dropout, num_classes, criterion

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

YEAR_CH     = 9
YEAR_FROZEN = (2013 - 1979) / (2023 - 1979)   # 0.7727..., same constant train + eval
NUM_EPOCHS  = 30
FAST_LR, SLOW_LR, LR_SWITCH = 1e-5, 1e-7, 5

def compute_kge(yhat, yobs):
    m = ~np.isnan(yhat) & ~np.isnan(yobs)
    if m.sum() < 2:
        return np.nan
    s, o = yhat[m], yobs[m]
    if s.std() == 0 or o.std() == 0:
        return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    template = StreamflowTrainDatasetTemplate(csv_file=get_first_file_path(LOADER_DIR))
    dim_1, dim_2 = template.x_tensor.shape[1], template.x_tensor.shape[2]
    input_size = dim_2
    print(f"[ft] depth={DEPTH} input_size={input_size} (year freeze ch{YEAR_CH}={YEAR_FROZEN:.6f})")

    nw = int(os.environ.get("NW", "64"))   # lower (e.g. NW=12) when sharing the box with the regen
    train_data   = StreamflowTrainDatasetXY(f"{SPLIT_DIR}/finetune_train.csv", dim_1, dim_2)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_data     = StreamflowTrainDatasetXY_Test(f"{SPLIT_DIR}/finetune_val.csv", dim_2)
    val_loader   = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8)
    print(f"[ft] train windows={len(train_data)}  val windows={len(val_data)}")

    model = uNet(num_classes, input_size, max_channels, dropout).to('cuda')
    model.load_state_dict(torch.load(PRETRAINED, map_location='cuda'))
    print(f"[ft] loaded pretrained: {PRETRAINED}")

    # match production: encoder trainable (whole net fine-tunes)
    for name, p in model.named_parameters():
        if any(name.startswith(e) for e in ("encoder1", "encoder2", "encoder3", "encoder4")):
            p.requires_grad = True
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FAST_LR)

    def val_kge():
        model.eval(); ks = []
        with torch.no_grad():
            for X, y, _ in val_loader:
                X = X.to('cuda'); X[:, :, YEAR_CH] = YEAR_FROZEN
                yh = model(X).cpu().view(-1).numpy(); yo = y.view(-1).numpy()
                ks.append(compute_kge(yh, yo))
        return float(np.nanmedian(ks))

    t0 = time.time()
    print(f"[ft] epoch 0 (pretrained) val KGE: {val_kge():.4f}", flush=True)
    hist = []
    for epoch in range(NUM_EPOCHS):
        if epoch == LR_SWITCH:
            for g in opt.param_groups: g['lr'] = SLOW_LR
        model.train(); bl = []
        for X, y in train_loader:
            X = X.to('cuda'); X[:, :, YEAR_CH] = YEAR_FROZEN     # <-- freeze year (train)
            opt.zero_grad()
            loss = criterion(model(X), y.to('cuda'))
            loss.backward(); opt.step(); bl.append(loss.item())
        msg = f"[ft] epoch {epoch} loss {np.mean(bl):.5f}"
        if epoch % 5 == 0 and epoch > 0:
            vk = val_kge(); hist.append((epoch, vk)); msg += f"  val_KGE {vk:.4f}"
        print(msg, flush=True)

    torch.save(model.state_dict(), f"{OUT_DIR}/model.pt")
    pd.DataFrame(hist, columns=['epoch', 'val_kge']).to_csv(f"{OUT_DIR}/val_kge.csv", index=False)
    print(f"[ft] DONE ({(time.time()-t0)/60:.1f} min) -> {OUT_DIR}/model.pt")

if __name__ == "__main__":
    main()
