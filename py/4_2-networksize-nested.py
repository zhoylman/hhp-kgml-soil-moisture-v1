# Network-size (training-station-count) experiment, REDESIGNED per reviewer
# feedback: for each of the 10 existing k-fold partitions, validation is the
# FIXED held-out 10% for that fold (unchanged across sizes), and training uses
# NESTED prefixes of increasing size (100 subset of 200 subset of 300, ...)
# drawn from that fold's training pool. This isolates the effect of training-
# set size alone: fixed validation population, nested (not independently
# resampled) training sets, and 10 replicate folds giving real uncertainty at
# each size. Uses the PRODUCTION recipe (pretrained init, same two-phase LR
# schedule, same loss) -- identical to how the real k-fold models were built.
#
# Run:  DEPTH_FLAG=middle FOLD=1 SIZE=100 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
#         python py/4_2-networksize-nested.py

import sys, time, datetime, os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_config import *
from define_hyperparams import *

DEPTH = os.environ["DEPTH_FLAG"]
FOLD  = int(os.environ["FOLD"])
SIZE  = int(os.environ["SIZE"])
assert DEPTH in ("shallow", "middle")
BASE = "/data/ssd2/soil-moisture-ml"

PRETRAINED = {
    "middle":  f"{BASE}/results-pretrain/uNET_middle_18339_sites_08_09_2025-23_18_00_709581/model.pt",
    "shallow": f"{BASE}/results-pretrain/uNET-shallow_18893_sites_08_11_2025-11_06_05_942687/model.pt",
}[DEPTH]

TRAIN_CSV = f"{BASE}/split-definitions-networksize/{DEPTH}_fold{FOLD}_size{SIZE}.csv"
VAL_CSV   = f"{BASE}/split-definitions-kfold-{DEPTH}/validation_split_fold_{FOLD}.csv"   # FIXED across all sizes
OUT_DIR   = f"{BASE}/results-networksize-{DEPTH}/fold{FOLD}_size{SIZE}"

num_epochs = 30
criterion = L1WithWindowedDerivativeAndBiasLoss(lambda_deriv=1, lambda_bias=0.5)

def compute_kge(yhat, yobs):
    m = ~np.isnan(yhat) & ~np.isnan(yobs)
    if m.sum() < 2:
        return np.nan
    s, o = yhat[m], yobs[m]
    if s.std() == 0 or o.std() == 0 or o.mean() == 0:
        return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)

def get_first_file_path(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return os.path.join(directory, files[0]) if files else None

def main():
    print(f"\n========== [{DEPTH}] NETWORK-SIZE fold={FOLD} size={SIZE} ==========")
    start_time = time.time()

    template = StreamflowTrainDatasetTemplate(csv_file=get_first_file_path(f'{BASE}/full-dataloader-{DEPTH}'))
    input_size = template.x_tensor.size()[2]

    train_data = StreamflowTrainDatasetXY(path_csv=TRAIN_CSV, dim_1=template.x_tensor.shape[1], dim_2=template.x_tensor.shape[2])
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=32)
    test_data = StreamflowTrainDatasetXY_Test(path_csv=VAL_CSV, dim_2=template.x_tensor.shape[2])
    data_loader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

    model = uNet(num_classes, input_size, max_channels, dropout).to('cuda')
    model.load_state_dict(torch.load(PRETRAINED, map_location='cuda'))
    print(f"✔ Loaded pretrained: {PRETRAINED} | train n_sites={SIZE} (nested) | val=fixed fold {FOLD} holdout")

    encoder_layers = ["encoder1", "encoder2", "encoder3", "encoder4"]
    for name, param in model.named_parameters():
        if any(name.startswith(enc) for enc in encoder_layers):
            param.requires_grad = True

    initial_fast_lr, slow_lr, lr_switch_epoch = 1e-5, 1e-7, 5
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_fast_lr)

    losses = []
    for epoch in range(num_epochs):
        if epoch == lr_switch_epoch:
            for pg in optimizer.param_groups:
                pg['lr'] = slow_lr
        model.train()
        batchloss = []
        for X_batch, y_batch in data_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.to('cuda'))
            loss = criterion(outputs, y_batch.to('cuda'))
            loss.backward(); optimizer.step()
            batchloss.append(loss.item())
        losses.append(np.mean(batchloss))
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {np.mean(batchloss):.5f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), f'{OUT_DIR}/model.pt')
    pd.DataFrame(losses, columns=['train_loss']).to_csv(f'{OUT_DIR}/train_losses.csv', index=False)

    # final validation KGE on the FIXED fold holdout (raw window-level, quick sanity print;
    # the authoritative per-site metrics come from the point-based pipeline run separately)
    model.eval()
    test_kges = []
    with torch.no_grad():
        for X, y, path in data_loader_test:
            outputs = model(X.to('cuda'))
            yhat = outputs.cpu().view(-1).numpy(); yobs = y.cpu().view(-1).numpy()
            test_kges.append(compute_kge(yhat, yobs))
    median_kge = np.nanmedian(test_kges)
    pd.DataFrame({'var': ['depth','fold','size','median_kge','runtime_secs'],
                  'val': [DEPTH, FOLD, SIZE, median_kge, time.time()-start_time]}).to_csv(f'{OUT_DIR}/summary.csv', index=False)
    print(f"✔ [{DEPTH} fold {FOLD} size {SIZE}] final raw-window median KGE = {median_kge:.4f}  "
          f"| runtime {(time.time()-start_time)/60:.2f} min")

if __name__ == "__main__":
    main()
