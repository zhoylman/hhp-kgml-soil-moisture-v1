# DIAGNOSTIC ONLY -- checks whether the production fine-tuning recipe's LR
# schedule (1e-5 for 5 epochs -> 1e-7) leaves accuracy on the table, by running
# the REAL pretrained-weight fine-tuning with LR held constant instead. Same
# architecture, same pretrained SPoRT-LIS init, same encoder-unfreezing, same
# loss -- ONLY the LR schedule differs. Writes to a NEW, separate directory;
# NEVER touches results-kfold-{depth}/ (production) or the nopretrain ablation
# dirs.
#
# Run:  DEPTH_FLAG=shallow LR_SCHEDULE=heldlr FOLDS=1 \
#         CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python py/4_2-kfold-finetune-heldlr-diagnostic.py

import sys, time, datetime, os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_config import *
from define_hyperparams import *

DEPTH = os.environ.get("DEPTH_FLAG", "middle")
assert DEPTH in ("shallow", "middle")
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "twophase")   # twophase (matches production) | heldlr (diagnostic)
FOLDS = [int(x) for x in os.environ.get("FOLDS", "").split(",")] if os.environ.get("FOLDS") else None
BASE = "/data/ssd2/soil-moisture-ml"

PRETRAINED = {
    "middle":  f"{BASE}/results-pretrain/uNET_middle_18339_sites_08_09_2025-23_18_00_709581/model.pt",
    "shallow": f"{BASE}/results-pretrain/uNET-shallow_18893_sites_08_11_2025-11_06_05_942687/model.pt",
}[DEPTH]

num_epochs = int(os.environ.get("NUM_EPOCHS", "30"))
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

def run_fold(fold, train_path, val_path):
    print(f"\n========== [{DEPTH}] FINETUNE-{LR_SCHEDULE.upper()} DIAGNOSTIC FOLD {fold} ==========")
    model_name = f'uNET_ftdiag_{LR_SCHEDULE}_{DEPTH}_fold_{fold}_{datetime.datetime.now().strftime("%m_%d_%Y-%H_%M_%S_%f")}'
    start_time = time.time()

    template = StreamflowTrainDatasetTemplate(csv_file=get_first_file_path(f'{BASE}/full-dataloader-{DEPTH}'))
    input_size = template.x_tensor.size()[2]

    train_data = StreamflowTrainDatasetXY(path_csv=train_path, dim_1=template.x_tensor.shape[1], dim_2=template.x_tensor.shape[2])
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=64)
    test_data = StreamflowTrainDatasetXY_Test(path_csv=val_path, dim_2=template.x_tensor.shape[2])
    data_loader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

    model = uNet(num_classes, input_size, max_channels, dropout).to('cuda')
    model.load_state_dict(torch.load(PRETRAINED, map_location='cuda'))
    print(f"✔ Loaded pre-trained model from {PRETRAINED}")

    tag = "" if LR_SCHEDULE == "twophase" else f"-{LR_SCHEDULE}"
    out_dir = f'{BASE}/results-kfold-{DEPTH}-finetune-diagnostic{tag}/fold_{fold}_{model_name}'

    encoder_layers = ["encoder1", "encoder2", "encoder3", "encoder4"]
    for name, param in model.named_parameters():
        if any(name.startswith(enc) for enc in encoder_layers):
            param.requires_grad = True

    initial_fast_lr = 1e-5
    slow_lr = 1e-7 if LR_SCHEDULE == "twophase" else initial_fast_lr
    lr_switch_epoch = 5
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_fast_lr)

    losses = []
    test_kge_df = pd.DataFrame(columns=['epoch', 'median_kge', 'lower_kge', 'upper_kge'])

    model.eval()
    test_kges = []
    for X, y, path in data_loader_test:
        with torch.no_grad():
            outputs = model(X.to('cuda'))
            yhat = outputs.cpu().view(-1).numpy(); yobs = y.cpu().view(-1).numpy()
            test_kges.append(compute_kge(yhat, yobs))
    test_kge_df = pd.concat([test_kge_df, pd.DataFrame([[0, np.nanmedian(test_kges), np.nanpercentile(test_kges,25), np.nanpercentile(test_kges,75)]], columns=['epoch','median_kge','lower_kge','upper_kge'])], ignore_index=True)
    print(f"✔ Initial (pretrained) Validation KGE: {np.nanmedian(test_kges)}")

    for epoch in range(num_epochs):
        if epoch == lr_switch_epoch:
            for pg in optimizer.param_groups:
                pg['lr'] = slow_lr
            print(f"✔ Switched to LR: {slow_lr}")

        model.train()
        batchloss = []
        for X_batch, y_batch in data_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.to('cuda'))
            loss = criterion(outputs, y_batch.to('cuda'))
            loss.backward(); optimizer.step()
            batchloss.append(loss.item())
        losses.append(np.mean(batchloss))
        print(f"Epoch: {epoch}, Loss: {np.mean(batchloss):.5f}")

        if epoch % 5 == 0 and epoch != 0:
            model.eval()
            test_kges = []
            for X, y, path in data_loader_test:
                with torch.no_grad():
                    outputs = model(X.to('cuda'))
                yhat = outputs.cpu().view(-1).numpy(); yobs = y.cpu().view(-1).numpy()
                test_kges.append(compute_kge(yhat, yobs))
            test_kge_df = pd.concat([test_kge_df, pd.DataFrame([[epoch, np.nanmedian(test_kges), np.nanpercentile(test_kges,25), np.nanpercentile(test_kges,75)]], columns=['epoch','median_kge','lower_kge','upper_kge'])], ignore_index=True)

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{out_dir}/model.pt')
    pd.DataFrame(losses, columns=['train_loss']).to_csv(f'{out_dir}/train_losses.csv', index=False)
    test_kge_df.to_csv(f'{out_dir}/test_kge.csv', index=False)

    runtime = time.time() - start_time
    print(f'⏱ [{DEPTH}] fold {fold} runtime: {runtime/60:.2f} minutes')
    del model; torch.cuda.empty_cache()

split_dir = f'{BASE}/split-definitions-kfold-{DEPTH}'
train_paths = sorted(glob.glob(f"{split_dir}/train_split_fold_*.csv"))
val_paths = sorted(glob.glob(f"{split_dir}/validation_split_fold_*.csv"))
assert len(train_paths) == len(val_paths) == 10

for fold, (tp, vp) in enumerate(zip(train_paths, val_paths), start=1):
    if FOLDS is not None and fold not in FOLDS:
        continue
    run_fold(fold, tp, vp)
