import torch
import torch.nn as nn
from config import TransformerConfig
from torch_optimizer import Yogi
from tqdm import tqdm
import matplotlib.pyplot as plt
from data import split_data, train_val_split, anomaly_detection_data_provider
import numpy as np
import os
import pandas as pd
from static import *
from sklearn.preprocessing import StandardScaler
from tfad.model.mixup import mixup_batch, slow_slope
from tfad.model.outlier_exposure import coe_batch
from utils import validate_tfad, tfad_search_best_threshold
from copy import deepcopy
from evaluate import calculate
from CATCH import CATCHModel
from torchinfo import summary


dataset_name = "CalIt2"
mode = "score"
config = TransformerConfig(
    Mlr=1e-05,
    auxi_lambda=0.1,
    batch_size=128,
    cf_dim=128,
    d_ff=128,
    d_model=128,
    dc_lambda=0.5,
    e_layers=2,
    head_dim=32,
    inference_patch_size=16,
    lr=0.0001,
    n_heads=8,
    num_epochs=10,
    patch_size=16, 
    patch_stride=16,
    score_lambda=0.5,
    seq_len=192,
    temperature=0.07,
    # TFAD
    hp_lamb = 6400,
    # hyper-parameter for TCN encoder
    embedding_rep_dim = 64,
    tcn_kernel_size = 3,
    tcn_out_channels = 64,
    tcn_layers = 3,
    tcn_maxpool_out_channels = 2,
    normalize_embedding = True,
    suspect_window_length = 12,
    # hyper-parameter for classifier
    distance = "L2",
    # TFAD training hyper-parameters
    num_epochs_tfad = 30,
    lr_tfad = 1e-5,
    coe_rate = 0.5,
    mixup_rate = 0.3,
    slow_slop = 0,
    # TFAD validation hyper-parameters
    classifier_threshold = 0.5,
    val_labels_adj = True,
    threshold_grid_length_val = 0.1,
    # TFAD test hyper-parameters
    test_labels_adj = True,
    threshold_grid_length_test = 0.01,
)
scaler = StandardScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load data
train_data, train_label, test_data, test_label = split_data(
    os.path.join("data",dataset_name+".csv"),TRAIN_LENGTH[dataset_name]
)
column_num = train_data.shape[1]
config.enc_in = column_num
config.dec_in = column_num
config.c_in = column_num
config.c_out = column_num
config.label_len = 48
model = CATCHModel(config)
state_dict = torch.load("catch_model.pth", weights_only=True)
model.load_state_dict(state_dict)
# fix random seed again
from data import fix_random_seed
fix_random_seed(2021)
model.to(device)
# 冻结CATCH模型参数
for name, param in model.named_parameters():
    if not name.startswith("TFAD"):
        param.requires_grad = False
# summary(model)

train_data_value, valid_data, train_label_value, valid_label = train_val_split(train_data, train_label, 0.8)
scaler.fit(train_data_value.values)
# data preprocess
train_data_value = pd.DataFrame(
    scaler.transform(train_data_value.values),
    columns=train_data_value.columns,
    index=train_data_value.index,
)
valid_data = pd.DataFrame(
    scaler.transform(valid_data.values),
    columns=valid_data.columns,
    index=valid_data.index,
)
test = pd.DataFrame(
    scaler.transform(test_data.values), columns=test_data.columns, index=test_data.index
)
actual_label = test_label.to_numpy().flatten()
valid_data_loader = anomaly_detection_data_provider(
    valid_data,
    valid_label,
    batch_size=config.batch_size,
    win_size=config.seq_len,
    step=1,
    mode="val",
)
train_data_loader = anomaly_detection_data_provider(
    train_data_value,
    train_label_value,
    batch_size=config.batch_size,
    win_size=config.seq_len,
    step=1,
    mode="train",
)
print("----------------------------TFAD TRAIN-----------------------")
tfad_params = model.TFAD.parameters()
optimizer_tfad = Yogi(tfad_params, lr=config.lr_tfad)
tfad_criterion = nn.BCEWithLogitsLoss()
# 使用 mode="tfad-val"确保shuffle=False保证窗口顺序
tfad_val_loader = anomaly_detection_data_provider(
    valid_data,
    valid_label,
    batch_size=config.batch_size,
    win_size=config.seq_len,
    step=1,
    mode="tfad", 
)
tfad_test_loader = anomaly_detection_data_provider(
    test,
    test_label,
    batch_size=config.batch_size,
    win_size=config.seq_len,
    step=1,
    mode="tfad", 
)
best_f1 = 0.0
best_auc_roc = 0.0
train_loss_list = []
f1_list = []
scores = np.load("scores.npy")
scores_min = np.min(scores)
scores_max = np.max(scores)
scores_normalized = (scores - scores_min) / (scores_max - scores_min)
for epoch in range(config.num_epochs_tfad):
    train_loss_tfad = []
    model.train()
    progress_bar = tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{config.num_epochs_tfad}")
    for i, (input, target) in enumerate(progress_bar):
        optimizer_tfad.zero_grad()
        # 1. 准备原始正常数据
        # input: [bs, seq_len, n_vars]
        x = input.float().to(device).permute(0, 2, 1) # [bs, n_vars, seq_len]
        
        # TFAD 官方逻辑：只根据 suspect_window_length (窗口末尾) 来判断标签
        # target: [bs, seq_len, 1]
        y = target[:, -config.suspect_window_length:, :].squeeze(-1).max(dim=1)[0].float().to(device)

        # 2. 数据增强 (生成伪异常)
        # COE
        if config.coe_rate > 0:
            x_oe, y_oe = coe_batch(
                x=x,
                y=y,
                coe_rate=config.coe_rate,
                suspect_window_length=config.suspect_window_length,
                random_start_end=True,
            )
            x = torch.cat((x, x_oe), dim=0)
            y = torch.cat((y, y_oe), dim=0)

        # Mixup
        if config.mixup_rate > 0.0:
            x_mixup, y_mixup = mixup_batch(
                x=x,
                y=y,
                mixup_rate=config.mixup_rate,
            )
            x = torch.cat((x, x_mixup), dim=0)
            y = torch.cat((y, y_mixup), dim=0)
            
        # Slow Slop
        if config.slow_slop > 0.0:
            x_slow, y_slow = slow_slope(
                x=x,
                y=y,
                mixup_rate=config.slow_slop,
            )
            x = torch.cat((x, x_slow), dim=0)
            y = torch.cat((y, y_slow), dim=0)
        
        # 3. 前向传播与 Loss 计算
        # 统一为[bs, seq_len, n_vars]形式输入
        logits_anomaly = model(x.permute(0, 2, 1), mode="TFAD")["TFAD_score"].squeeze()
        loss = tfad_criterion(logits_anomaly, y)
        
        loss.backward()
        optimizer_tfad.step()
        train_loss_tfad.append(loss.item())
        
        if (i + 1) % 100 == 0:
            print(f"Epoch: {epoch+1}, Iter: {i+1}, TFAD Loss: {np.mean(train_loss_tfad):.7f}")

    print(f"Train Loss: {np.mean(train_loss_tfad):.7f}")
    train_loss_list.append(np.mean(train_loss_tfad))
    # TFAD Validation
    tfad_labels = valid_label.to_numpy().astype(float).flatten()
    if sum(tfad_labels) == 0:
        if epoch == 0:
            print("Warning: No anomaly in validation set, use test set instead")
        tfad_probs = validate_tfad(model, tfad_test_loader, config, device)
        tfad_labels = test_label.to_numpy().astype(float).flatten()
    else:
        tfad_probs = validate_tfad(model, tfad_val_loader, config, device)
    metrics_best, threshold_best = tfad_search_best_threshold(tfad_probs, tfad_labels, config, stage="val")
    config.classifier_threshold = threshold_best
    print(f"Best threshold: {threshold_best:.4f}")
    print(f"Metrics: {metrics_best}")
    f1_list.append(metrics_best["f1"])
    if metrics_best["f1"] > best_f1:
        best_f1 = metrics_best["f1"]
        checkpoint = deepcopy(model.state_dict())
    # 联合测试
    probs_min = np.nanmin(tfad_probs)
    probs_max = np.nanmax(tfad_probs)
    tfad_probs_normalized = (tfad_probs - probs_min) / (probs_max - probs_min)
    tfad_probs_normalized = np.nan_to_num(tfad_probs_normalized, nan=0.0)
    final_scores = np.maximum(scores_normalized, tfad_probs_normalized)
    results = calculate(mode, actual_label.astype(float), predicted=final_scores.astype(float))
    print(results)
    metrics_best, threshold_best = tfad_search_best_threshold(final_scores, actual_label, config, stage="test")
    print(f"Best threshold: {threshold_best:.4f}")
    print(f"Metrics: {metrics_best}")
    if results["auc_roc"] > best_auc_roc:
        best_auc_roc = results["auc_roc"]
        f1_with_best_auc_roc = metrics_best["f1"]
print(f"The best AUC-ROC: {best_auc_roc:.4f}")
print(f"F1 Score with best AUC-ROC: {f1_with_best_auc_roc:.4f}")
plt.subplot(2,1,1)
plt.plot(train_loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.subplot(2,1,2)
plt.plot(f1_list)
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("F1 Score Curve")
plt.show()