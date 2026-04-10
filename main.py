from static import *
from config import TransformerConfig
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from loss import frequency_loss, frequency_criterion, coral_loss
from data import split_data, train_val_split, anomaly_detection_data_provider
import os
from CATCH import CATCHModel
import pandas as pd
from earlyStopping import EarlyStopping
from torch.optim import lr_scheduler
import numpy as np
from utils import detect_validate, adjust_learning_rate, padding, coe_batch, mixup_batch, slow_slope, plot_anomaly_detection_result
from predict import infer_score, infer_label
from evaluate import calculate
from copy import deepcopy
import matplotlib.pyplot as plt
from json import load

dataset_name = "CalIt2"
mode = "score"
with open(os.path.join("config",dataset_name+".json"), "r") as f:
    config = load(f)
config = TransformerConfig(**config)

scaler = StandardScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
auxi_loss_fn = frequency_loss(config)
tfad_criterion = nn.BCEWithLogitsLoss()

# load data
train_data, train_label, test_data, test_label = split_data(
    os.path.join("data",dataset_name+".csv"),TRAIN_LENGTH[dataset_name]
)

# fit data
column_num = train_data.shape[1]
config.enc_in = column_num
config.dec_in = column_num
config.c_in = column_num
config.c_out = column_num
config.label_len = 48
model = CATCHModel(config)
# fix random seed again
from data import fix_random_seed
fix_random_seed(2021)
model.to(device)

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

# train
early_stopping = EarlyStopping(patience=config.patience, verbose=True)
train_steps = len(train_data_loader)
main_params = [
    param for name, param in model.named_parameters() if 'mask_generator' not in name
]
optimizer = torch.optim.Adam(main_params, lr=config.lr)
optimizerM = torch.optim.Adam(model.mask_generator.parameters(), lr=config.Mlr)
scheduler = lr_scheduler.OneCycleLR(
    optimizer=optimizer,
    steps_per_epoch=train_steps,
    pct_start=config.pct_start,
    epochs=config.num_epochs,
    max_lr=config.lr,
)
schedulerM = lr_scheduler.OneCycleLR(
    optimizer=optimizerM,
    steps_per_epoch=train_steps,
    pct_start=config.pct_start,
    epochs=config.num_epochs,
    max_lr=config.Mlr,
)
print("---------------------PHASE TRAIN------------------------")
tfad_loss_list = []
for epoch in range(config.num_epochs):
    iter_count = 0
    train_loss = []
    tfad_temp_loss = []
    model.train()

    step = min(int(len(train_data_loader) / 10), 100)
    for i, (input, target) in enumerate(train_data_loader):
        # target: [bs, seq_len, 1]
        y = target[:, -config.suspect_window_length:, :].squeeze(-1).max(dim=1)[0].float().to(device) # y: [batch_size]
        x = input.float().to(device)

        if config.coe_rate > 0:
            # print(" coe_rate x shape is", x.shape)
            x_oe, y_oe = coe_batch(
                x=x,
                y=y,
                coe_rate=config.coe_rate,
                suspect_window_length=config.suspect_window_length,
                random_start_end=True,
            )
            # Add COE to training batch
            x = torch.cat((x, x_oe), dim=0)
            y = torch.cat((y, y_oe), dim=0)

        if config.mixup_rate > 0.0:
            # print("mixup_rate x shape is", x.shape)
            x_mixup, y_mixup = mixup_batch(
                x=x,
                y=y,
                mixup_rate=config.mixup_rate,
            )
            # Add Mixup to training batch
            x = torch.cat((x, x_mixup), dim=0)
            y = torch.cat((y, y_mixup), dim=0)
            
        if config.slow_slop > 0.0:
            # print("slow_slop x shape is", x.shape)
            x_mixup, y_mixup = slow_slope(
                x=x,
                y=y,
                mixup_rate=config.slow_slop,
            )
            # Add Mixup to training batch
            x = torch.cat((x, x_mixup), dim=0)
            y = torch.cat((y, y_mixup), dim=0)

        iter_count += 1
        optimizer.zero_grad()
        outputs = model(x)
        output = outputs["z"][:, :, :]
        output_complex = outputs["complex_z"]
        dcloss = outputs["dcloss"]
        TFAD_score = outputs["TFAD_score"].reshape(-1)
        z_freq = outputs["z_freq"]
        tfad_embedding_patch = outputs["tfad_embedding_patch"]
        
        rec_loss = criterion(output, x)
        norm_input = model.revin_layer(x, 'transform')
        auxi_loss = auxi_loss_fn(output_complex, norm_input)
        tfad_loss = tfad_criterion(TFAD_score, y)
        z_pool = z_freq.mean(dim=1)
        t_pool = tfad_embedding_patch.mean(dim=1)
        align_loss = coral_loss(z_pool, t_pool)

        loss = rec_loss + config.dc_lambda * dcloss + config.auxi_lambda * auxi_loss + config.tfad_lambda * tfad_loss + config.coral_lambda * align_loss
        train_loss.append(loss.item())
        tfad_temp_loss.append(tfad_loss.item())

        if (i + 1) % step == 0:
            optimizerM.step()
            optimizerM.zero_grad()

        if (i + 1) % 100 == 0:
            print(
                "\titers: {0}, epoch: {1} | training time loss: {2:.7f} | training fre loss: {3:.7f} | training dc loss: {4:.7f} | training tfad loss: {5:.7f}".format(
                    i + 1,
                    epoch + 1,
                    rec_loss.item(),
                    auxi_loss.item(),
                    dcloss.item(),
                    tfad_loss.item()
                )
            )
            iter_count = 0

        loss.backward()
        optimizer.step()

    print("Epoch: {}".format(epoch + 1))
    train_loss = np.average(train_loss)
    tfad_loss_list.append(np.average(tfad_temp_loss))
    valid_loss = detect_validate(model, valid_data_loader, criterion)
    print(
        "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, valid_loss
        )
    )

    early_stopping(valid_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    adjust_learning_rate(optimizer, scheduler, epoch + 1, config)
    adjust_learning_rate(optimizerM, schedulerM, epoch + 1, config, printout=False)

# validate
model.load_state_dict(early_stopping.check_point)
model.to(device)
model.eval()
test = pd.DataFrame(
    scaler.transform(test_data.values), columns=test_data.columns, index=test_data.index
)
thre_loader = anomaly_detection_data_provider(
    test,
    None,
    batch_size=config.batch_size,
    win_size=config.seq_len,
    step=1,
    mode="thre",
)
test_data_loader = anomaly_detection_data_provider(
    test,
    None,
    batch_size=config.batch_size,
    win_size=config.seq_len,
    step=1,
    mode="test",
)
time_anomaly_criterion = nn.MSELoss(reduction='none')
freq_anomaly_criterion = frequency_criterion(config)
actual_label = test_label.to_numpy().flatten()
print("---------------------------PHASE TEST-------------------------")
if mode=="score":
    scores = infer_score(model, thre_loader, time_anomaly_criterion, freq_anomaly_criterion, config.score_lambda)
    predict_labels = deepcopy(scores)
    predict_labels, scores = padding(actual_label, predict_labels, scores)
    results = calculate(mode, actual_label.astype(float), predicted=predict_labels.astype(float))
    print(results)
elif mode=="label":
    if not isinstance(config.anomaly_ratio, list):
        config.anomaly_ratio = [config.anomaly_ratio]
    predict_labels, scores = infer_label(model, thre_loader, time_anomaly_criterion, freq_anomaly_criterion, config.score_lambda, train_data_loader, test_data_loader, config.anomaly_ratio)
    results_list = []
    best_f1 = 0
    best_ratio = 0
    best_predict_label = None
    for ratio, predict_label in predict_labels.items():
        predict_label, scores = padding(actual_label, predict_label, scores)
        results = calculate(mode, actual_label.astype(float), predicted=predict_label.astype(float))
        results["ratio"] = ratio
        if results["f_score"] > best_f1:
            best_f1 = results["f_score"]
            best_ratio = ratio
            best_results = results
            best_predict_label = deepcopy(predict_label)
        print(results)
    print("The best f1 is {} with the ratio of {}".format(best_f1, best_ratio))
    
    all_real_labels = np.concatenate([train_label.to_numpy().flatten(), test_label.to_numpy().flatten()])
    plot_anomaly_detection_result(dataset_name, all_real_labels, best_predict_label)
# plt.plot(tfad_loss_list)
# plt.show()
