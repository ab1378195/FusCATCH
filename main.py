from static import *
from config import TransformerConfig
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from loss import frequency_loss, frequency_criterion
from data import split_data, train_val_split, anomaly_detection_data_provider
import os
from CATCH import CATCHModel
from torchinfo import summary
import pandas as pd
from earlyStopping import EarlyStopping
from torch.optim import lr_scheduler
import numpy as np
from utils import detect_validate, adjust_learning_rate, padding, coe_batch, mixup_batch, slow_slope, coral_loss
from predict import infer_score, infer_label
from evaluate import calculate
from copy import deepcopy
import matplotlib.pyplot as plt
import time
import itertools
import math

def main():
    auxi_loss_fn = frequency_loss(config)
    tfad_criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=config.patience, verbose=True)
    model = CATCHModel(config)
    # fix random seed again
    from data import fix_random_seed
    fix_random_seed(2021)
    model.to(device)
    # summary(model)

    # train
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
    print("---------------------CATCH TRAIN------------------------")
    tfad_loss_list = []
    time_now = time.time()
    for epoch in range(config.num_epochs):
        iter_count = 0
        train_loss = []
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
            z_freq = outputs.get("z_freq", None)
            tfad_embedding_patch = outputs.get("tfad_embedding_patch", None)
            
            rec_loss = criterion(output, x)
            norm_input = model.revin_layer(x, 'transform')
            auxi_loss = auxi_loss_fn(output_complex, norm_input)
            tfad_loss = tfad_criterion(TFAD_score, y)

            loss = rec_loss + config.dc_lambda * dcloss + config.auxi_lambda * auxi_loss + config.tfad_lambda * tfad_loss
            if getattr(config, "coral_lambda", 0.0) > 0 and z_freq is not None and tfad_embedding_patch is not None:
                z_pool = z_freq.mean(dim=1)
                t_pool = tfad_embedding_patch.mean(dim=1)
                align_loss = coral_loss(z_pool, t_pool)
                loss = loss + config.coral_lambda * align_loss
            # print("RecLoss:{:.7f}, DCLoss:{:.7f}, AuxiLoss:{:.7f}, TFADLoss:{:.7f}".format(rec_loss, dcloss, auxi_loss, tfad_loss))
            train_loss.append(loss.item())
            tfad_loss_list.append(tfad_loss.item())

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
                speed = (time.time() - time_now) / iter_count
                left_time = speed * (
                        (config.num_epochs - epoch) * train_steps - i
                )
                print(
                    "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                        speed, left_time
                    )
                )
                iter_count = 0
                time_now = time.time()

            loss.backward()
            optimizer.step()

        print("Epoch: {}".format(epoch + 1))
        train_loss = np.average(train_loss)
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
    print("---------------------------CATCH TEST-------------------------")
    if mode=="score":
        scores = infer_score(model, thre_loader, time_anomaly_criterion, freq_anomaly_criterion, config.score_lambda)
        predict_labels = deepcopy(scores)
        predict_labels, scores = padding(actual_label, predict_labels, scores)
        results = calculate(mode, actual_label.astype(float), predicted=predict_labels.astype(float))
        print(results)
        return results
    elif mode=="label":
        if not isinstance(config.anomaly_ratio, list):
            config.anomaly_ratio = [config.anomaly_ratio]
        predict_labels, scores = infer_label(model, thre_loader, time_anomaly_criterion, freq_anomaly_criterion, config.score_lambda, train_data_loader, test_data_loader, config.anomaly_ratio)
        results_list = []
        for ratio, predict_label in predict_labels.items():
            predict_label, scores = padding(actual_label, predict_label, scores)
            results = calculate(mode, actual_label.astype(float), predicted=predict_label.astype(float))
            results["ratio"] = ratio
            results_list.append(results)
        print(results_list)
        return results_list

def _format_hparam_key(hparams: dict) -> str:
    return ",".join([f"{k}:{hparams[k]}" for k in sorted(hparams.keys())])

def run_hparam_search(parameter_search_dict: dict) -> tuple[dict, str, float]:
    keys = list(parameter_search_dict.keys())
    values_list = [parameter_search_dict[k] for k in keys]
    total = 1
    for vs in values_list:
        total *= len(vs)
    print(f"Total combinations: {total}")

    results_by_config: dict[str, dict] = {}
    best_auc = -math.inf
    best_key = ""

    for combo in itertools.product(*values_list):
        hparams = dict(zip(keys, combo))
        for k, v in hparams.items():
            setattr(config, k, v)

        config_key = _format_hparam_key(hparams)
        try:
            result = main()
        except Exception as e:
            results_by_config[config_key] = {"error": str(e)}
            continue

        results_by_config[config_key] = result
        auc = None
        if isinstance(result, dict):
            auc = result.get("auc_roc", None)
        if auc is None:
            continue
        try:
            auc_val = float(auc)
        except Exception:
            continue
        if auc_val > best_auc:
            best_auc = auc_val
            best_key = config_key

    print(f"Best auc_roc: {best_auc}")
    print(f"Best hparams: {best_key}")
    return results_by_config, best_key, best_auc

if __name__ == "__main__":
    # parameters
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
        patch_size=16, 
        patch_stride=16,
        score_lambda=0.5,
        seq_len=192,
        temperature=0.07,
        normalize_embedding = True,
    )

    scaler = StandardScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()

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

    # parameter_search_dict = {
    #     "hp_lamb": [1, 400, 1600, 6400, 12800],
    #     "tcn_kernel_size": [3, 5],
    #     "tcn_out_channels": [32, 64, 128, 256],
    #     "tcn_layers": [2, 3, 4, 5],
    #     "tcn_maxpool_out_channels": [1, 2, 4, 8],
    #     "suspect_window_length": [1, 6, 12, 18],
    #     "distance": ["L2", "cosine"],
    #     # hyper-parameter for cross-attention
    #     "num_heads": [2, 4, 8],
    #     # TFAD training hyper-parameters
    #     "tfad_lambda": [0.01, 0.1, 0.5],
    #     # TFAD data augmentation hyper-parameters
    #     "coe_rate": [0.0, 0.01, 0.1, 0.5],
    #     "mixup_rate": [0.0, 0.01, 0.1, 0.5],
    #     "slow_slop": [0.0, 0.01, 0.1, 0.5], 
    #     "num_epochs": [10, 8, 15],
    # }
    parameter_search_dict = {
        "hp_lamb": [1, 400],
        "tcn_kernel_size": [3],
        "tcn_out_channels": [64],
        "tcn_layers": [3],
        "tcn_maxpool_out_channels": [4],
        "suspect_window_length": [12],
        "distance": ["L2"],
        # hyper-parameter for cross-attention
        "num_heads": [4],
        # TFAD training hyper-parameters
        "tfad_lambda": [0.5],
        # TFAD data augmentation hyper-parameters
        "coe_rate": [0.0],
        "mixup_rate": [0.0],
        "slow_slop": [0.01], 
        "num_epochs": [10, 15],
    }
    results_by_config, best_hparams, best_auc = run_hparam_search(parameter_search_dict)
