from torch import logit
import torch
import numpy as np


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == 'type3':
        lr_adjust = {
            epoch: (
                args.learning_rate
                if epoch < 3
                else args.learning_rate * (0.9 ** ((epoch - 3) // 1))
            )
        }
    elif args.lradj == 'type4':
        lr_adjust = {
            epoch: (
                args.learning_rate
                if epoch < 20
                else args.learning_rate * (0.5 ** ((epoch // 20) // 1))
            )
        }
    elif args.lradj == 'type5':
        lr_adjust = {
            epoch: (
                args.learning_rate
                if epoch < 10
                else args.learning_rate * (0.5 ** ((epoch // 10) // 1))
            )
        }
    elif args.lradj == 'type6':
        lr_adjust = {
            20: args.learning_rate * 0.5,
            40: args.learning_rate * 0.01,
            60: args.learning_rate * 0.01,
            8: args.learning_rate * 0.01,
            100: args.learning_rate * 0.01,
        }
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {
            epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1
        }
    elif args.lradj == '4':
        lr_adjust = {
            epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1
        }
    elif args.lradj == '5':
        lr_adjust = {
            epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1
        }
    elif args.lradj == '6':
        lr_adjust = {
            epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1
        }
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print('Updating learning rate to {}'.format(lr))

def detect_validate(model, valid_data_loader, criterion):
    total_loss = []
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for input, _ in valid_data_loader:
            
            input = input.to(device)

            outputs= model(input)

            output = outputs["z"][:, :, :]

            output = output.detach().cpu()
            true = input.detach().cpu()
            loss = criterion(output, true).detach().cpu().numpy()
            total_loss.append(loss)

    total_loss = np.mean(total_loss)
    model.train()
    return total_loss

def padding(actual_label, predict_label, scores):
    remaining_length = len(actual_label) - len(predict_label)
    # Pad the predict_label array with zeros at the end
    if remaining_length > 0:
        predict_label = np.pad(
            predict_label,
            (0, remaining_length),
            mode="constant",
            constant_values=0,
        )
        scores = np.pad(
            scores,
            (0, remaining_length),
            mode="constant",
            constant_values=0,
        )
    return predict_label, scores

def coe_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    coe_rate: float,
    suspect_window_length: int,
    random_start_end: bool = True,
) -> torch.Tensor:
    """Contextual Outlier Exposure.

    Args:
        x : Tensor of shape (batch, ts channels, time)
        y : Tensor of shape (batch, )
        coe_rate : Number of generated anomalies as proportion of the batch size.
        random_start_end : If True, a random subset within the suspect segment is permuted between time series;
            if False, the whole suspect segment is randomly permuted.
    """

    if coe_rate == 0:
        raise ValueError(f"coe_rate must be > 0.")
    batch_size = x.shape[0]
    ts_channels = x.shape[1]
    oe_size = int(batch_size * coe_rate)

    # Select indices
    idx_1 = torch.arange(oe_size)
    idx_2 = torch.arange(oe_size)
    while torch.any(idx_1 == idx_2):
        idx_1 = torch.randint(low=0, high=batch_size, size=(oe_size,)).type_as(x).long()
        idx_2 = torch.randint(low=0, high=batch_size, size=(oe_size,)).type_as(x).long()

    if ts_channels > 3:
        numb_dim_to_swap = np.random.randint(low=3, high=ts_channels, size=(oe_size))
        # print(numb_dim_to_swap)
    else:
        numb_dim_to_swap = np.ones(oe_size) * ts_channels

    x_oe = x[idx_1].clone()  # .detach()
    oe_time_start_end = np.random.randint(
        low=x.shape[-1] - suspect_window_length, high=x.shape[-1] + 1, size=(oe_size, 2)
    )
    oe_time_start_end.sort(axis=1)
    # for start, end in oe_time_start_end:
    for i in range(len(idx_2)):
        # obtain the dimensons to swap
        numb_dim_to_swap_here = int(numb_dim_to_swap[i])
        dims_to_swap_here = np.random.choice(
            range(ts_channels), size=numb_dim_to_swap_here, replace=False
        )

        # obtain start and end of swap
        start, end = oe_time_start_end[i]

        # swap
        x_oe[i, dims_to_swap_here, start:end] = x[idx_2[i], dims_to_swap_here, start:end]

    # Label as positive anomalies
    y_oe = torch.ones(oe_size).type_as(y)

    return x_oe, y_oe

def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    mixup_rate: float,
) -> torch.Tensor:
    """
    Args:
        x : Tensor of shape (batch, ts channels, time)
        y : Tensor of shape (batch, )
        mixup_rate : Number of generated anomalies as proportion of the batch size.
    """

    if mixup_rate == 0:
        raise ValueError(f"mixup_rate must be > 0.")
    batch_size = x.shape[0]
    mixup_size = int(batch_size * mixup_rate)  #

    # Select indices
    idx_1 = torch.arange(mixup_size)
    idx_2 = torch.arange(mixup_size)
    while torch.any(idx_1 == idx_2):
        idx_1 = torch.randint(low=0, high=batch_size, size=(mixup_size,)).type_as(x).long()
        idx_2 = torch.randint(low=0, high=batch_size, size=(mixup_size,)).type_as(x).long()

    # sample mixing weights:
    beta_param = float(0.05)
    beta_distr = torch.distributions.beta.Beta(
        torch.tensor([beta_param]), torch.tensor([beta_param])
    )
    weights = torch.from_numpy(np.random.beta(beta_param, beta_param, (mixup_size,))).type_as(x)
    oppose_weights = 1.0 - weights

    # Create contamination
    # corrected x_mix_2: idx_2
    x_mix_1 = x[idx_1].clone()
    x_mix_2 = x[idx_2].clone()
    x_mixup = (
        x_mix_1 * weights[:, None, None] + x_mix_2 * oppose_weights[:, None, None]
    )  # .detach()

    # Label as positive anomalies
    y_mixup = y[idx_1].clone() * weights + y[idx_2].clone() * oppose_weights

    return x_mixup, y_mixup


def slow_slope(
    x: torch.Tensor,
    y: torch.Tensor,
    mixup_rate: float,
) -> torch.Tensor:
    # print("x shape is", x.shape)
    # print("y shape is", y.shape)
    # print("y is", y)
    # print("x in slow_slop is", x)
    """
    Args:
        x : Tensor of shape (batch, ts channels, time)
        y : Tensor of shape (batch, )
        mixup_rate : Number of generated anomalies as proportion of the batch size.
    """

    if mixup_rate == 0:
        raise ValueError(f"mixup_rate must be > 0.")
    batch_size = x.shape[0]
    mixup_size = int(batch_size * mixup_rate)  #
    

    # Select indices
    idx_1 = torch.arange(mixup_size)
    x_mix_1 = x[idx_1].clone()

    # 构造一个单调增函数
    slop = torch.arange(x_mix_1[:, 0, :].shape[0])
    # print(slop.shape)
    

    # Create contamination
    # corrected x_mix_2: idx_2
    x_mixup = x[idx_1].clone()
    
#     print("x_mixup[:, 0, :] shape is", x_mixup[:][0][:].shape)
    
#     print("x_mixup[:, 0, :] is", x_mixup[:, 0, :])
    oe_size = int(x_mixup[:, 0, :].shape[1])
    # idx_2 = torch.arange(oe_size)
    
    s_r = torch.ones(slop.shape)
    s_c = (0.00001*torch.arange(oe_size))
    s_slop = torch.unsqueeze(s_r, dim=1)*torch.unsqueeze(s_c, dim=0)
    # print(s_slop.shape)
    # print(s_slop)
    
    
    x_mixup[:, 0, :] = x_mix_1[:, 0, :] + s_slop.cuda()
    
    # print("x_mixup[:, 0, :] in slow_slop is", x_mixup[:, 0, :])
    
    # Label as positive anomalies
    y_oe= torch.ones(mixup_size).type_as(y)

    return x_mixup, y_oe

def plot_anomaly_detection_result(dataset_name, real_labels_full, detected_labels_test, plot_column='Volume'):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    path = os.path.join("data", "raw", f"{dataset_name}.csv")
    df = pd.read_csv(path)
    
    # Convert to standard format
    dates = pd.to_datetime(df["Date"])
    values = df[plot_column].to_numpy()
    real = np.array(real_labels_full).astype(int)
    
    # Construct detection labels covering full dataset
    L = len(values)
    det = np.zeros(L, dtype=int)
    # Direct mapping: detected_labels_test corresponds to the last N points
    N_test = len(detected_labels_test)
    det[L-N_test:L] = detected_labels_test

    def intervals(labels):
        """将标签数组转换为（起始，结束）的区间列表"""
        res = []
        s = None
        for i, v in enumerate(labels):
            if v == 1 and s is None:
                s = i
            elif v == 0 and s is not None:
                res.append((s, i))
                s = None
        if s is not None:
            res.append((s, len(labels)))
        return res

    # 修改开始：合并绘图
    plt.figure(figsize=(16, 5))
    # 只创建一个子图
    plt.plot(dates, values, color="tab:blue", linewidth=1, alpha=0.85, label=f'{plot_column}') # 可选的曲线图例

    # 1. 绘制真实异常区域（保持原来的样式：绿色半透明背景）
    ri = intervals(real)
    for i, (s, e) in enumerate(ri):
        xs, xe = dates.iloc[s], dates.iloc[min(e, L - 1)]
        plt.axvspan(xs, xe, color="lime", alpha=0.3, zorder=2, label="Real anomalies" if i == 0 else "")

    # 2. 绘制预测异常区域（修改为：无填充，橙色边框和//图案）
    di = intervals(det)
    for i, (s, e) in enumerate(di):
        xs, xe = dates.iloc[s], dates.iloc[min(e, L - 1)]
        # 使用 edgecolor 设置边框颜色，facecolor='none' 确保无填充
        # hatch 设置斜线图案，lw 设置边框线宽
        plt.axvspan(xs, xe, 
                    facecolor='none',        # 无填充色
                    edgecolor='orange',      # 边框为橙色
                    linewidth=2,             # 边框线宽
                    hatch='//',              # 斜线图案
                    zorder=3, 
                    label="Detected anomalies" if i == 0 else "")
        
    # 设置图表信息
    plt.xlabel("Time", fontsize=12)
    plt.ylabel(plot_column, fontsize=12)
    plt.title(f"Anomaly Detection Results for {dataset_name}", fontsize=16)
    plt.legend(loc="upper right")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(dataset_name, actual, scores):
    """
    绘制 ROC 曲线并保存或显示。

    :param dataset_name: 数据集名称
    :param actual: 真实标签 (0 或 1)
    :param scores: 模型输出的异常分数
    """
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    fpr, tpr, thresholds = roc_curve(actual, scores)
    roc_auc = auc(fpr, tpr)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (面积: {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title(f'ROC曲线 - {dataset_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()
