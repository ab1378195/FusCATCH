import os
import pandas as pd
import numpy as np
import random
import torch
from typing import Union, Optional, Tuple
from torch.utils.data import DataLoader


def is_st(data: pd.DataFrame) -> bool:
    return data.shape[1] == 4


def process_data_df(data: pd.DataFrame, nrows=None) -> pd.DataFrame:
    label_exists = "label" in data["cols"].values

    all_points = data.shape[0]
    columns = data.columns

    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
    else:
        n_points = data.iloc[:, 1].value_counts().max()

    is_univariate = n_points == all_points
    n_cols = all_points // n_points
    df = pd.DataFrame()
    cols_name = data["cols"].unique()

    if columns[0] == "date" and not is_univariate:
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    elif columns[0] != "date" and not is_univariate:
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

    elif columns[0] == "date" and is_univariate:
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    else:
        df[cols_name[0]] = data.iloc[:, 0]

    if label_exists:
        last_col_name = df.columns[-1]
        df.rename(columns={last_col_name: "label"}, inplace=True)

    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]

    return df


def process_data_np(df: pd.DataFrame, nrows=None) -> np.ndarray:
    pivot_df = df.pivot_table(index="date", columns=["id", "cols"], values="data")

    sensors = df["id"].unique()
    features = df["cols"].unique()

    pivot_df = pivot_df.reindex(
        columns=pd.MultiIndex.from_product([sensors, features]), fill_value=np.nan
    )

    data_np = pivot_df.to_numpy().reshape(len(pivot_df), len(sensors), len(features))

    data_np = np.transpose(data_np, (0, 2, 1))

    if nrows is not None:
        data_np = data_np[:nrows, :, :]

    return data_np


def read_data(path: str, nrows=None) -> Union[pd.DataFrame, np.ndarray]:
    data = pd.read_csv(path)

    if is_st(data):
        return process_data_np(data, nrows)
    else:
        return process_data_df(data, nrows)


def split_before(
    data: Union[pd.DataFrame, np.ndarray], index: int
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[np.ndarray, np.ndarray]]:

    if isinstance(data, pd.DataFrame):
        return data.iloc[:index, :], data.iloc[index:, :]
    elif isinstance(data, np.ndarray):
        return data[:index, :], data[index:, :]
    else:
        raise TypeError("Unsupported data type")


def fix_random_seed(seed: Optional[int] = 2021):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_data(dataset_path: str, train_length: int):
    fix_random_seed()

    data = read_data(dataset_path)
    data = data.reset_index(drop=True)

    train, test = split_before(data, train_length)

    train_data = train.loc[:, train.columns != "label"]
    train_label = train.loc[:, ["label"]]

    test_data = test.loc[:, train.columns != "label"]
    test_label = test.loc[:, ["label"]]

    return train_data, train_label, test_data, test_label


def train_val_split(train_data, train_label, ratio):
    if ratio == 1:
        return train_data, None
    else:
        border = int((train_data.shape[0]) * ratio)
        train_data_value, valid_data_rest = split_before(train_data, border)
        train_label_value, valid_label_value = split_before(train_label, border)
        return train_data_value, valid_data_rest, train_label_value, valid_label_value


class SegLoader(object):
    def __init__(self, data, labels, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.data = data
        self.labels = labels
        self.test_labels = data

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train" or self.mode == "val" or self.mode == "tfad":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        # elif self.mode == "val":
        #     return (self.data.shape[0] - self.win_size) // self.step + 1
        # elif self.mode == "test":
        #     return (self.data.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.data.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train" or self.mode == "val" or self.mode == "tfad":
            return np.float32(self.data[index : index + self.win_size]), np.float32(
                self.labels[index: index+self.win_size]
            )
        # elif self.mode == 'val':
        #     return np.float32(self.data[index : index + self.win_size]), np.float32(
        #         self.labels[index: index+self.win_size]
        #     )
        elif self.mode == 'test':
            return np.float32(self.data[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.data[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


def anomaly_detection_data_provider(
    data, label, batch_size, win_size=100, step=100, mode='train'
):
    dataset = SegLoader(data, label, win_size, 1, mode)

    shuffle = False
    if mode == 'train' or mode == 'val':
        shuffle = True

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )
    return data_loader
