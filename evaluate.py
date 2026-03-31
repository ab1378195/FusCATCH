# -*- coding: utf-8 -*-

import numpy as np
from sklearn import metrics
from metrics.vus_metrics import metricor, generate_curve
from metrics.generics import convert_vector_to_events
from metrics.metrics import pr_from_events
from metrics.vus_metrics import metricor
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from typing import List
from static import METRICS_SCORE, METRICS_LABEL

metricor_grader = metricor()
# metrics for score mode
def get_list_anomaly(labels: np.ndarray) -> List[int]:
    """
    获取时间序列标签中的异常间隔长度列表。

    :param labels: 时间序列标签列表，1 表示异常，0 表示正常。
    :return: 异常间隔长度列表。
    """
    end_pos = np.diff(np.array(labels, dtype=int), append=0) < 0
    return np.diff(np.cumsum(labels)[end_pos], prepend=0)

def best_ratio(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return anomaly_rate
def best_f_score(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return best_f1_score


def best_accuracy(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return accuracy


def best_recall(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return recall


def best_precision(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return precision

def auc_roc(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return metrics.roc_auc_score(actual, predicted)


def auc_pr(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return metrics.average_precision_score(actual, predicted)


def R_AUC_ROC(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    slidingWindow = int(np.median(get_list_anomaly(actual)))
    # slidingWindow = 100
    R_AUC_ROC, R_AUC_PR, _, _, _ = metricor_grader.RangeAUC(
        labels=actual, score=predicted, window=slidingWindow, plot_ROC=True
    )
    return R_AUC_ROC


def R_AUC_PR(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    slidingWindow = int(np.median(get_list_anomaly(actual)))
    # slidingWindow = 100
    R_AUC_ROC, R_AUC_PR, _, _, _ = metricor_grader.RangeAUC(
        labels=actual, score=predicted, window=slidingWindow, plot_ROC=True
    )
    return R_AUC_PR


def VUS_ROC(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    slidingWindow = int(np.median(get_list_anomaly(actual)))
    # slidingWindow = 100

    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(
        actual, predicted, 2 * slidingWindow
    )
    return VUS_ROC


def VUS_PR(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    slidingWindow = int(np.median(get_list_anomaly(actual)))
    # slidingWindow = 100

    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(
        actual, predicted, 2 * slidingWindow
    )
    return VUS_PR


# metrics for label mode
def adjust_predicts(actual: np.ndarray, predicted: np.ndarray, **kwargs) -> np.ndarray:
    """
    调整检测结果
    异常检测算法在一个异常区间检测到某点存在异常，则认为算法检测到整个异常区间的所有异常点
    先从检测到的异常点从后往前调整检测结果，随后再从该点从前往后调整检测结果，直到真实的异常为False
    退出异常状态，结束当前区间的调整

    :param actual: 真实的异常。
    :param predicted: 检测所得的异常。
    :return: 调整后的异常检测结果。
    """
    predicted = predicted.copy()
    anomaly_state = False
    for i in range(len(actual)):
        if actual[i] == 1 and predicted[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if actual[j] == 0:
                    break
                else:
                    if predicted[j] == 0:
                        predicted[j] = 1
            for j in range(i, len(actual)):
                if actual[j] == 0:
                    break
                else:
                    if predicted[j] == 0:
                        predicted[j] = 1
        elif actual[i] == 0:
            anomaly_state = False
        if anomaly_state:
            predicted[i] = 1
    return predicted

def adjust_precision(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    predicted = adjust_predicts(actual, predicted)
    Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
        actual, predicted, zero_division=0
    )
    return Precision[1]


def adjust_recall(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    predicted = adjust_predicts(actual, predicted)
    Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
        actual, predicted, zero_division=0
    )
    return Recall[1]


def adjust_f_score(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    predicted = adjust_predicts(actual, predicted)
    Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
        actual, predicted, zero_division=0
    )
    return F[1]


def adjust_accuracy(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    predicted = adjust_predicts(actual, predicted)
    accuracy = accuracy_score(actual, predicted)
    return accuracy


def precision(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
        actual, predicted, zero_division=0
    )
    return Precision[1]


def recall(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
        actual, predicted, zero_division=0
    )
    return Recall[1]


def f_score(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
        actual, predicted, zero_division=0
    )
    return F[1]


def accuracy(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    accuracy = accuracy_score(actual, predicted)
    return accuracy


def rrecall(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    (
        AUC_ROC,
        Precision,
        Recall,
        F,
        Rrecall,
        ExistenceReward,
        OverlapReward,
        Rprecision,
        RF,
        Precision_at_k,
    ) = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
    return Rrecall


def rprecision(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    (
        AUC_ROC,
        Precision,
        Recall,
        F,
        Rrecall,
        ExistenceReward,
        OverlapReward,
        Rprecision,
        RF,
        Precision_at_k,
    ) = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
    return Rprecision


def rf(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    (
        AUC_ROC,
        Precision,
        Recall,
        F,
        Rrecall,
        ExistenceReward,
        OverlapReward,
        Rprecision,
        RF,
        Precision_at_k,
    ) = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
    return RF


def precision_at_k(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    (
        AUC_ROC,
        Precision,
        Recall,
        F,
        Rrecall,
        ExistenceReward,
        OverlapReward,
        Rprecision,
        RF,
        Precision_at_k,
    ) = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
    return Precision_at_k

def affiliation_f(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    events_pred = convert_vector_to_events(predicted)
    events_label = convert_vector_to_events(actual)
    Trange = (0, len(predicted))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result['precision']
    R = result['recall']
    F = 2 * P * R / (P + R)

    return F
def affiliation_precision(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    events_pred = convert_vector_to_events(predicted)
    events_label = convert_vector_to_events(actual)
    Trange = (0, len(predicted))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result['precision']
    R = result['recall']
    F = 2 * P * R / (P + R)

    return P

def affiliation_recall(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    events_pred = convert_vector_to_events(predicted)
    events_label = convert_vector_to_events(actual)
    Trange = (0, len(predicted))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result['precision']
    R = result['recall']
    F = 2 * P * R / (P + R)

    return R

def calculate(mode: str, actual: np.ndarray, predicted: np.ndarray, scaler: object = None, hist_data: np.ndarray = None) -> dict:
    results = {}
    if mode=="score":
        metrics_list = METRICS_SCORE
    elif mode=="label":
        metrics_list = METRICS_LABEL
    for name in metrics_list:
        try:
            func = globals()[name]
            results[name] = func(
                actual,
                predicted,
                scaler=scaler,
                hist_data=hist_data
            )
        except Exception as e:
            results[name] = np.nan
            print(f"Error computing {name}: {e}")
    return results