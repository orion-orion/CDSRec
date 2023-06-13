# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict


def compute_rel_pos(batch_size, seq_len):    
    time_matrix = np.zeros([seq_len, seq_len], dtype=np.int32)
    for i in range(seq_len):
        for j in range(seq_len):
            span = abs(i-j)
            time_matrix[i][j] = span
    return np.stack([time_matrix] * batch_size, axis=0)


def split_A_and_B(predictions, ground_truths, neg_samples, num_items_A):
    A_or_B = np.where(ground_truths < num_items_A, 0, 1)
    predictions_A, predictions_B = [], []
    grounds_A, grounds_B = [], []
    neg_samples_A, neg_samples_B = [], []
    for item_predictions, ground_truth, item_neg_samples, is_A_or_B in zip(predictions, ground_truths, \
        neg_samples, A_or_B):
        if is_A_or_B == 0:
            predictions_A.append(item_predictions)
            grounds_A.append(ground_truth)
            neg_samples_A.append(item_neg_samples)
        else:
            predictions_B.append(item_predictions)
            grounds_B.append(ground_truth)
            neg_samples_B.append(item_neg_samples)
    return np.array(predictions_A), np.array(predictions_B), np.array(grounds_A), np.array(grounds_B), \
        np.array(neg_samples_A), np.array(neg_samples_B)


def get_ui_and_labels(user_ids, items, neg_samples, pad_int):
    ui, labels = [], []
    for user_id, user_items, user_neg_samples in zip(user_ids, items, neg_samples):
        for item, item_neg_samples in zip(user_items, user_neg_samples):
            if item == pad_int:
                break
            ui.append([user_id, item])
            labels.append([1, 0])
            for neg_sample in item_neg_samples:
                ui.append([user_id, neg_sample])
                labels.append([0, 1])        
    return np.array(ui), np.array(labels)
    

def align_A_and_B(ui_A, labels_A, ui_B, labels_B):
    inc_ui, inc_labels = [], []
    num_samples_A, num_samples_B = labels_A.shape[0], labels_B.shape[0]
    if num_samples_A > num_samples_B:
        for _ in range(num_samples_A - num_samples_B):
            sample_idx = np.random.randint(num_samples_B)
            inc_ui.append(ui_B[sample_idx])
            inc_labels.append(labels_B[sample_idx])
        ui_B = np.concatenate([ui_B, np.array(inc_ui)], axis=0)
        labels_B = np.concatenate([labels_B, np.array(inc_labels)], axis=0)
    elif num_samples_A < num_samples_B:
        for _ in range(num_samples_B - num_samples_A):
            sample_idx = np.random.randint(num_samples_A)
            inc_ui.append(ui_A[sample_idx])
            inc_labels.append(labels_A[sample_idx])
        ui_A = np.concatenate([ui_A, np.array(inc_ui)], axis=0)
        labels_A = np.concatenate([labels_A, np.array(inc_labels)], axis=0)    
    return ui_A, labels_A, ui_B, labels_B
            

def get_eval_ui(user_ids, ground_truths, neg_samples):
    test_ui = []
    for user_id, ground_truth, item_neg_samples in zip(user_ids, ground_truths, neg_samples):
        test_ui.append([user_id, ground_truth])
        for neg_sample in item_neg_samples:
            test_ui.append([user_id, neg_sample])
    return np.array(test_ui)


def get_predictions(ui, ui_preds):
    predictions = defaultdict(dict)
    for ui, pred in zip(ui, ui_preds):
        user, item = ui[0], ui[1]
        predictions[user][item] = pred[0]
    return predictions