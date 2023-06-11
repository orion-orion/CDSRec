import numpy as np


def compute_rel_pos(batch_size, seq_len):    
    time_matrix = np.zeros([seq_len, seq_len], dtype=np.int32)
    for i in range(seq_len):
        for j in range(seq_len):
            span = abs(i-j)
            time_matrix[i][j] = span
    return np.stack([time_matrix] * batch_size, axis=0)


def split_A_and_B(logits, ground_truth, num_items_A):
    A_or_B = np.where(ground_truth < num_items_A, 0, 1)
    A_logits, B_logits = [], []
    A_ground_truth, B_ground_truth = [], []
    for logits, ground_truth, is_A_or_B in zip(logits, ground_truth, A_or_B):
        if is_A_or_B == 0:
            A_logits.append(logits)
            A_ground_truth.append(ground_truth)
        else:
            B_logits.append(logits)
            B_ground_truth.append(ground_truth)
    return np.array(A_logits), np.array(B_logits), np.array(A_ground_truth), np.array(B_ground_truth)