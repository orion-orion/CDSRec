# -*- coding: utf-8 -*-
import random
from collections import defaultdict
import numpy as np
from kg_utils import get_neighbours1


def compute_rel_pos(batch_size, seq_len):    
    time_matrix = np.zeros([seq_len, seq_len], dtype=np.int32)
    for i in range(seq_len):
        for j in range(seq_len):
            span = abs(i-j)
            time_matrix[i][j] = span
    return np.stack([time_matrix] * batch_size, axis=0)


def split_A_and_B(predictions, ground_truths, neg_samples, method="TiSASRec", num_items_A=0, A_or_B=None):
    if method == "TiSASRec":
        A_or_B = np.where(ground_truths < num_items_A, 0, 1)
    else:
        predictions = zip(*predictions)
    
    predictions_A, predictions_B = [], []
    grounds_A, grounds_B = [], []
    neg_samples_A, neg_samples_B = [], []
    for item_predictions, ground_truth, item_neg_samples, is_A_or_B in zip(predictions, ground_truths, \
        neg_samples, A_or_B):
        if is_A_or_B == 0:
            if method == "TiSASRec":
                predictions_A.append(item_predictions)
            else:
                predictions_A.append(item_predictions[0])
            grounds_A.append(ground_truth)
            neg_samples_A.append(item_neg_samples)                
        else:
            if method == "TiSASRec":
                predictions_B.append(item_predictions)                
            else:
                predictions_B.append(item_predictions[1])                
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


def stack_with_indexes(*input_matrixs):
    # indexes indicate the indexes of the samples in the batch    
    indexes = np.arange(input_matrixs[0].shape[0])
    indexes = np.expand_dims(indexes, axis=-1)
    new_matrixs = []
    for input_matrix in input_matrixs:
        indexes_matrix = np.repeat(indexes, input_matrix.shape[1], axis=1)
        new_matrixs.append(np.stack([indexes_matrix, input_matrix], axis=-1))
    return new_matrixs


def unpack_tisasrec_sessions(model, sessions, mode="train"):
    is_training = (True if mode == "train" else False)
    if mode == "train":
        seqs, pos_samples, neg_samples = sessions 
        feed_dict = {model.input_seq: seqs, model.time_matrix: compute_rel_pos(seqs.shape[0], seqs.shape[1]), \
                                model.pos_samples: pos_samples, model.neg_samples: neg_samples, model.is_training: is_training}
        return feed_dict
    else:
        seqs, ground_truths, neg_samples = sessions
        feed_dict = {model.input_seq: seqs, model.time_matrix: compute_rel_pos(seqs.shape[0], seqs.shape[1]), model.is_training: is_training}
    return feed_dict, ground_truths, neg_samples


def unpack_conet_sessions(model, sessions, user_ids, pad_int=0, mode="train"):
    if mode == "train":
        items_A, items_B, neg_samples_A, neg_samples_B = sessions
        ui_A, labels_A = get_ui_and_labels(user_ids, items_A, neg_samples_A, pad_int)
        ui_B, labels_B = get_ui_and_labels(user_ids, items_B, neg_samples_B, pad_int)
        ui_A, labels_A, ui_B, labels_B = \
            align_A_and_B(ui_A, labels_A, ui_B, labels_B) 
        ui_A, labels_A = get_ui_and_labels(user_ids, items_A, neg_samples_A, pad_int)
        ui_B, labels_B = get_ui_and_labels(user_ids, items_B, neg_samples_B, pad_int)
        ui_A, labels_A, ui_B, labels_B = \
            align_A_and_B(ui_A, labels_A, ui_B, labels_B)   
        feed_dict = {model.input_A: ui_A, model.label_A: labels_A, \
                model.input_B: ui_B, model.label_B: labels_B}
        return feed_dict
    else:
        grounds_A, grounds_B, neg_samples_A, neg_samples_B = sessions
        ui_A = get_eval_ui(user_ids, grounds_A, neg_samples_A)
        ui_B = get_eval_ui(user_ids, grounds_B, neg_samples_B)
        feed_dict_A, feed_dict_B = {model.input_A: ui_A}, {model.input_B: ui_B}
    return feed_dict_A, feed_dict_B, grounds_A, grounds_B, neg_samples_A, neg_samples_B, ui_A, ui_B


def unpack_pinet_sessions(model, sessions, mode="train"):
    if mode == "train":          
        seqs_A, seqs_B, positions_A, positions_B, lens_A, lens_B, grounds_A, grounds_B = sessions
        positions_A, positions_B = stack_with_indexes(positions_A, positions_B)
        feed_dict = {model.seq_A: seqs_A, model.seq_B: seqs_B, \
            model.pos_A: positions_A, model.pos_B: positions_B, model.len_A: lens_A, model.len_B: lens_B, \
                model.ground_truth_A: grounds_A, model.ground_truth_B: grounds_B}
        return feed_dict  
    else:
        seqs_A, seqs_B, positions_A, positions_B, lens_A, lens_B, A_or_B, ground_truths, neg_samples = sessions
        positions_A, positions_B = stack_with_indexes(positions_A, positions_B)
        feed_dict = {model.seq_A: seqs_A, model.seq_B: seqs_B, model.pos_A: positions_A, model.pos_B: positions_B, \
            model.len_A: lens_A, model.len_B: lens_B, model.ground_truth_A: ground_truths, model.ground_truth_B: ground_truths}
        return feed_dict, A_or_B, ground_truths, neg_samples


def unpack_mifn_sessions(model, sessions, num_items_A, num_items_B, mode="train"):
    from mifn import config    
    seqs_A, seqs_B, positions_A, positions_B, lens_A, lens_B, indexes_A, indexes_B, adjs_1, adjs_2, adjs_3, adjs_4, adjs_5 = sessions[: 13]
    positions_A, positions_B, indexes_A, indexes_B = stack_with_indexes(positions_A, positions_B, indexes_A, indexes_B)
    seqs = [np.concatenate([seq_A, seq_B], axis=0) for seq_A, seq_B in zip(seqs_A, seqs_B)]
    if mode == "train":
        grounds_A, grounds_B = sessions[13: 15]       
        neighbors, nei_indexes_A, nei_indexes_B, nei_in_A, nei_in_B, _, nei_masks_A, nei_masks_B,  nei_masks_L_A, \
        nei_masks_L_B, grounds_A_in_nei, grounds_B_in_nei = get_neighbours1(seqs, lens_A, lens_B, seqs_A, seqs_B, grounds_A, grounds_B, \
            num_items_A, num_items_B, config.num_neighbors)
        grounds_A_in_nei= np.expand_dims(grounds_A_in_nei,axis=1)
        grounds_B_in_nei = np.expand_dims(grounds_B_in_nei,axis=1)
        
        feed_dict = {model.seq_A: seqs_A, model.seq_B: seqs_B, model.pos_A: positions_A, model.pos_B: positions_B,
                     model.len_A: lens_A, model.len_B: lens_B, model.ground_truth_A: grounds_A, model.ground_truth_B: grounds_B,
                     model.ground_A_in_nei: grounds_A_in_nei, model.ground_B_in_nei: grounds_B_in_nei,
                     model.adj_1: adjs_1, model.adj_2: adjs_2, model.adj_3: adjs_3, model.adj_4: adjs_4, model.adj_5: adjs_5,
                     model.neighbors: neighbors,
                     model.index_A: indexes_A, model.index_B: indexes_B,
                     model.nei_index_A: nei_indexes_A, model.nei_index_B: nei_indexes_B,
                     model.nei_A_mask: nei_masks_A, model.nei_B_mask: nei_masks_B,
                     model.nei_L_A_mask: nei_masks_L_A, model.nei_L_T_mask: nei_masks_L_B,
                     model.nei_is_in_A: nei_in_A, model.nei_is_in_B: nei_in_B}
        return feed_dict
    else:
        A_or_B, ground_truths, neg_samples = sessions[13: 16]        
        neighbors, nei_indexes_A, nei_indexes_B, nei_in_A, nei_in_B, _, nei_masks_A, nei_masks_B, nei_masks_L_A, \
        nei_masks_L_T, grounds_in_nei, _ = get_neighbours1(seqs, lens_A, lens_B, seqs_A, seqs_B, ground_truths, ground_truths, \
            num_items_A, num_items_B, config.num_neighbors)
        grounds_in_nei = np.expand_dims(grounds_in_nei, axis=1)

        feed_dict = {model.seq_A: seqs_A, model.seq_B: seqs_B, model.pos_A: positions_A, model.pos_B: positions_B,
                     model.len_A: lens_A, model.len_B: lens_B, model.ground_truth_A: ground_truths, model.ground_truth_B: ground_truths,
                     model.ground_A_in_nei: grounds_in_nei, model.ground_B_in_nei: grounds_in_nei,
                     model.adj_1: adjs_1, model.adj_2: adjs_2, model.adj_3: adjs_3, model.adj_4: adjs_4, model.adj_5: adjs_5,
                     model.neighbors: neighbors,
                     model.index_A: indexes_A, model.index_B: indexes_B,
                     model.nei_index_A: nei_indexes_A, model.nei_index_B: nei_indexes_B,
                     model.nei_A_mask: nei_masks_A, model.nei_B_mask: nei_masks_B,
                     model.nei_L_A_mask: nei_masks_L_A, model.nei_L_T_mask: nei_masks_L_T,
                     model.nei_is_in_A: nei_in_A, model.nei_is_in_B: nei_in_B}
        return feed_dict, A_or_B, ground_truths, neg_samples
    
    
def unpack_sessions(model, sessions, method, *args, mode="train"):
    unpack_functions = {"TiSASRec": unpack_tisasrec_sessions, "CoNet": unpack_conet_sessions, "PINet": unpack_pinet_sessions, \
        "MIFN": unpack_mifn_sessions}
    return unpack_functions[method](model, sessions, *args, mode=mode)