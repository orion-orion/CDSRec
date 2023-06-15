# -*- coding: utf-8 -*-
import random
import numpy as np
from scipy import sparse


def get_neighbours1(sessions, lens_A, lens_B, seqs_A, seqs_B, grounds_A, grounds_B, num_items_A, num_items_B, num_neighbors):
    """this function is used to run as a sample to the model."""
    real_seqs_A = []
    for session in seqs_A:
        seq_A = session[np.where(session > 0)]
        real_seqs_A.append(seq_A)
    real_seqs_B = []
    for session in seqs_B:
        seq_B = session[np.where(session > 0)]
        real_seqs_B.append(seq_B)

    neighbors_all = []
    indexes_A,indexes_B = [],[]
    nei_in_A,nei_in_B = [],[]
    train_ids = []
    nei_masks_A,nei_masks_B = [],[]
    nei_masks_L_A, nei_masks_L_T = [], []
    grounds_A_in_nei,grounds_B_in_nei = [],[]
    for idx in range(len(sessions)):
        nodes = set()
        itemA = set(real_seqs_A[idx])
        nodes.update(itemA)
        itemB = set(real_seqs_B[idx])
        nodes.update(itemB)

        random_num = random.randint(0, 9)
        if random_num <= 1:
            nodes.add(grounds_A[idx])
            nodes.add(grounds_B[idx])
            train_ids.append(idx + 10)
            grounds_A_in_nei.append(1)
            grounds_B_in_nei.append(1)
        elif random_num >= 2 and random_num <= 4:
            nodes.add(grounds_A[idx])
            train_ids.append(idx + 20)
            grounds_A_in_nei.append(1)
            grounds_B_in_nei.append(0)
        elif random_num >= 5 and random_num <= 7:
            nodes.add(grounds_B[idx])
            train_ids.append(idx + 30)
            grounds_A_in_nei.append(0)
            grounds_B_in_nei.append(1)
        else:
            train_ids.append(idx + 40)
            grounds_A_in_nei.append(0)
            grounds_B_in_nei.append(0)

        pool = set(range(1000)).difference(nodes)
        rest = list(random.sample(list(pool), num_neighbors - len(nodes)))
        # print(set(rest))
        nodes.update(set(rest))
        # print(nodes)
        node_indexes = dict(zip(list(nodes), range(len(nodes))))
        neighbors_all.append(list(nodes))

        tmp_nei_in_A, tmp_nei_in_B = [], []
        items_set_A = set(range(num_items_A))
        items_set_B = set(range(num_items_A, num_items_A + num_items_B))
        for item_id in node_indexes.keys():
            if item_id in items_set_A:
                tmp_nei_in_A.append(1)
                tmp_nei_in_B.append(0)
            elif item_id in items_set_B:
                tmp_nei_in_A.append(0)
                tmp_nei_in_B.append(1)
            else:
                tmp_nei_in_A.append(0)
                tmp_nei_in_B.append(0)
        nei_in_A.append(tmp_nei_in_A)
        nei_in_B.append(tmp_nei_in_B)

        tmp_nei_mask_A = np.ones(num_neighbors)
        tmp_nei_mask_A[np.where(np.array(list(nodes)) >= num_items_A)] = 0
        tmp_nei_mask_B = np.ones(num_neighbors)
        tmp_nei_mask_B[np.where(np.array(list(nodes)) < num_items_A)] = 0
        nei_masks_A.append(tmp_nei_mask_A)
        nei_masks_B.append(tmp_nei_mask_B)

        temp_A, temp_T = [],[]
        for item in sessions[idx]:
            if item < num_items_A:
                temp_A.append(tmp_nei_mask_A)
                temp_T.append(tmp_nei_mask_B)
            else:
                temp_A.append(tmp_nei_mask_B)
                temp_T.append(tmp_nei_mask_A)
        nei_masks_L_A.append(temp_A)
        nei_masks_L_T.append(temp_T)

        zeros_1 = np.zeros(num_neighbors)
        indexes_1 = np.where(np.array(list(nodes)) <= 99)[0]
        ent_ids = np.array(list(nodes))[indexes_1]
        item_ids = []
        for item_id in ent_ids:
            item_ids.append(item_id)
        zeros_1[indexes_1] = item_ids
        indexes_A.append(zeros_1)

        zeros_2 = np.zeros(num_neighbors)
        indexes_2 = []
        for item_id in nodes:
            if item_id >= num_items_A and item_id < num_items_A + num_items_B:
                indexes_2.append(node_indexes[item_id])
        ent_ids = np.array(list(nodes))[indexes_2]
        item_ids = []
        for item_id in ent_ids:
            item_ids.append(item_id - num_items_A)
        zeros_2[indexes_2] = item_ids
        indexes_B.append(zeros_2)

    neighbors_all = np.array(neighbors_all)
    indexes = np.arange(len(sessions))
    indexes = np.expand_dims(indexes, axis=-1)
    indexes_p = np.repeat(indexes, num_neighbors, axis=1)
    nei_indexes_A = np.stack([indexes_p, np.array(indexes_A)], axis=-1)
    nei_indexes_B = np.stack([indexes_p, np.array(indexes_B)], axis=-1)
    nei_masks_A = np.array(nei_masks_A)
    nei_masks_B = np.array(nei_masks_B)
    nei_masks_L_A = np.array(nei_masks_L_A)
    nei_masks_L_T = np.array(nei_masks_L_T)

    return neighbors_all, nei_indexes_A, nei_indexes_B, nei_in_A, nei_in_B, train_ids, \
        nei_masks_A, nei_masks_B, nei_masks_L_A, nei_masks_L_T, grounds_A_in_nei, grounds_B_in_nei