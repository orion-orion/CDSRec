# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pickle
from kg_utils import get_subgraph3, gen_mask, gen_index, load_kg


class CDSDataSet(object):
    data_dir = "data"
    prep_dir = "prep_data"
    num_test_neg = 999 # The number of negative samples to test all methods
    # Hyperparameters of CoNet. 
    # `num_train_neg`: The number of negative samples to train CoNet
    # `num_test_neg`: The number of negative samples to test CoNet
    conet_params = {"num_train_neg": 1, "num_test_neg": 99}
    # Hyperparameters of MIFN. `max_H`: hop layer H, `max_nei`: The number of neighbor entities to use
    mifn_params = {"max_H": 2, "max_nei": 200}
    def __init__(self, dataset, method="PINet", mode="train", pad_int=0, load_prep=True):
        assert(method in ["TiSASRec","CoNet", "PINet", "MIFN"])
        assert(mode in ["train", "valid", "test"])

        self.dataset = dataset
        self.method = method
        self.mode = mode
        self.pad_int = pad_int
        self.dataset_dir = os.path.join(self.data_dir, dataset)
        self.user_ids, self.sessions, self.seq_lens_A, self.max_seq_len_A, self.seq_lens_B, self.max_seq_len_B, \
            self.seq_lens, self.max_seq_len, self.num_users, self.num_items_A, self.num_items_B \
            = self.read_data(self.dataset_dir) 

        self.prep_sessions = self.preprocess_data(self.dataset_dir, load_prep)
       
    def read_data(self, dataset_dir):
        with open(os.path.join(dataset_dir, "num_items.txt"), "rt") as f:
            num_items_A = int(f.readline())
            num_items_B = int(f.readline())

        with open(os.path.join(dataset_dir, "num_users.txt"), "rt") as f:
            num_users = int(f.readline())
                    
        seq_lens, seq_lens_A, seq_lens_B = [], [], []
        with open(os.path.join(self.data_dir, self.dataset, "%s_data.txt" % self.mode), "rt") as f:
            user_ids, sessions = [], []
            for line in f.readlines():
                session = []
                line = line.strip().split("\t")
                # Note that the ground truth is included when computing the sequence lengths of domain A and domain B
                seq_len_A, seq_len_B = 0, 0
                for item in line[1:]: # Start from index 1 to exclude user ID
                    item = int(item)
                    if item < num_items_A:
                        seq_len_A += 1
                    else:
                        seq_len_B += 1
                    session.append(item)
                user_ids.append(int(line[0]))
                sessions.append(session)
                seq_lens_A.append(seq_len_A)
                seq_lens_B.append(seq_len_B)
                seq_lens.append(seq_len_A + seq_len_B)
        max_seq_len_A = max(seq_lens_A)
        max_seq_len_B = max(seq_lens_B)
        max_seq_len = max(seq_lens)
        print("Successfully load %s %s data!" % (self.dataset, self.mode))        

        return user_ids, sessions, seq_lens_A, max_seq_len_A, seq_lens_B, max_seq_len_B, \
            seq_lens, max_seq_len, num_users, num_items_A, num_items_B

    def preprocess_data(self, dataset_dir, load_prep):
        self.prep_data_path = os.path.join(dataset_dir, self.prep_dir, "%s_%s_data.pkl" % (self.method, self.mode))
        if os.path.exists(self.prep_data_path) and load_prep:
            with open(os.path.join(self.prep_data_path), "rb") as f:
                prep_sessions = pickle.load(f)
            print("Successfully load preprocessed %s %s data!" % (self.dataset, self.mode))
        else:
            if self.method == "TiSASRec":
                prep_sessions = self.preprocess_tisasrec(self.sessions, mode=self.mode)
            elif self.method == "CoNet":
                prep_sessions = self.preprocess_conet(self.sessions, mode=self.mode)
            elif self.method == "PINet":
                prep_sessions = self.preprocess_pinet(self.sessions, mode=self.mode)
            elif self.method == "MIFN":
                prep_sessions = self.preprocess_mifn(self.sessions, mode=self.mode)
            with open(self.prep_data_path, "wb") as f:
                pickle.dump(prep_sessions, f)
            print("Successfully preprocess %s %s data!" % (self.dataset, self.mode))    

        return prep_sessions
     
    @ staticmethod
    def random_neg(left, right, excl): # [left, right)
        sample = np.random.randint(left, right)
        while sample in excl:
            sample = np.random.randint(left, right)
        return sample
     
    def preprocess_tisasrec(self, sessions, mode="train"):
        prep_sessions = []
        for idx, session in enumerate(sessions):              
            temp = []
            if mode == "train":
                padded_seq = session + [self.pad_int] * (self.max_seq_len - self.seq_lens[idx])
                temp.append(padded_seq[:-1])  # Items for input
                pos_samples = np.zeros([self.max_seq_len - 1], dtype=np.int32)
                neg_samples = np.zeros([self.max_seq_len - 1], dtype=np.int32)
                for idx in range(self.max_seq_len - 1):
                    nxt = padded_seq[idx + 1]
                    pos_samples[idx] = nxt  # The ID of the next item is used as ground truth
                    neg_samples[idx] = self.random_neg(0, self.num_items_A + self.num_items_B, excl=session) 
                temp.append(pos_samples)
                temp.append(neg_samples)
            else:
                items_input, ground_truth = session[:-1], session[-1]
                temp.append(items_input + [self.pad_int] * (self.max_seq_len - self.seq_lens[idx]))
                temp.append(ground_truth)   
                neg_samples = []
                for _ in range(self.num_test_neg):
                    # Negative samples must be generated in the corresponding domain,
                    # otherwise the results will become too optimistic                    
                    if ground_truth < self.num_items_A:
                        neg_sample = self.random_neg(0, self.num_items_A, [ground_truth])
                    else:
                        neg_sample = self.random_neg(self.num_items_A, self.num_items_A + self.num_items_B, excl=[ground_truth])
                    neg_samples.append(neg_sample)
                temp.append(neg_samples)
            prep_sessions.append(temp)     
        return prep_sessions 
        
    def preprocess_conet(self, sessions, mode="train"):    
        prep_sessions = []
        for idx, session in enumerate(sessions):  
            temp = []
            session_A = []
            session_B = []
            for item in session:
                if item < self.num_items_A:
                    session_A.append(item)
                else:
                    session_B.append(item - self.num_items_A)
            if mode == "train":
                temp.append(session_A + [self.pad_int] * (self.max_seq_len_A - self.seq_lens_A[idx]))
                temp.append(session_B + [self.pad_int] * (self.max_seq_len_B - self.seq_lens_B[idx]))                
                num_neg = self.conet_params["num_train_neg"]
                neg_samples_A = []
                for item_A in session_A:
                    item_neg_samples = []
                    for _ in range(num_neg):
                        # Negative samples must be generated in the corresponding domain
                        neg_sample = self.random_neg(0, self.num_items_A, [item_A])
                        item_neg_samples.append(neg_sample)            
                    neg_samples_A.append(item_neg_samples)
                temp.append(neg_samples_A + [[0] * num_neg] * (self.max_seq_len_A - self.seq_lens_A[idx]))
                neg_samples_B = []
                for item_B in session_B:
                    item_neg_samples = []
                    for _ in range(self.conet_params["num_train_neg"]):
                        # Negative samples must be generated in the corresponding domain
                        neg_sample = self.random_neg(0, self.num_items_B, [item_B - self.num_items_A])
                        item_neg_samples.append(neg_sample)            
                    neg_samples_B.append(item_neg_samples)
                temp.append(neg_samples_B + [[0] * num_neg] * (self.max_seq_len_B - self.seq_lens_B[idx]))                                 
            else:
                num_neg = self.conet_params["num_test_neg"]
                ground_truth_A, ground_truth_B = session_A[-1], session_B[-1]
                temp.append(ground_truth_A)
                temp.append(ground_truth_B)
                neg_samples_A, neg_samples_B = [], []
                for _ in range(num_neg):
                    # Negative samples must be generated in the corresponding domain
                    neg_sample_A = self.random_neg(0, self.num_items_A, [ground_truth_A])
                    neg_sample_B = self.random_neg(0, self.num_items_B, [ground_truth_B])
                    neg_samples_A.append(neg_sample_A)   
                    neg_samples_B.append(neg_sample_B)
                temp.append(neg_samples_A)
                temp.append(neg_samples_B)
            prep_sessions.append(temp)
        return prep_sessions 

    @staticmethod    
    def leave_out_ground_A_and_ground_B(session, num_items_A):
        new_session = []
        last_A, last_B = -1, -1
        for idx, item_id in enumerate(reversed(session)):
            if not last_A == -1 and not last_B == -1:
                new_session = session[: (len(session) - idx)] + list(reversed(new_session)) + [last_A, last_B]
                break    
            if last_A == -1 and item_id < num_items_A:
                last_A = item_id
                continue
            if last_B == -1 and item_id >= num_items_A:
                last_B = item_id
                continue
            else:
                new_session.append(item_id)   
        return new_session      
        
    def preprocess_pinet(self, sessions, mode="train"):
        prep_sessions = []
        for idx, session in enumerate(sessions):
            temp = []
            if mode == "train":
                session = self.leave_out_ground_A_and_ground_B(session, self.num_items_A)
                items_input, ground_truth_A, ground_truth_B = session[:-2], session[-2], session[-1]
            else:
                items_input, ground_truth = session[:-1], session[-1]
                # If the ground truth is in domain A
                if ground_truth < self.num_items_A:
                    # Exclude the ground truth A when computing the sequence length of domain A
                    self.seq_lens_A[idx] -= 1 
                else: # If the ground truth is in domain B
                    # Exclude the ground truth B when computing the sequence length of domain B
                    self.seq_lens_B[idx] -= 1
            
            seq_A, seq_B, pos_A, pos_B = [], [], [], []
            len_A, len_B = 0, 0
            for item in items_input:
                if item < self.num_items_A:
                    seq_A.append(item)
                    pos_A.append(len_B)
                    len_A += 1
                else:
                    seq_B.append(item - self.num_items_A)
                    pos_B.append(len_A)
                    len_B += 1
                    
            temp.append(seq_A + [self.pad_int] * (self.max_seq_len_A - self.seq_lens_A[idx]))
            temp.append(seq_B + [self.pad_int] * (self.max_seq_len_B - self.seq_lens_B[idx]))
            temp.append(pos_A + [0] * (self.max_seq_len_A - self.seq_lens_A[idx]))
            temp.append(pos_B + [0] * (self.max_seq_len_B - self.seq_lens_B[idx]))
            temp.append(len_A)
            temp.append(len_B)
            
            if mode == "train":
                temp.append(ground_truth_A) # The ground truth for domain A
                temp.append(ground_truth_B - self.num_items_A) # The ground truth for domain B
            else:
                # The ground truth is in domain A or domain B
                is_A_or_B = (0 if ground_truth < self.num_items_A else 1) 
                temp.append(is_A_or_B)
                if is_A_or_B == 0:
                    temp.append(ground_truth)   
                else:
                    temp.append(ground_truth - self.num_items_A)
                neg_samples = []
                for _ in range(self.num_test_neg):
                    # Negative samples must be generated in the corresponding domain
                    if is_A_or_B == 0:
                        neg_sample = self.random_neg(0, self.num_items_A, excl=[ground_truth])
                    else:
                        neg_sample = self.random_neg(0, self.num_items_B, excl=[ground_truth - self.num_items_A])
                    neg_samples.append(neg_sample)
                temp.append(neg_samples)
            prep_sessions.append(temp)
        return prep_sessions
    
    def preprocess_mifn(self, dataset, mode="train"):
        kg_file_path = os.path.join(self.data_dir, self.dataset, "knowledge_graph/%s_kg.npy" % mode)
        kg = load_kg(kg_file_path)
        
        sessions = []
        for idx, session in enumerate(dataset):
            if mode == "train":
                session = self.leave_out_ground_A_and_ground_B(session, self.num_items_A)
                items_input, ground_truth_A, ground_truth_B = session[:-2], session[-2], session[-1]
            else:
                items_input, ground_truth = session[:-1], session[-1]
                
            temp = []

            seq_A, seq_B = [],[]
            pos_A, pos_B = [],[]
            index_A, index_B = [],[]
            len_A, len_B = 0,0
            for idx, item in enumerate(items_input):  # All the items except ground truth
                if item < self.num_items_A: # The item is in domain A
                    seq_A.append(item)
                    pos_A.append(len_B)
                    len_A += 1
                    index_A.append(idx)
                else: # The item is in domain B
                    seq_B.append(item)
                    pos_B.append(len_A)
                    len_B += 1
                    index_B.append(idx)
                
            _, _, _, _, _, nei_index_dict = get_subgraph3(items_input, kg, self.mifn_params["max_H"], self.mifn_params["max_nei"], self.num_items_A)
                
            nei_mask_A, nei_mask_B, _, _ = gen_mask(nei_index_dict, items_input, self.max_nei, self.num_items_A)
            nei_index_A, nei_index_B, nei_is_in_A, nei_is_in_B = gen_index(nei_index_dict, self.max_nei, self.num_items_A, self.num_items_B)
            temp.append(seq_A + [self.pad_int] * (self.max_seq_len_A - self.seq_lens_A[idx]))
            temp.append(seq_B + [self.pad_int] * (self.max_seq_len_B - self.seq_lens_B[idx]))
            temp.append(pos_A + [0] * (self.max_seq_len_A - self.seq_lens_A[idx]))
            temp.append(pos_B + [0] * (self.max_seq_len_B - self.seq_lens_B[idx]))
            temp.append(len_A)
            temp.append(len_B)

            if mode == "train":
                temp.append(ground_truth_A)
                temp.append(ground_truth_B - self.num_items_A)
            else:
                is_A_or_B = ( 0 if ground_truth < self.num_items_A else 1) # The ground truth is in domain A or domain B
                temp.append(is_A_or_B)
                if is_A_or_B == 0:
                    temp.append(ground_truth)   
                else:
                    temp.append(ground_truth - self.num_items_A)
                neg_samples = []
                for _ in range(self.num_test_neg):
                    # Negative samples must be generated in the corresponding domain
                    if is_A_or_B == 0:
                        neg_sample = self.random_neg(0, self.num_items_A, excl=[ground_truth])
                    else:
                        neg_sample = self.random_neg(0, self.num_items_B, excl=[ground_truth - self.num_items_A])
                    neg_samples.append(neg_sample)   
                temp.append(neg_samples)

            temp.append(len_A + len_B)
            temp.append(index_A + [0] * (self.max_seq_len_A - self.seq_lens_A[idx]))
            temp.append(index_B + [0] * (self.max_seq_len_B - self.seq_lens_B[idx]))
            temp.append(np.random.randint(0, 2, (self.max_nei, self.max_nei)))
            temp.append(np.random.randint(0, 2, (self.max_nei, self.max_nei)))
            temp.append(np.random.randint(0, 2, (self.max_nei, self.max_nei)))
            temp.append(np.random.randint(0, 2, (self.max_nei, self.max_nei)))
            temp.append(np.random.randint(0, 2,( self.max_nei, self.max_nei)))
            temp.append(list(nei_index_dict.keys())) # Neighbors
            temp.append(nei_index_A)
            temp.append(nei_index_B)
            temp.append(nei_mask_A)
            temp.append(nei_mask_B)
            temp.append(np.array([]))
            temp.append(np.array([]))
            temp.append(nei_is_in_A)
            temp.append(nei_is_in_B)
            
            if mode == "train":
                ground_A_in_nei = (1 if ground_truth_A in nei_index_dict.keys() else 0)
                ground_B_in_nei = (1 if ground_truth_B - self.num_items_A in nei_index_dict.keys() else 0)
                temp.append(ground_A_in_nei)
                temp.append(ground_B_in_nei)    
            else:
                train = (1 if ground_truth in nei_index_dict.keys() else 0)
                temp.append(train)

            sessions.append(temp)
        return sessions
    
    def __len__(self):
        return len(self.prep_sessions)

    def __getitem__(self, idx):
        user_ids = self.user_ids[idx]
        session = self.prep_sessions[idx]
        return user_ids, session
    
    def __setitem__(self, idx, value): # To support shuffle operation
        self.user_ids[idx] = value[0]
        self.prep_sessions[idx] = value[1]
    

class Dataloader(object):
    def __init__(self, dataset, batch_size=128, shuffle=True):
        self.dataset = dataset
        self.num_items_A, self.num_items_B = dataset.num_items_A, dataset.num_items_B
        self.batch_size = batch_size
        self.shuffle = True
        
        if shuffle == True:
            random.shuffle(self.dataset)

        if len(self.dataset) % batch_size == 0:
            self.num_batch = len(dataset)//batch_size
        else:
            self.num_batch = len(dataset)//batch_size + 1
        
    def __iter__(self):  
        for batch_idx in range(self.num_batch):
            start_idx =  batch_idx * self.batch_size
            batch_user_ids, batch_sessions = self.dataset[start_idx: start_idx + self.batch_size]
            batch_sessions = list(zip(*batch_sessions))
            yield np.array(batch_user_ids), tuple(np.array(x) for x in batch_sessions)