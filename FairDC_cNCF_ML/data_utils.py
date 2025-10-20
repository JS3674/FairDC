import numpy as np 
import scipy.sparse as sp 

import torch.utils.data as data
import pdb
from torch.autograd import Variable
import torch
import math
import random
import os


class BPRData(data.Dataset):
    def __init__(self, train_dict=None, num_item=0, num_ng=1, is_training=None, data_set_count=0):
        super().__init__()
        self.num_item = num_item
        self.train_dict = train_dict
        self.num_ng = num_ng
        self.is_training = is_training
        self.data_set_count = data_set_count 
        self.set_all_item = set(range(num_item))  
        self.features_fill = []
        self.base_path = ''
        if self.is_training == 0:
            self.base_path = '../data/ml_reload/train/'
        elif self.is_training == 1:
            self.base_path = '../data/ml_reload/test/'
        else:
            self.base_path = '../data/ml_reload/val/'
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
            print(f'reload {"train" if self.is_training == 0 else "test" if self.is_training == 1 else "val"} has model save path created')
        else:
            print(f'reload {"train" if self.is_training == 0 else "test" if self.is_training == 1 else "val"} has model save path')
        self.save_dataid = self.countPath(self.base_path) 
        self.seed = 2022
        self.cache_data = {}

    def ng_sample(self):
        if self.save_dataid in self.cache_data:
            self.features_fill = self.cache_data[self.save_dataid]
            self.seed += 1
            return
        if self.save_dataid > (self.seed - 2022):
            rand_id = self.seed - 2022
            save_path = self.base_path + str(rand_id) + '.npy'
            if os.path.exists(save_path):
                self.features_fill = np.load(save_path)
                self.cache_data[self.save_dataid] = self.features_fill
                self.seed += 1
                return
        self.features_fill = []
        np.random.seed(self.seed)
        self.seed += 1 
        for user_id in self.train_dict:
            positive_list = self.train_dict[user_id] 
            for item_i in positive_list:
                for t in range(self.num_ng):
                    item_j = np.random.randint(self.num_item) 
                    while item_j in positive_list:
                        item_j = np.random.randint(self.num_item)
                    self.features_fill.append([user_id, item_i, item_j]) 
        if self.save_dataid == 0 or (self.seed % 5 == 0):
            save_path = self.base_path + str(self.save_dataid) + '.npy'
            print("Saving new data to:", save_path)
            np.save(save_path, self.features_fill)
            self.cache_data[self.save_dataid] = self.features_fill
        self.save_dataid += 1

    def countPath(self, base_path):
        tmp = 0 
        for item in os.listdir(base_path):
            if os.path.isfile(os.path.join(base_path, item)):
                tmp += 1
        return tmp

    def __len__(self):  
        if len(self.features_fill) == 0:
            self.ng_sample()
        return len(self.features_fill)

    def __getitem__(self, idx):
        if idx >= len(self.features_fill):
            raise IndexError(f"Index {idx} out of bounds for dataset with size {len(self.features_fill)}")
        features = self.features_fill
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] 
        return user, item_i, item_j


class generate_adj:
    def __init__(self, training_user_set, training_item_set, user_num, item_num):
        self.training_user_set = training_user_set
        self.training_item_set = training_item_set
        self.user_num = user_num
        self.item_num = item_num 
    def readD(self, set_matrix, num_):
        user_d = [] 
        for i in range(num_): 
            len_set = 1.0 / (len(set_matrix[i]) + 1)  
            user_d.append(len_set)
        return user_d 
    def readTrainSparseMatrix(self, set_matrix, is_user, u_d, i_d):
        user_items_matrix_i = []
        user_items_matrix_v = [] 
        if is_user:
            d_i = u_d
            d_j = i_d
        else:
            d_i = i_d
            d_j = u_d
        for i in set_matrix: 
            len_set = len(set_matrix[i])
            for pair_v in set_matrix[i]:
                r_v, j = pair_v
                user_items_matrix_i.append([i, j])
                user_items_matrix_v.append(r_v * 1. / len_set) 
        user_items_matrix_i = torch.cuda.LongTensor(user_items_matrix_i)
        user_items_matrix_v = torch.cuda.FloatTensor(user_items_matrix_v)
        return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v) 
    def generate_pos(self): 
        u_d = self.readD(self.training_user_set, self.user_num)
        i_d = self.readD(self.training_item_set, self.item_num)
        sparse_u_i = self.readTrainSparseMatrix(self.training_user_set, True, u_d, i_d)
        sparse_i_u = self.readTrainSparseMatrix(self.training_item_set, False, u_d, i_d)
        return sparse_u_i, sparse_i_u, u_d, i_d
