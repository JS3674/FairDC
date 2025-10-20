# -- coding:UTF-8
import numpy as np 
import scipy.sparse as sp 
import torch.utils.data as data
from torch.autograd import Variable
import torch
import math
import random
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BPRData(data.Dataset):
    def __init__(self, train_dict=None, num_item=0, num_ng=1, is_training=None, data_set_count=0):
        super(BPRData, self).__init__()

        self.num_item = num_item
        self.train_dict = train_dict
        self.num_ng = num_ng
        self.is_training = is_training
        self.data_set_count = data_set_count 
        self.set_all_item = set(range(num_item)) 
        
        self.base_path = '../data/ml_reload/' + ('train/' if self.is_training == 0 
                                                else 'test/' if self.is_training == 1 
                                                else 'val/')
        
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
            logging.info(f'Created directory: {self.base_path}')
        
        self.save_dataid = self.countPath(self.base_path) 
        self.seed = 2022
        
        self.cache_data = {}
        
        self.save_frequency = 5 
        self.max_cache_size = 10  
        
    def ng_sample(self):
        if self.save_dataid in self.cache_data:
            logging.debug(f'Using cached data for id: {self.save_dataid}')
            self.features_fill = self.cache_data[self.save_dataid]
            self.seed += 1
            return
            
        rand_id = self.seed - 2022
        save_path = os.path.join(self.base_path, f'{rand_id}.npy')
        if os.path.exists(save_path):
            try:
                logging.debug(f'Loading data from file: {save_path}')
                self.features_fill = np.load(save_path)
                self.update_cache(self.save_dataid, self.features_fill)
                self.seed += 1
                return
            except Exception as e:
                logging.warning(f'Error loading file {save_path}: {e}')

        logging.info('Generating new negative samples...')
        self.features_fill = []
        np.random.seed(self.seed)
        self.seed += 1

        for user_id, positive_list in self.train_dict.items():
            negative_candidates = list(self.set_all_item - set(positive_list))
            
            for item_i in positive_list:
                neg_samples = np.random.choice(
                    negative_candidates, 
                    size=self.num_ng, 
                    replace=True
                )
                self.features_fill.extend(
                    [user_id, item_i, item_j] for item_j in neg_samples
                )

        self.features_fill = np.array(self.features_fill)
        
        if self.save_dataid == 0 or (self.seed % self.save_frequency == 0):
            try:
                logging.info(f'Saving data to: {save_path}')
                np.save(save_path, self.features_fill)
                self.update_cache(self.save_dataid, self.features_fill)
            except Exception as e:
                logging.error(f'Error saving data: {e}')
        
        self.save_dataid += 1

    def update_cache(self, data_id, data):
        if len(self.cache_data) >= self.max_cache_size:
            # 移除最早的缓存
            oldest_key = next(iter(self.cache_data))
            del self.cache_data[oldest_key]
        self.cache_data[data_id] = data

    def clear_cache(self):
        self.cache_data.clear()
        logging.info('Cache cleared')

    def countPath(self, base_path):
        return sum(1 for item in os.listdir(base_path) 
                  if os.path.isfile(os.path.join(base_path, item)))
           
    def __len__(self):  
        return self.num_ng * self.data_set_count

    def __getitem__(self, idx):
        features = self.features_fill
        return features[idx][0], features[idx][1], features[idx][2]


class generate_adj():
    def __init__(self, training_user_set, training_item_set, user_num, item_num):
        self.training_user_set = training_user_set
        self.training_item_set = training_item_set
        self.user_num = user_num
        self.item_num = item_num 

    def readD(self, set_matrix, num_):
        d = np.ones(num_) 
        for i, items in set_matrix.items():
            d[i] = 1.0 / (len(items) + 1)
        return d.tolist()

    def readTrainSparseMatrix(self, set_matrix, is_user, u_d, i_d):
        matrix_i = []
        matrix_v = []

        if is_user:
            d_i, d_j = u_d, i_d
            matrix_i.append([self.user_num-1, self.item_num-1])
        else:
            d_i, d_j = i_d, u_d
            matrix_i.append([self.item_num-1, self.user_num-1])
        matrix_v.append(0)

        for i, items in set_matrix.items():
            for j in items:
                matrix_i.append([i, j])
                matrix_v.append(d_i[i] * d_j[j])

        matrix_i = torch.cuda.LongTensor(matrix_i)
        matrix_v = torch.cuda.FloatTensor(matrix_v)
        
        return torch.sparse.FloatTensor(matrix_i.t(), matrix_v)
    
    def generate_pos(self): 
        u_d = self.readD(self.training_user_set, self.user_num)
        i_d = self.readD(self.training_item_set, self.item_num)
        
        sparse_u_i = self.readTrainSparseMatrix(self.training_user_set, True, u_d, i_d)
        sparse_i_u = self.readTrainSparseMatrix(self.training_item_set, False, u_d, i_d)
        
        return sparse_u_i, sparse_i_u, u_d, i_d