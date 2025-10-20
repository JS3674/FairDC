import torch
import torch.nn as nn 
import os
import numpy as np
import math
import sys 
import pdb
from collections import defaultdict
import time
from shutil import copyfile 
import gc  
from scipy import sparse

saved_model_path = './best_model/' 
dataset_base_path='../data/ml/' 
users_features=np.load(dataset_base_path+'/users_features.npy')
users_features = users_features[:,0]
train_dict, train_dict_count = np.load(dataset_base_path + '/train.npy', allow_pickle=True)
user_num = len(users_features)
item_num = max(max(items) for items in train_dict.values() if items) + 1
factor_num=64
batch_size=2048*8

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--runid', type=str, default = "none") 
args = parser.parse_args()
run_id=args.runid

path_save_model_base=saved_model_path+run_id
if (os.path.exists(path_save_model_base)):
    print("Model:",run_id)
else:
    print('Model_Save_Path does not exist',path_save_model_base)

training_user_set,train_dict_count = np.load(dataset_base_path+'/train.npy',allow_pickle=True)
testing_user_set,test_dict_count = np.load(dataset_base_path+'/test.npy',allow_pickle=True)

########################### TESTING ##################################### 
np.seterr(divide='ignore',invalid='ignore')
def get_real_data(idx_pre_s,idx_pre_e):
    all_row = []
    all_col = []
    all_data =[]
    test_row = []
    test_col = []
    test_data =[]
    user_id_test =[]
    one_data_len = idx_pre_e - idx_pre_s
    for u_i in range(idx_pre_s,idx_pre_e):   
        user_id_test.append(u_i) 
        if u_i not in testing_user_set or u_i not in training_user_set:
            continue
        else:
            all_one = list(set(training_user_set[u_i])|set(testing_user_set[u_i]))
        test_one = list(testing_user_set[u_i])
        all_len = len(all_one)
        test_len = len(test_one)
        all_row = all_row + [u_i-idx_pre_s]*all_len
        all_col = all_col + all_one
        all_data = all_data + [1]*all_len
        test_row = test_row + [u_i-idx_pre_s]*test_len
        test_col = test_col + test_one
        test_data = test_data + [1]*test_len
    all_idx = sparse.csr_matrix((all_data, (all_row, all_col)), shape=(one_data_len, item_num))    
    test_idx = sparse.csr_matrix((test_data, (test_row, test_col)), shape=(one_data_len, item_num))   
    return all_idx.toarray(),test_idx.toarray()
 
    
    
def largest_indices(ary, n):
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def all_metrics(top_k): 
    ndcg_max=[0]*top_k
    ndcg_each=[0]*top_k
    temp_max_ndcg=0
    for i_topK in range(top_k):
        ndcg_each[i_topK] = 1.0/math.log2(i_topK+2)
        temp_max_ndcg+=1.0/math.log2(i_topK+2)
        ndcg_max[i_topK]=temp_max_ndcg
    print('--------test processing, Top K:',top_k,'-------')
    count, best_hr = 0, 0
    for epoch in ['best']: 
        PATH_model_u=path_save_model_base+'/user_emb.npy'
        user_e = np.load(PATH_model_u)
        PATH_model_i=path_save_model_base+'/item_emb.npy'
        item_e = np.load(PATH_model_i) 
        rank_topk=np.zeros([user_num,top_k],dtype=int)
        HR, NDCG = 0, 0
        HR_num, NDCG_num = 0, 0
        test_start_time = time.time()
        idx_all = []
        len_split = int(user_num/3000)
        for i_g in range(len_split):
            idx_pre_s = i_g*3000
            idx_pre_e = (i_g+1)*3000
            idx_all.append([idx_pre_s,idx_pre_e])
        idx_all.append([idx_pre_e,user_e.shape[0]])
        i_pre=0
        for idx_pre_s,idx_pre_e in idx_all:  
            i_pre+=1
            s_time = time.time()
            pre_all = np.matmul(user_e[idx_pre_s:idx_pre_e,:], item_e.T) 
            all_data,test_data = get_real_data(idx_pre_s,idx_pre_e)
            len_test_data =test_data.sum(-1) 
            x1=np.where(all_data>0,-1000,pre_all)  
            x2=np.where(test_data>0,pre_all,-1000) 
            pre_rank = np.concatenate((x1,x2),axis=-1)
            del x1, x2,pre_all,all_data,test_data
            gc.collect() 
            indices_part = np.argpartition(pre_rank, -top_k)[:,-top_k:] 
            values_part = np.partition(pre_rank, -top_k)[:,-top_k:]
            indices_sort = np.argsort(-values_part)
            indice1=np.take_along_axis(indices_part, indices_sort, axis=-1) 
            del pre_rank,indices_part,values_part,indices_sort
            gc.collect() 
            indice2 = np.where(indice1>item_num,1,0)  
            rank_topk[idx_pre_s:idx_pre_e,:]=indice1  
            max_hr = len_test_data
            len_large_ndcg = np.minimum(len_test_data, top_k) - 1
            idcg = np.zeros_like(len_large_ndcg, dtype=float)
            for i, n_rel in enumerate(len_large_ndcg):
                if n_rel >= 0:
                    idcg[i] = ndcg_max[n_rel]
            dcg = np.zeros((indice2.shape[0]), dtype=float)
            for i in range(top_k):
                pos_discount = 1.0 / math.log2(i + 2)
                dcg += indice2[:, i] * pos_discount
            ndcg_scores = np.zeros_like(dcg)
            valid_idx = (idcg > 0)
            ndcg_scores[valid_idx] = dcg[valid_idx] / idcg[valid_idx]
            valid_users = (len_test_data > 0)
            ndcg_t = ndcg_scores[valid_users]
            hr_topK = indice2.sum(-1)
            hr_t1 = hr_topK/max_hr
            hr_t = hr_t1[~np.isnan(hr_t1)] 
            e4_time = time.time() - s_time 
            HR+=hr_t.sum(-1)
            NDCG+=ndcg_t.sum(-1)
            HR_num+=hr_t.shape[0]
            NDCG_num+=ndcg_t.shape[0] 
        hr_test=round(HR/HR_num,4)
        ndcg_test=round(NDCG/NDCG_num,4)    
        elapsed_time = time.time() - test_start_time     
        if i_pre<len_split:
            str_print_evl="part user test-"
        else:
            str_print_evl=""
        str_print_evl+=" Top K:"+str(top_k)+" test"+" HR:"+str(hr_test)+' NDCG:'+str(ndcg_test) 
        DP_res = dict()
        EO_res = dict()
        for v_id in range(item_num*2):
            DP_res[v_id]=[]
            EO_res[v_id]=[]
        for u_id in range(user_num):
            label_gender = users_features[u_id]
            for v_id in rank_topk[u_id]:
                DP_res[v_id].append(label_gender)
                if v_id >item_num:
                    EO_res[v_id].append(label_gender)
        DP=[] 
        male_num=users_features.sum()
        female_num=(1-users_features).sum() 
        for v_id in DP_res:
            one_data = np.array(DP_res[v_id])
            if len(one_data)<1:
                continue
            res2_v = one_data.sum()/len(one_data)-(1-one_data).sum()/len(one_data)
            DP.append(np.abs(res2_v))
        DP_test =round(np.array(DP).mean() ,6)
        EO=[] 
        for v_id in EO_res:
            one_data = np.array(EO_res[v_id])
            if len(one_data)<1:
                continue
            res2_v = one_data.sum()/len(one_data)-(1-one_data).sum()/len(one_data)
            EO.append(np.abs(res2_v)) 
        EO_test =round(np.array(EO).mean(),6)
        elapsed_time = time.time() - test_start_time
        str_print_evl+= '\t DP:'+str(DP_test)+' EO:'+str(EO_test)
        print(str_print_evl)

def gender_metrics(top_k):
    ndcg_max=[0]*top_k
    ndcg_each=[0]*top_k
    temp_max_ndcg=0
    for i_topK in range(top_k):
        ndcg_each[i_topK] = 1.0/math.log2(i_topK+2)
        temp_max_ndcg+=1.0/math.log2(i_topK+2)
        ndcg_max[i_topK]=temp_max_ndcg
        
    for epoch in ['best']: 
        PATH_model_u = path_save_model_base+'/user_emb.npy'
        user_e = np.load(PATH_model_u)
        PATH_model_i = path_save_model_base+'/item_emb.npy'
        item_e = np.load(PATH_model_i) 
        
        rank_topk = np.zeros([user_num,top_k],dtype=int)
        
        HR_male, NDCG_male = 0, 0
        HR_female, NDCG_female = 0, 0
        HR_num_male, NDCG_num_male = 0, 0
        HR_num_female, NDCG_num_female = 0, 0
        test_start_time = time.time()
        idx_all = []
        len_split = int(user_num/3000)
        for i_g in range(len_split):
            idx_pre_s = i_g*3000
            idx_pre_e = (i_g+1)*3000
            idx_all.append([idx_pre_s,idx_pre_e])
        idx_all.append([idx_pre_e,user_e.shape[0]])
        for idx_pre_s,idx_pre_e in idx_all:  
            pre_all = np.matmul(user_e[idx_pre_s:idx_pre_e,:], item_e.T) 
            all_data,test_data = get_real_data(idx_pre_s,idx_pre_e)
            len_test_data = test_data.sum(-1) 
            x1 = np.where(all_data>0,-1000,pre_all)  
            x2 = np.where(test_data>0,pre_all,-1000) 
            pre_rank = np.concatenate((x1,x2),axis=-1)
            del x1, x2, pre_all, all_data, test_data
            gc.collect() 
            indices_part = np.argpartition(pre_rank, -top_k)[:,-top_k:] 
            values_part = np.partition(pre_rank, -top_k)[:,-top_k:]
            indices_sort = np.argsort(-values_part)
            indice1 = np.take_along_axis(indices_part, indices_sort, axis=-1) 
            del pre_rank,indices_part,values_part,indices_sort
            gc.collect() 
            indice2 = np.where(indice1>item_num,1,0)  
            rank_topk[idx_pre_s:idx_pre_e,:] = indice1  
            max_hr = len_test_data
            len_large_ndcg = np.minimum(len_test_data, top_k) - 1
            idcg = np.zeros_like(len_large_ndcg, dtype=float)
            for i, n_rel in enumerate(len_large_ndcg):
                if n_rel >= 0:
                    idcg[i] = ndcg_max[n_rel]
            dcg = np.zeros((indice2.shape[0]), dtype=float)
            for i in range(top_k):
                pos_discount = 1.0 / math.log2(i + 2)
                dcg += indice2[:, i] * pos_discount
            ndcg_scores = np.zeros_like(dcg)
            valid_idx = (idcg > 0)
            ndcg_scores[valid_idx] = dcg[valid_idx] / idcg[valid_idx]
            hr_topK = indice2.sum(-1)
            current_users_gender = users_features[idx_pre_s:idx_pre_e]
            male_mask = current_users_gender == 0
            hr_t1_male = hr_topK[male_mask]/max_hr[male_mask]
            hr_t_male = hr_t1_male[~np.isnan(hr_t1_male)]
            valid_male_users = male_mask & (len_test_data > 0)
            ndcg_t_male = ndcg_scores[valid_male_users]
            HR_male += hr_t_male.sum(-1)
            NDCG_male += ndcg_t_male.sum(-1)
            HR_num_male += hr_t_male.shape[0]
            NDCG_num_male += ndcg_t_male.shape[0]
            female_mask = current_users_gender == 1
            hr_t1_female = hr_topK[female_mask]/max_hr[female_mask]
            hr_t_female = hr_t1_female[~np.isnan(hr_t1_female)]
            valid_female_users = female_mask & (len_test_data > 0)
            ndcg_t_female = ndcg_scores[valid_female_users]
            HR_female += hr_t_female.sum(-1)
            NDCG_female += ndcg_t_female.sum(-1)
            HR_num_female += hr_t_female.shape[0]
            NDCG_num_female += ndcg_t_female.shape[0]
        hr_male = round(HR_male/HR_num_male, 4) if HR_num_male > 0 else 0
        ndcg_male = round(NDCG_male/NDCG_num_male, 4) if NDCG_num_male > 0 else 0
        hr_female = round(HR_female/HR_num_female, 4) if HR_num_female > 0 else 0
        ndcg_female = round(NDCG_female/NDCG_num_female, 4) if NDCG_num_female > 0 else 0
        print(f"Male Users - HR@{top_k}: {hr_male}, NDCG@{top_k}: {ndcg_male}")
        print(f"Female Users - HR@{top_k}: {hr_female}, NDCG@{top_k}: {ndcg_female}")

for top_k_v in [10,20,30,40]:
    all_metrics(top_k_v)
    # gender_metrics(top_k_v)