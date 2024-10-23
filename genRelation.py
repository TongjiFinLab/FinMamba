from torch_geometric.nn import GATConv, TopKPooling
from torch_geometric.utils import dense_to_sparse

from sklearn.preprocessing import MinMaxScaler
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

stock = "sp"
fea = pd.read_pickle(f"data/{stock}fea.pkl")
fea = fea.reset_index()
fea.rename(columns={'kdcode':'instrument','dt':'datetime'},inplace=True)
fea.rename(columns={'Company':'instrument','Date':'datetime'},inplace=True)
stock_num = fea['instrument'].nunique()
day_num = fea['datetime'].nunique()
fea_num = fea.shape[1] - 2
fea = fea.drop(columns=['datetime', 'instrument'])
fea = fea.values
x = fea.reshape(day_num, stock_num, fea_num)
x = torch.FloatTensor(x).to(device)
print(x.shape)


prev_date_num = 20
# nasdaq100/S&P500
feature_cols = ['high','low','close','open','volume']
# CSI300/CSI500
# feature_cols = ['high','low','close','open','volume', 'turnover']

def cal_pccs(x, y, n):
    sum_xy = torch.sum(x * y)
    sum_x = torch.sum(x)
    sum_y = torch.sum(y)
    sum_x2 = torch.sum(x * x)
    sum_y2 = torch.sum(y * y)
    pcc = (n * sum_xy - sum_x * sum_y) / torch.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y) + 1e-6)
    return pcc

def cal_spearman(x, y, n):
    rank_x = torch.argsort(torch.argsort(x))
    rank_y = torch.argsort(torch.argsort(y))

    sum_rank_xy = torch.sum(rank_x * rank_y)
    sum_rank_x = torch.sum(rank_x)
    sum_rank_y = torch.sum(rank_y)
    sum_rank_x2 = torch.sum(rank_x * rank_x)
    sum_rank_y2 = torch.sum(rank_y * rank_y)
    
    spearman_corr = (n * sum_rank_xy - sum_rank_x * sum_rank_y) / torch.sqrt(
        (n * sum_rank_x2 - sum_rank_x * sum_rank_x) * (n * sum_rank_y2 - sum_rank_y * sum_rank_y) + 1e-6
    )
    return spearman_corr

def calculate_relations(xs, yss, n, method="spearman"):
    result = torch.zeros(stock_num).to(device)
    for i in range(stock_num):
        ys = yss[i]
        tmp_res = []
        for pos, x in enumerate(xs):
            y = ys[pos]
            if method == "spearman":
                tmp_res.append(cal_spearman(x, y, n))
            else:
                tmp_res.append(cal_pccs(x, y, n))
        tmp_res = torch.stack(tmp_res)
        result[i] = torch.mean(tmp_res)
    return result

def stock_cor_matrix(x, prev_date_num, day, processes=1):
    end_data = day
    start_data = max(0, end_data - (prev_date_num - 1))
    q = x.clone()[start_data:end_data+1]
    # q [len, stock_num, fea_num]
    q = q.permute(1, 2, 0)
    # [stock_num, fea_num, len]
    relation = torch.zeros([stock_num, stock_num]).to(device)
    for i in range(stock_num):
        relation[i, :] = calculate_relations(q[i], q, prev_date_num)
        relation[i,i] = 1
    relation[torch.isnan(relation)] = 0
    return relation


n = 0
end = day_num
for i in tqdm(range(n, end)):
    r = stock_cor_matrix(x, prev_date_num, i).cpu().detach().numpy()
    with open(f'{stock}_stock_relation/day{i}.pkl', 'wb') as f:
        pickle.dump(r, f)
