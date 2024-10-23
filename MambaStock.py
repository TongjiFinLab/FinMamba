from torch_geometric.nn import GATConv, TopKPooling, GCNConv
from torch_geometric.utils import dense_to_sparse

from sklearn.preprocessing import MinMaxScaler
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

import math

import random
import torch.backends.cudnn as cudnn

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(2024)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

stock = "nasdaq"
fea = pd.read_pickle(f"data/{stock}fea.pkl")
fea = fea.reset_index()
fea.rename(columns={'kdcode':'instrument','dt':'datetime'},inplace=True)
fea.rename(columns={'Company':'instrument','Date':'datetime'},inplace=True)
stock_num = fea['instrument'].nunique()
day_num = fea['datetime'].nunique()
fea_num = fea.shape[1] - 2

train = fea[(fea['datetime'] >= '2018-01-01') & (fea['datetime'] <= '2021-12-31')]
valid = fea[(fea['datetime'] >= '2022-01-01') & (fea['datetime'] <= '2022-12-31')]
test = fea[(fea['datetime'] >= '2023-01-01') & (fea['datetime'] <= '2023-12-31')]

train_day_num = train['datetime'].nunique()
valid_day_num = valid['datetime'].nunique()
test_day_num = test['datetime'].nunique()

train = train.drop(columns=['datetime', 'instrument']).values
valid = valid.drop(columns=['datetime', 'instrument']).values
test = test.drop(columns=['datetime', 'instrument']).values

train = train.reshape(train_day_num, stock_num, fea_num)
train = torch.FloatTensor(train).to(device)
valid = valid.reshape(valid_day_num, stock_num, fea_num)
valid = torch.FloatTensor(valid).to(device)
test = test.reshape(test_day_num, stock_num, fea_num)
test = torch.FloatTensor(test).to(device)


lab = pd.read_pickle(f"data/{stock}lab.pkl")
lab = lab.reset_index()
lab.rename(columns={'kdcode':'instrument','dt':'datetime'},inplace=True)
lab.rename(columns={'Company':'instrument','Date':'datetime'},inplace=True)

y_train = lab[(lab['datetime'] >= '2018-01-01') & (lab['datetime'] <= '2021-12-31')]
y_valid = lab[(lab['datetime'] >= '2022-01-01') & (lab['datetime'] <= '2022-12-31')]
y_test = lab[(lab['datetime'] >= '2023-01-01') & (lab['datetime'] <= '2023-12-31')]

y_train = y_train.drop(columns=['datetime', 'instrument']).values
y_valid = y_valid.drop(columns=['datetime', 'instrument']).values
y_test = y_test.drop(columns=['datetime', 'instrument']).values
y_train = y_train.reshape(train_day_num, stock_num)
y_train = torch.FloatTensor(y_train).to(device)
y_valid = y_valid.reshape(valid_day_num, stock_num)
y_valid = torch.FloatTensor(y_valid).to(device)
y_test = y_test.reshape(test_day_num, stock_num)
y_test = torch.FloatTensor(y_test).to(device)


market = torch.cat((torch.mean(train, dim=1), 
                    torch.mean(valid, dim=1), 
                    torch.mean(test, dim=1)), dim=0)


class MarketGuideInception(nn.Module):
    def __init__(self, input_dim, kernel_sizes=[4,10,20]):
        super(MarketGuideInception, self).__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=1, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(output_size=input_dim)
        self.fc = nn.Linear(len(kernel_sizes) * input_dim, 1)
        self.dim = input_dim
        self.init_sparsity = 0.2

    def forward(self, market_index):
        market_index = market_index.unsqueeze(0).transpose(1, 2)
        branch_outputs = [self.pool(branch(market_index)) for branch in self.branches]
        combined_out = torch.cat(branch_outputs, dim=1)
        combined_out = combined_out.view(-1, len(self.branches) * self.dim)

        market_index = self.fc(combined_out)        
        top_k_ratio = self.init_sparsity / (1 + torch.exp(-market_index))
        return top_k_ratio


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_heads=2):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        
        self.gat_layers = nn.ModuleList([GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)])
        
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True))
        
        self.gat_layers.append(GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False))
    
    def forward(self, x, edge_index):
        for layer in range(self.num_layers):
            x = self.gat_layers[layer](x, edge_index)
            if layer != self.num_layers - 1:
                x = F.gelu(x)
        return x

from mamba_ssm import Mamba

tmp_outcome_vis = []

class mamba(nn.Module):

    def __init__(self, input_size=6, hidden_size=[64,128], output_size=6, num_head=2, seq_len=20):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_head = num_head
        self.pred_len = 1
        
        self.in_layer  = nn.ModuleList([nn.Linear(self.input_size, self.hidden_size[i]) for i in range(self.num_head)])
        
        self.out_layer = nn.ModuleList([nn.Linear(self.hidden_size[i], self.output_size) for i in range(self.num_head)])

        self.mamba = nn.ModuleList([Mamba(
            d_model = self.hidden_size[i], # Model dimension d_model
            d_state = 128,         # SSM state expansion factor
            d_conv = 2,           # Local convolution width
            expand = 1,           # Block expansion factor
        ) for i in range(self.num_head)])

        self.dropout = nn.Dropout(0.1)

        self.gen_score = nn.Linear(self.output_size * self.num_head, 1)


    def forward(self, x, is_traing = 1):
        # x [batch_size, seq_len, num_fea]
        out = []
        for i in range(self.num_head):
            tmp = self.in_layer[i](x)
            tmp = tmp + self.mamba[i](tmp)
            # [batch_size, seq_len, hidden_size]
            tmp = self.out_layer[i](tmp)
            tmp = tmp.permute(0, 2, 1)
            # [batch_size, output_size, seq_len]
            tmp = tmp[:,:,-1]
            tmp = self.dropout(tmp)
            out.append(tmp)
        
        out = torch.cat(out, dim=1)
        score = self.gen_score(out)

        return score


def topk_matrix(adj_matrix, k):
    n = adj_matrix.size(0)
    
    triu_indices = torch.triu_indices(n, n, offset=1).to(device)
    triu_values = adj_matrix[triu_indices[0], triu_indices[1]]
    topk_values, topk_indices = torch.topk(triu_values, k)
    
    adj_matrix[triu_indices[0][topk_indices], triu_indices[1][topk_indices]] = topk_values
    adj_matrix = adj_matrix + adj_matrix.T
    adj_matrix.fill_diagonal_(1)
    return adj_matrix


def GIBLoss(x, z):
    z_mean = torch.mean(z, dim=0)
    x_mean = torch.mean(x, dim=0)
    z_var = torch.var(z, dim=0)
    x_var = torch.var(x, dim=0)
        
    gib_loss = torch.sum((z_mean - x_mean)**2) / torch.sum(z_var + x_var)
    gib_loss = torch.clamp(gib_loss, max=0.5)
    return gib_loss



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, output_size=16, num_layers=2, num_heads=2, dropout=0.1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_size)

        self.pe = PositionalEncoding(input_size, dropout)
        layer = nn.TransformerEncoderLayer(
            nhead=num_heads, dropout=dropout, d_model=hidden_size, dim_feedforward=hidden_size * 4
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.gen_score = nn.Linear(output_size, 1)

    def forward(self, x, is_traing=1):
        # input [batch_size, seq_len, num_fea]
        x = x.permute(1, 0, 2).contiguous()
        # x [seq_len, batch_size, num_fea]
        x = self.pe(x)
        # x [seq_len, batch_size, num_fea]
        x = self.input_proj(x)
        # x [seq_len, batch_size, hidden_size]
        out = self.encoder(x)
        # out [seq_len, batch_size, hidden_size]
        # out[-1] [batch_size, hidden_size]
        out = self.output_proj(out)
        out = self.gen_score(out[-1])

        return out


class mambastock(nn.Module):
    def __init__(self, input_dim=6, hidden_channels=32, out_channels=6, hidden_size=[64, 64, 32], output_size=16,
                 seq_len=20):
        super(mambastock, self).__init__()
        self.MG = MarketGuideInception(input_dim=input_dim)
        self.AGG = GAT(in_channels=input_dim, hidden_channels=hidden_channels, out_channels=out_channels)
        self.MAMBA = mamba(input_size=input_dim + out_channels, hidden_size=hidden_size, output_size=output_size, seq_len=seq_len)
        # self.MAMBA = mamba(input_size=input_dim, hidden_size=hidden_size, output_size=output_size, seq_len=seq_len)
        # self.linear = nn.Linear(input_dim + out_channels, 1)
        self.transformer = Transformer(input_size=input_dim + out_channels, hidden_size=128, output_size=output_size)

        self.seq_len = seq_len
        self.input_dim = input_dim
    
    def forward(self, x, relation, market_index, window, is_traing = 1):
        rate = self.MG(market_index)
        top_k = int(rate.item() * stock_num)
        relation = topk_matrix(relation, top_k)
        edge_index, _ = dense_to_sparse(relation)
        xr = self.AGG(x, edge_index)
        loss = GIBLoss(x, xr)
        input = torch.concat((xr, x), dim=1)
        # input = x
        # [stock_num, fea_num + out_dim]
        window.append(input)

        assert len(window) == self.seq_len
        
        ts_input = torch.stack(window).permute(1, 0, 2)
        # [stock_num, seq_len, feature]
        score = self.MAMBA(ts_input, is_traing).squeeze(1)
        # score = self.transformer(ts_input, is_traing).squeeze(1)
        # score = self.linear(ts_input).squeeze(1)[:,-1].squeeze(1)
        
        window.pop(0)

        return score, window, loss

class HingeMSELoss(nn.Module):
    def __init__(self, hinge_weight=1.0, mse_weight=1.0, stock_num=100):
        super(HingeMSELoss, self).__init__()
        self.hinge_weight = hinge_weight
        self.mse_weight = mse_weight
        self.mse_loss = nn.MSELoss()
        self.stock_num = stock_num

    def forward(self, scores, targets):
        mse_loss = self.mse_loss(scores, targets)

        scores_diff = scores.unsqueeze(2) - scores.unsqueeze(1)
        targets_diff = targets.unsqueeze(2) - targets.unsqueeze(1)
    
        hinge_loss = F.relu(- (scores_diff * targets_diff))
        
        hinge_loss = hinge_loss.sum(dim=[1, 2]) / self.stock_num
        hinge_loss = hinge_loss.mean()

        combined_loss = self.hinge_weight * hinge_loss + self.mse_weight * mse_loss
        
        return combined_loss

train_epochs = 5
seq_len = 20
batch_size = 16

model = mambastock(input_dim=fea_num, out_channels=fea_num, seq_len=seq_len).to(device)
# criterion = torch.nn.MSELoss()
criterion = HingeMSELoss(hinge_weight=3.0, mse_weight=1.0, stock_num=stock_num)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-7)

window = []

patience = 10
best_val_loss = float('inf')
early_stop_counter = 0

p = np.load(f"data/{stock}_industry_relationship.npy")
p = torch.tensor(p).to(device)

for epoch in range(train_epochs):
    print(f"========== epoch {epoch} ==========")

    model.train()

    window = []
    for i in range(seq_len - 1):
        window.append(torch.concat((train[i], train[i]), dim=1))
        # window.append(train[i])
    
    epoch_loss = 0.0
    n_batches = 0
    for d in range(seq_len - 1, train_day_num, batch_size):
        loss_gib = 0.0
        count = 0
        scores = []
        for i in range(d, min(d + batch_size, train_day_num)):
            assert len(window) == seq_len - 1
            with open(f'{stock}_stock_relation/day{i}.pkl', 'rb') as f:
                r = pickle.load(f)
                r = torch.FloatTensor(r).to(device)
                r = p * r
            score, window, l_gib = model(train[i], r, market[i - seq_len + 1 : i + 1], window)
            loss_gib += l_gib
            # score [stock_num]
            scores.append(score.unsqueeze(0))
            count += 1

        scores = torch.cat(scores, dim=0)
        loss = criterion(scores, y_train[d:min(d + batch_size, train_day_num)]) + loss_gib / count
        epoch_loss += loss.item()
        if n_batches % 10 == 0:
            print(f"batch{n_batches} loss: ", loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        window = [window[i].detach() for i in range(seq_len - 1)]
        n_batches += 1

    print("train loss: ", epoch_loss / n_batches)


    model.eval()
    with torch.no_grad():
        window = []
        for i in range(train_day_num - (seq_len - 1), train_day_num):
            window.append(torch.concat((train[i], train[i]), dim=1))
            # window.append(train[i])

        scores = []
        for i in range(0, valid_day_num):
            assert len(window) == seq_len - 1
            with open(f'{stock}_stock_relation/day{i + train_day_num}.pkl', 'rb') as f:
                r = pickle.load(f)
                r = torch.FloatTensor(r).to(device)
                r = p * r
            score, window, _ = model(valid[i], r, market[i - seq_len + 1 + train_day_num : i + 1 + train_day_num], window)
            # score [stock_num]
            scores.append(score.unsqueeze(0))
    
    scores = torch.cat(scores, dim=0)
    val_loss = criterion(scores, y_valid)
    print("valid loss: ", val_loss.item())

    if val_loss < best_val_loss:# or early_stop_counter == train_epochs:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

model.load_state_dict(torch.load('best_model.pth'))


with torch.no_grad():
    window = []
    for i in range(valid_day_num - (seq_len - 1), valid_day_num):
        window.append(torch.concat((valid[i], valid[i]), dim=1))
        # window.append(valid[i])
    scores = []
    for i in range(0, test_day_num):
        assert len(window) == seq_len - 1
        with open(f'{stock}_stock_relation/day{i + train_day_num + valid_day_num}.pkl', 'rb') as f:
           r = pickle.load(f)
           r = torch.FloatTensor(r).to(device)
           r = p * r
        score, window, _ = model(test[i], r, 
                              market[i - seq_len + 1 + train_day_num + valid_day_num: i + 1 + train_day_num + valid_day_num], window, 
                              is_traing = 0)
        # score [stock_num]
        scores.append(score.unsqueeze(0))
    
    scores = torch.cat(scores, dim=0)
    print(scores.shape)
    loss = criterion(scores, y_test)
    print("test loss: ", loss.item())
    np_scores = scores.cpu().detach().numpy()
    df_scores = pd.DataFrame(np_scores)
    df_scores.to_csv('scores.csv', index=False, header=False)


lab = pd.read_pickle(f"data/{stock}lab.pkl")
lab = lab.reset_index()
lab.rename(columns={'kdcode':'instrument','dt':'datetime'},inplace=True)
lab.rename(columns={'Company':'instrument','Date':'datetime'},inplace=True)
lab = lab[lab['datetime'] >= "2023-01-01"]
lab = lab.reset_index(drop=True)

l = test_day_num

for i in range(lab.shape[0]):
    lab.loc[i, 'label'] = df_scores[int(i/l)][i%l]

lab.to_csv("pred.csv")
print(lab.shape)
print("finished!")
