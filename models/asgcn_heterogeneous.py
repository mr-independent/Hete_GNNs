
import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
import torch.nn as nn
from layers.dynamic_rnn import DynamicLSTM
from torch_geometric.nn import SGConv, GCNConv, GATConv
from layers.squeeze_embedding import SqueezeEmbedding
from layers.attention import Attention, NoQueryAttention
from layers.point_wise_feed_forward import PositionwiseFeedForward
import scipy.sparse as sp
import numpy

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class ASGCN_HETEROGENEOUS(torch.nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGCN_HETEROGENEOUS, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        self.attn_k = Attention(opt.embed_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.attn_q = Attention(opt.embed_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        # self.gc1 = SGConv(
        #     opt.hidden_dim, opt.hidden_dim, K=2, add_self_loops=False)
        # self.gc2 = SGConv(
        #     opt.hidden_dim, opt.hidden_dim, K=2, add_self_loops=False)
        # self.gc1 = GCNConv(opt.hidden_dim,  opt.hidden_dim, add_self_loops=False)
        # self.gc2 = GCNConv(opt.hidden_dim, opt.hidden_dim, add_self_loops=False)

        self.gc1 = GATConv(opt.hidden_dim,  opt.hidden_dim, heads=1, add_self_loops=False)
        self.gc2 = GATConv(opt.hidden_dim,  opt.hidden_dim, heads=1, add_self_loops=False)

        # self.gc1 = GraphConvolution(opt.hidden_dim,  opt.hidden_dim)
        # self.gc2 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)

        self.fc = nn.Linear(3* opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        # print("seq_len:", seq_len, "text_len:", text_len, "aspect_len:", aspect_len)
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(1 - (aspect_double_idx[i, 0] - j) / context_len)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                weight[i].append(1 - (j - aspect_double_idx[i, 1]) / context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight * x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask * x

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        # text
        text_len = torch.add(text_len, 5)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)

        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.squeeze_embedding(text, text_len)
        aspect = self.embed(aspect_indices)
        aspect = self.squeeze_embedding(aspect, aspect_len)

        h_text, h_text_score = self.attn_k(text, text)
        h_text = self.ffn_c(h_text)
        h_aspect, h_aspect_score = self.attn_q(text, aspect)
        h_aspect = self.ffn_t(h_aspect)

        text_len = torch.tensor(text_len, dtype=torch.float).to(self.opt.device)
        aspect_len = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)

        h_text_mean = torch.div(torch.sum(h_text, dim=1), text_len.view(text_len.size(0), 1))
        h_aspect_mean = torch.div(torch.sum(h_aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))

        ######################  GCNConv添加测试 ########################
        x = torch.zeros(h_text.shape)
        for i in range(len(adj)):
            edge_index_temp = sp.coo_matrix(adj[i])
            edge_index_numpy = numpy.vstack((edge_index_temp.row, edge_index_temp.col))
            edge_index = torch.LongTensor(edge_index_numpy)
            x_i = F.relu(self.gc1(h_text[i], edge_index))
            x_i = F.relu(self.gc2(x_i, edge_index))
            x[i] = x_i
        ######################  测试结束 ########################
        # x = F.relu(self.gc1(h_text, adj))
        # x = F.relu(self.gc2(x, adj))

        x_mean = torch.div(torch.sum(x, dim=1), text_len.view(x.size(0), 1))
        x = torch.cat((x_mean, h_text_mean, h_aspect_mean), dim=-1)
        output = self.fc(x)
        return output, h_text_score, h_aspect_score


