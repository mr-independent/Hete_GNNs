# -*- coding: utf-8 -*-
# file: ian.py
# author: lijiecheng <mr_independent@sina.com>
# Copyright (C) 2021. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GAT, GCNConv, GATv2Conv
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
import scipy.sparse as sp
import numpy


class Hete_GNNs(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(Hete_GNNs, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, rnn_type='GRU')
        self.lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, rnn_type='GRU')
        self.attention_aspect = Attention(opt.hidden_dim, n_head=32, score_function='bi_linear')
        self.attention_context = Attention(opt.hidden_dim, n_head=32, score_function='bi_linear')
        self.gat2v = GATv2Conv(opt.hidden_dim, opt.hidden_dim, heads=16, concat=False)
        self.fc = nn.Linear(3 * opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs

        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)

        # text_len + 5
        text_len = torch.add(text_len, 5)

        context = self.embed(text_indices)
        aspect = self.embed(aspect_indices)
        context, (_, _) = self.lstm_context(context, text_len.cpu())
        aspect, (_, _) = self.lstm_aspect(aspect, aspect_len.cpu())

        # mask, aspect position
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)

        adj_pos = adj[:, :, -5:]
        adj_sum = torch.sum(adj_pos, dim=2).squeeze(1)
        context_mask = self.mask_pos(context, aspect_double_idx, adj_sum)

        # adj to coo format, Take advantage of block_diag
        adj_chunk = adj.reshape((-1, adj.shape[-1]))
        adj_chunk = adj_chunk.chunk(adj.shape[0], dim=0)
        adj_block_diag = torch.block_diag(*adj_chunk)
        edge_index_temp = sp.coo_matrix(adj_block_diag.cpu())
        edge_index_numpy = numpy.vstack((edge_index_temp.row, edge_index_temp.col))
        edge_index_block_diag = torch.LongTensor(edge_index_numpy).cuda()

        context_block_diag = context.reshape((-1, context.shape[2]))

        # GATv2
        x_i = F.leaky_relu(self.gat2v(context_block_diag, edge_index_block_diag))

        # Reconstructed back to the original structure
        x = torch.stack(torch.chunk(x_i, context.shape[0], dim=0), dim=0)

        # mask
        x = self.mask(x, aspect_double_idx)  # mask1
        alpha_mat = torch.matmul(x.cuda(), context_mask.transpose(1, 2))  # mask2
        # alpha_mat = torch.matmul(x.cuda(), context.transpose(1, 2))    # no mask2

        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        gta_final = torch.matmul(alpha, context_mask).squeeze(1)  # mask2
        # gta_final = torch.matmul(alpha, context).squeeze(1)    # no mask2

        aspect_len = torch.as_tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        aspect_pool = torch.sum(aspect, dim=1)
        aspect_pool = torch.div(aspect_pool, aspect_len.view(aspect_len.size(0), 1))

        text_raw_len = torch.as_tensor(text_len, dtype=torch.float).to(self.opt.device)
        context_pool = torch.sum(context, dim=1)
        context_pool = torch.div(context_pool, text_raw_len.view(text_raw_len.size(0), 1))

        aspect_final, _ = self.attention_aspect(aspect, context_pool)
        aspect_final = aspect_final.squeeze(dim=1)
        context_final, _ = self.attention_context(context, aspect_pool)
        context_final = context_final.squeeze(dim=1)

        x_final = torch.cat([aspect_final, context_final, gta_final], dim=-1)
        output = self.fc(x_final)
        return output

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
        x = x.cuda()
        return mask * x

    def mask_pos(self, x, aspect_double_idx, adj_sum):
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
            for k in range(len(adj_sum[i]) - 5):
                if adj_sum[i][k] != 0:
                    mask[i][k] = 1
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        x = x.cuda()
        return mask * x