# -*- coding: utf-8 -*-

import os
import math
import argparse
import random
import numpy
import torch
import torch.nn as nn
from bucket_iterator import BucketIterator
from sklearn import metrics
from data_utils import ABSADatesetReader
from models import LSTM,  ASGCN
from models import AEN_GloVe, AEN_BERT, CrossEntropyLoss_LSR
from models import ASMGCN_GloVe
from models import BERT_SPC
from models import ASGCN_HETEROGENEOUS
from models import ASGAT_IAN
from models import IAN
from models import Hete_GNNs
from models import AOA
from models import TD_LSTM, ATAE_LSTM
from models import MemNet
from models import RAM
from models import TNet_LF
from models import MGAN

import time
import logging
import os, sys
from time import strftime, localtime

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class Instructor:
    def __init__(self, opt):
        self.opt = opt

        # absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim)
        absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim,\
                                         model_name= opt.model_name, max_seq_len=opt.max_seq_len,\
                                         pretrained_bert_name=opt.pretrained_bert_name)

        self.train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=opt.batch_size,\
                                                sort_key= 'text_indices', shuffle=True)
        self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=opt.batch_size,\
                                                sort_key= 'text_indices', shuffle=False)
        if 'bert' in opt.model_name:
            self.model = opt.model_class(absa_dataset.bert, opt).to(opt.device)
        else:
            self.model = opt.model_class(absa_dataset.embedding_matrix, opt).to(opt.device)
        self._print_args()
        self.global_f1 = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer):
        max_test_acc = 0
        max_test_f1 = 0
        global_step = 0
        continue_not_increase = 0
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            increase_flag = False
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)

                # outputs, h_text_score, h_aspect_score = self.model(inputs)
                outputs = self.model(inputs)
                # print('outputs:', outputs)
                # print('h_text_score:', h_text_score) #attention评分
                # print('h_aspect_score:', h_aspect_score) #aspect_score评分

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    test_acc, test_f1 = self._evaluate_acc_f1()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                    if test_f1 > max_test_f1:
                        increase_flag = True
                        max_test_f1 = test_f1
                        if self.opt.save and test_f1 > self.global_f1:
                            self.global_f1 = test_f1
                            torch.save(self.model.state_dict(),
                                       'state_dict/' + self.opt.model_name + '_' + self.opt.dataset + '.pkl')
                            print('>>> best model saved.')
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}'.format(loss.item(), train_acc,
                                                                                                test_acc, test_f1))
            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= 5:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0
        return max_test_acc, max_test_f1

    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(opt.device)
                # t_outputs, _, _ = self.model(t_inputs)
                # test
                t_outputs = self.model(t_inputs)


                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        return test_acc, f1

    def run(self, repeats=5):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        # criterion = CrossEntropyLoss_LSR()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        if not os.path.exists('log/'):
            os.mkdir('log/')

        f_out = open('log/' + self.opt.model_name + '_' + self.opt.dataset + '_val.csv', 'a', encoding='utf-8')

        max_test_acc_avg = 0
        max_test_f1_avg = 0
        time_avg = 0
        for i in range(repeats):
            print('repeat: ', (i + 1))
            start = time.time()
            f_out.write('repeat,' + str(i + 1))
            self._reset_params()
            max_test_acc, max_test_f1 = self._train(criterion, optimizer)
            print('max_test_acc: {0}     max_test_f1: {1}'.format(max_test_acc, max_test_f1))
            f_out.write(',max_test_acc,{0},max_test_f1,{1}\n'.format(max_test_acc, max_test_f1))
            max_test_acc_avg += max_test_acc
            max_test_f1_avg += max_test_f1
            end = time.time()
            time_avg += end - start
            print('take %s seconds' % (end - start))
            print('#' * 100)
        print("max_test_acc_avg:", max_test_acc_avg / repeats)
        print("max_test_f1_avg:", max_test_f1_avg / repeats)
        print("time_avg:", time_avg / repeats)

        f_out.close()


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', default='asgcn_heterogeneous', type=str)
    parser.add_argument('--model_name', default='Hete_GNNs', type=str)
    # parser.add_argument('--dataset', default='lap14', type=str, help='twitter, rest14, lap14, rest15, rest16')
    parser.add_argument('--dataset', default='twitter', type=str, help='twitter, rest14, lap14, rest15, rest16')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--seed', default=776, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--hops', default=3, type=int)

    opt = parser.parse_args()

    model_classes = {
        'lstm': LSTM,
        # 'ascnn': ASCNN,
        'asgcn': ASGCN,
        'astcn': ASGCN,
        'aen_glove': AEN_GloVe,
        'aen_bert': AEN_BERT,
        'asmgcn': ASMGCN_GloVe,
        'bert_spc': BERT_SPC,
        'asgcn_heterogeneous': ASGCN_HETEROGENEOUS,
        'asgat_ian': ASGAT_IAN,
        'ian': IAN,
        'aoa': AOA,
        'td_lstm': TD_LSTM,
        'memnet': MemNet,
        'ram': RAM,
        'atae_lstm': ATAE_LSTM,
        'tnet_lf': TNet_LF,
        'mgan': MGAN,
        'Hete_GNNs': Hete_GNNs,
    }
    input_colses = {
        'lstm': ['text_indices'],
        # 'ascnn': ['text_indices', 'aspect_indices', 'left_indices'],
        'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'astcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'aen_glove': ['text_indices', 'aspect_indices'],
        'aen_bert': ['text_raw_bert_indices', 'aspect_bert_indices'],
        'asmgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'bert_spc': ['text_bert_indices', 'bert_segments_ids'],
        'asgcn_heterogeneous': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'asgat_ian': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'ian': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'aoa': ['text_indices', 'aspect_indices'],
        'td_lstm': ['text_indices', 'aspect_indices'],
        'memnet': ['text_indices', 'aspect_indices'],
        'atae_lstm': ['text_indices', 'aspect_indices'],
        'ram': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'mgan': ['text_indices', 'aspect_indices', 'left_indices'],
        'Hete_GNNs': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    # logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()
