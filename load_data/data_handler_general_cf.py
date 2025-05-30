import os
import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import scipy.sparse as sp
from config.configurator import configs
from load_data.datasets_general_cf import PairwiseTrnData, PairwiseWEpochFlagTrnData, AllRankTstData
import torch as t
import torch.utils.data as data


class DataHandlerGeneralCF:
    def __init__(self):
        file_pre_dir = os.getcwd()
        if configs['data']['name'] == 'amazon':
            data_pre_dir = f'{file_pre_dir}/data/amazon/'
        elif configs['data']['name'] == 'yelp':
            data_pre_dir = f'{file_pre_dir}/data/yelp/'
        elif configs['data']['name'] == 'steam':
            data_pre_dir = f'{file_pre_dir}/data/steam/'
        elif configs['data']['name'] == 'movie':
            data_pre_dir = f'{file_pre_dir}/data/movie/'
        elif configs['data']['name'] == 'sports':
            data_pre_dir = f'{file_pre_dir}/data/sports/'
        else:
            raise NotImplementedError
        self.trn_file = data_pre_dir + 'trn_mat.pkl'
        self.val_file = data_pre_dir + 'val_mat.pkl'
        self.tst_file = data_pre_dir + 'tst_mat.pkl'

    def _load_one_mat(self, file):
        with open(file, 'rb') as fs:
            mat = (pickle.load(fs) != 0).astype(np.float32)
        if type(mat) != coo_matrix:
            mat = coo_matrix(mat)
        return mat

    def _normalize_adj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

    def _make_torch_adj(self, mat, self_loop=False):
        if not self_loop:
            a = csr_matrix((configs['data']['user_num'], configs['data']['user_num']))
            b = csr_matrix((configs['data']['item_num'], configs['data']['item_num']))
        else:
            data = np.ones(configs['data']['user_num'])
            row_indices = np.arange(configs['data']['user_num'])
            column_indices = np.arange(configs['data']['user_num'])
            a = csr_matrix((data, (row_indices, column_indices)),
                           shape=(configs['data']['user_num'], configs['data']['user_num']))

            data = np.ones(configs['data']['item_num'])
            row_indices = np.arange(configs['data']['item_num'])
            column_indices = np.arange(configs['data']['item_num'])
            b = csr_matrix((data, (row_indices, column_indices)),
                           shape=(configs['data']['item_num'], configs['data']['item_num']))

        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = self._normalize_adj(mat)

        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse_coo_tensor(t.LongTensor(idxs), t.FloatTensor(vals), shape).to(configs['device'])

    def load_data(self):
        trn_mat = self._load_one_mat(self.trn_file)
        val_mat = self._load_one_mat(self.val_file)
        tst_mat = self._load_one_mat(self.tst_file)

        self.trn_mat = trn_mat
        configs['data']['user_num'], configs['data']['item_num'] = trn_mat.shape
        self.torch_adj = self._make_torch_adj(trn_mat)

        if configs['model']['name'] == 'gccf':
            self.torch_adj = self._make_torch_adj(trn_mat, self_loop=True)

        if configs['train']['loss'] == 'pairwise':
            trn_data = PairwiseTrnData(trn_mat)
        elif configs['train']['loss'] == 'pairwise_with_epoch_flag':
            trn_data = PairwiseWEpochFlagTrnData(trn_mat)

        val_data = AllRankTstData(val_mat, trn_mat)
        tst_data = AllRankTstData(tst_mat, trn_mat)
        self.test_dataloader = data.DataLoader(tst_data, batch_size=configs['test']['batch_size'], shuffle=False,
                                               num_workers=0)
        self.valid_dataloader = data.DataLoader(val_data, batch_size=configs['test']['batch_size'], shuffle=False,
                                                num_workers=0)
        self.train_dataloader = data.DataLoader(trn_data, batch_size=configs['train']['batch_size'], shuffle=True,
                                                num_workers=0)
