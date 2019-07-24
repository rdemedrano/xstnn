import os

import numpy as np
import torch

from utils import DotDict, normalize


def dataset_factory(data_dir, name, k=1):
    # get dataset
    if name[:5] == 'crash':
        opt, data, relations = crash_ex(data_dir, '{}.csv'.format(name))
    else:
        raise ValueError('Non dataset named `{}`.'.format(name))
    # make k hop
    new_rels = [relations]
    for n in range(k - 1):
        new_rels.append(torch.stack([new_rels[-1][:, r].matmul(new_rels[0][:, r]) for r in range(relations.size(1))], 1))
    relations = torch.cat(new_rels, 1)
    # split train / test
    train_data = data[:opt.nt_train, :, :opt.nd]
    test_data = data[opt.nt_train:, :, :opt.nd]
    # Exogenous variables
    exogenous_train = data[:opt.nt_train, :, opt.nd:]
    exogenous_test = data[opt.nt_train:, :, opt.nd:]
    return opt, (train_data, test_data), relations, (exogenous_train, exogenous_test)


def crash_ex(data_dir, file='crash_.csv'):
    # dataset configuration
    opt = DotDict()
    opt.nt = 1085
    opt.nt_train = 1080
    opt.nx = 131
    opt.np = 8
    opt.nd = 1
    opt.periode = opt.nt
    # loading data
    data = torch.Tensor(np.genfromtxt(os.path.join(data_dir, file))).view(opt.nt, opt.nx, opt.np + opt.nd)
    # load relations
    relations = torch.Tensor(np.genfromtxt(os.path.join(data_dir, 'crash_relations.csv')))
    relations = normalize(relations).unsqueeze(1)
    return opt, data, relations

