import os

import numpy as np
import torch

from utils import DotDict, normalize


def dataset_factory(data_dir, name, k=1):
    # get dataset
#    if name[:5] == 'crash':
#        opt, data, relations = crash(data_dir, '{}.csv'.format(name))
#    else:
#        raise ValueError('Non dataset named `{}`.'.format(name))
    if name[:8] == 'crash_ex':
        opt, data, relations = crash_ex(data_dir, '{}.csv'.format(name))
    else:
        raise ValueError('Non dataset named `{}`.'.format(name))
#    if name[:4] == 'heat':
#        opt, data, relations = heat(data_dir, '{}.csv'.format(name))
#    else:
#        raise ValueError('Non dataset named `{}`.'.format(name))
#    if name[:6] == 'prueba':
#        opt, data, relations = prueba(data_dir, '{}.csv'.format(name))
#    else:
#        raise ValueError('Non dataset named `{}`.'.format(name))
    # make k hop
    new_rels = [relations]
    for n in range(k - 1):
        new_rels.append(torch.stack([new_rels[-1][:, r].matmul(new_rels[0][:, r]) for r in range(relations.size(1))], 1))
    relations = torch.cat(new_rels, 1)
    # split train / test
    train_data = data[:opt.nt_train, :, :opt.nd]
    test_data = data[opt.nt_train:, :, :opt.nd]
    # Variables exógenas
    exogenous_train = data[:opt.nt_train, :, opt.nd:]
    exogenous_test = data[opt.nt_train:, :, opt.nd:]
    return opt, (train_data, test_data), relations, (exogenous_train, exogenous_test)


def crash(data_dir, file='crash.csv'):
    # dataset configuration
    opt = DotDict()
    opt.nt = 365
    opt.nt_train = 355
    opt.nx = 49
    opt.np = 0
    opt.nd = 1
    opt.periode = opt.nt
    # loading data
    data = torch.Tensor(np.genfromtxt(os.path.join(data_dir, file))).view(opt.nt, opt.nx, opt.np + opt.nd)
    # load relations
    relations = torch.Tensor(np.genfromtxt(os.path.join(data_dir, 'crash_relations.csv')))
    relations = normalize(relations).unsqueeze(1)
    return opt, data, relations

def crash_ex(data_dir, file='crash_ex.csv'):
    # dataset configuration
    opt = DotDict()
    opt.nt = 1128
    opt.nt_train = 1080
    opt.nx = 49
    opt.np = 1
    opt.nd = 1
    opt.periode = opt.nt
    # loading data
    data = torch.Tensor(np.genfromtxt(os.path.join(data_dir, file))).view(opt.nt, opt.nx, opt.np + opt.nd)
    # load relations
    relations = torch.Tensor(np.genfromtxt(os.path.join(data_dir, 'crash_relations.csv')))
    relations = normalize(relations).unsqueeze(1)
    return opt, data, relations


def heat(data_dir, file='heat.csv'):
    # dataset configuration
    opt = DotDict()
    opt.nt = 200
    opt.nt_train = 100
    opt.nx = 41
    opt.np = 0
    opt.nd = 1
    opt.periode = opt.nt
    # loading data
    data = torch.Tensor(np.genfromtxt(os.path.join(data_dir, file))).view(opt.nt, opt.nx, opt.np + opt.nd)
    # load relations
    relations = torch.Tensor(np.genfromtxt(os.path.join(data_dir, 'heat_relations.csv')))
    relations = normalize(relations).unsqueeze(1)
    return opt, data, relations


def prueba(data_dir, file='prueba.csv'):
    # dataset configuration
    opt = DotDict()
    opt.nt = 5
    opt.nt_train = 3
    opt.nx = 4
    opt.np = 2
    opt.nd = 1
    opt.periode = opt.nt
    # loading data
    data = torch.Tensor(np.genfromtxt(os.path.join(data_dir, file))).view(opt.nt, opt.nx, opt.np + opt.nd)
    # La propia serie temporal sería: data[:,:,0:opt.nd]
    # load relations
    relations = torch.Tensor(np.genfromtxt(os.path.join(data_dir, 'prueba_relations.csv')))
    relations = normalize(relations).unsqueeze(1)
    return opt, data, relations
