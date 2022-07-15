from copy import deepcopy
from numbers import Number

import numpy as np
import torch
from torch.autograd import Variable


def to_np_array(*arrays, **kwargs):
    array_list = []
    for array in arrays:
        if isinstance(array, Variable):
            if array.is_cuda:
                array = array.cpu()
            array = array.data
        if isinstance(array, torch.Tensor) or \
           isinstance(array, torch.FloatTensor) or \
           isinstance(array, torch.LongTensor) or \
           isinstance(array, torch.ByteTensor) or \
           isinstance(array, torch.cuda.FloatTensor) or \
           isinstance(array, torch.cuda.LongTensor) or \
           isinstance(array, torch.cuda.ByteTensor):
            if array.is_cuda:
                array = array.cpu()
            array = array.numpy()
        if isinstance(array, Number):
            pass
        elif isinstance(array, list) or isinstance(array, tuple):
            array = np.array(array)
        elif array.shape == (1, ):
            if "full_reduce" in kwargs and kwargs["full_reduce"] is False:
                pass
            else:
                array = array[0]
        elif array.shape == ():
            array = array.tolist()
        array_list.append(array)
    if len(array_list) == 1:
        array_list = array_list[0]
    return array_list


def random_remove_edge(data, remove_edge_fraction):
    """
    Randomly remove a certain fraction of edges.
    """
    data_c = deepcopy(data)
    num_edges = int(data_c.edge_index.shape[1] / 2)
    num_removed_edges = int(num_edges * remove_edge_fraction)
    edges = [tuple(ele) for ele in to_np_array(data_c.edge_index.T)]
    for _ in range(num_removed_edges):
        idx = np.random.choice(len(edges))
        edge = edges[idx]
        edge_r = (edge[1], edge[0])
        edges.pop(idx)
        try:
            edges.remove(edge_r)
        except:
            pass
    data_c.edge_index = torch.LongTensor(np.array(edges).T).to(
        data.edge_index.device)
    return data_c


def random_add_edge(data, added_edge_fraction):
    """
    Randomly Add edges to the original data's edge_index.
    """
    if added_edge_fraction == 0:
        return data
    data_c = deepcopy(data)
    num_edges = int(data.edge_index.shape[1] / 2)
    num_added_edges = int(num_edges * added_edge_fraction)
    edges = [tuple(ele) for ele in to_np_array(data.edge_index.T)]
    added_edges = []
    for _ in range(num_added_edges):
        while True:
            added_edge_cand = tuple(
                np.random.choice(data.x.shape[0], size=2, replace=False))
            added_edge_r_cand = (added_edge_cand[1], added_edge_cand[0])
            if added_edge_cand in edges or added_edge_cand in added_edges:
                if added_edge_cand in edges:
                    assert added_edge_r_cand in edges
                if added_edge_cand in added_edges:
                    assert added_edge_r_cand in added_edges
                continue
            else:
                added_edges.append(added_edge_cand)
                added_edges.append(added_edge_r_cand)
                break

    added_edge_index = torch.LongTensor(np.array(added_edges).T).to(
        data.edge_index.device)
    data_c.edge_index = torch.cat([data.edge_index, added_edge_index], 1)
    return data_c


def attack_edge(dataset, edge_fraction, mode='remove'):
    assert mode in ['remove', 'add']
    tmp_dataset = []
    for data in dataset:
        if mode == 'remove':
            data_c = random_remove_edge(data, edge_fraction)
        else:
            data_c = random_add_edge(data, edge_fraction)
        tmp_dataset.append(data_c)
    return tmp_dataset
