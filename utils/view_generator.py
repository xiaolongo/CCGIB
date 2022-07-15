import torch
from torch_geometric.utils import dense_to_sparse, to_dense_adj, add_remaining_self_loops


def cosine_similarity(x):
    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    x_norm = x.div(x_norm + 5e-10)
    cos_adj = torch.mm(x_norm, x_norm.transpose(0, 1))
    return cos_adj


def build_knn_graph(x, k):
    '''generating knn graph from data x
    Args:
        x: input data (n, m)
        k: number of nearst neighbors           
    returns:
        knn_edge_index, knn_edge_weight
    '''
    cos_adj = cosine_similarity(x)
    topk = min(k + 1, cos_adj.size(-1))
    knn_val, knn_ind = torch.topk(cos_adj, topk, dim=-1)
    weighted_adj = (torch.zeros_like(cos_adj)).scatter_(-1, knn_ind, knn_val)
    knn_edge_index, knn_edge_weight = dense_to_sparse(weighted_adj)
    return knn_edge_index, knn_edge_weight


def build_dilated_knn_graph(x, k1, k2):
    '''generating dilated knn graph from data x
    Args:
        x: input data (n, m)
        k1: number of nearst neighbors
        k2: number of dilations           
    returns:
        knn_edge_index, knn_edge_weight
    '''
    cos_adj = cosine_similarity(x)
    topk = min(k1 + 1, cos_adj.size(-1))
    knn_val, knn_ind = torch.topk(cos_adj, topk, dim=-1)
    knn_val = knn_val[:, k2:]  #
    knn_ind = knn_ind[:, k2:]  #
    weighted_adj = (torch.zeros_like(cos_adj)).scatter_(-1, knn_ind, knn_val)
    knn_edge_index, knn_edge_weight = dense_to_sparse(weighted_adj)
    return knn_edge_index, knn_edge_weight


def build_ppr_graph(edge_index, alpha=0.1):
    '''generating PageRank graph from adj
    Args:
        edge_index: input adj
        alpha: hyper parameter    
    returns:
        ppr_edge_index, ppr_edge_weight
    '''
    edge_weight = torch.ones(edge_index.size(-1), dtype=torch.long)
    edge_weight = (alpha - 1) * edge_weight
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight)
    adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight).squeeze()
    ppr_adj = alpha * torch.inverse(adj)
    ppr_edge_index, ppr_edge_weight = dense_to_sparse(ppr_adj)
    return ppr_edge_index, ppr_edge_weight


def view_generator(dataset, view1='adj', view2='KNN'):
    temp_dataset = []
    assert view1 in ['PPR', 'KNN', 'DKNN', 'adj']
    assert view2 in ['PPR', 'KNN', 'DKNN', 'adj']

    if view1 == 'PPR':
        for data in dataset:
            data.edge_index1, data.edge_weight1 = build_ppr_graph(
                data.edge_index)
            temp_dataset.append(data)
    elif view1 == 'KNN':
        for data in dataset:
            data.edge_index1, _ = build_knn_graph(data.x, k=5)
            temp_dataset.append(data)
    elif view1 == 'DKNN':
        for data in dataset:
            data.edge_index1, _ = build_dilated_knn_graph(data.x, k1=10, k2=5)
            temp_dataset.append(data)
    elif view1 == 'adj':
        for data in dataset:
            data.edge_index1 = data.edge_index
            temp_dataset.append(data)

    if view2 == 'PPR':
        for data in temp_dataset:
            data.edge_index2, data.edge_weight2 = build_ppr_graph(
                data.edge_index)
    elif view2 == 'KNN':
        for data in temp_dataset:
            data.edge_index2, _ = build_knn_graph(data.x, k=5)
    elif view2 == 'DKNN':
        for data in temp_dataset:
            data.edge_index2, _ = build_dilated_knn_graph(data.x, k1=10, k2=5)
    return temp_dataset
