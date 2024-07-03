import os
import os.path as osp
import math

import numpy as np
import torch
import gc
import torch.nn as nn
from torch.nn.functional import softplus
import torch_geometric.transforms as T

from torch.utils.checkpoint import checkpoint
from torch_cluster import knn_graph, graclus_cluster
from torch_scatter import scatter
from torch_sparse.storage import SparseStorage

from torch import Tensor
from torch_geometric.typing import OptTensor, Optional, Tuple


from torch_geometric.nn import EdgeConv, NNConv
from torch_geometric.nn.pool.pool import pool_batch
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import normalized_cut
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import (max_pool, max_pool_x, global_max_pool,
                                avg_pool, avg_pool_x, global_mean_pool, 
                                global_add_pool)

from torch_sparse import SparseTensor

transform = T.Cartesian(cat=False)

def normalized_cut_2d(edge_index, pos):
    row, col = edge_index[0], edge_index[1]
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

# jit compatible version of coalesce
def coalesce(index, value: OptTensor, m: int, n: int, op: str = "add"):
    storage = SparseStorage(row=index[0], col=index[1], value=value,
                            sparse_sizes=(m, n), is_sorted=False)
    storage = storage.coalesce(reduce=op)
    return torch.stack([storage.row(), storage.col()], dim=0), storage.value()

# jit compatible version of to_undirected
def to_undirected(edge_index, num_nodes: Optional[int] = None) -> Tensor:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index[0], edge_index[1]
    temp = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    row, col = temp[0], temp[1]
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index

# jit compatible version of pool_edge, depends on coalesce
def pool_edge(cluster, edge_index, edge_attr: Optional[torch.Tensor] = None):
    num_nodes = cluster.size(0)
    edge_index = cluster[edge_index.view(-1)].view(2, -1)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if edge_index.numel() > 0:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                         num_nodes)
    return edge_index, edge_attr

def _aggr_pool_x(cluster, x, aggr: str, size: Optional[int] = None):
    """Call into scatter with configurable reduction op"""
    return scatter(x, cluster, dim=0, dim_size=size, reduce=aggr)

def global_pool_aggr(x, batch: Tensor, aggr: str, size: Optional[int] = None):
    """Global pool via passed aggregator: 'mean', 'add', 'max'"""
    if batch is None and size is None:
        raise Exception('Must provide at least one of "batch" or "size"')
    if batch is not None:
        size = int(batch.max().item() + 1)
    assert batch is not None
    return scatter(x, batch, dim=0, dim_size=size, reduce=aggr)

# this function is specialized compared to the more general non-jittable version
# in particular edge_attr can be removed since it is always None
def aggr_pool(cluster, x, batch: Tensor, aggr: str) -> Tuple[Tensor, Tensor]:
    """jit-friendly version of max/mean/add pool"""
    cluster, perm = consecutive_cluster(cluster)
    x = _aggr_pool_x(cluster, x, aggr)
    if batch is not None:
        batch = pool_batch(perm, batch)
    return x, batch

def aggr_pool_x(cluster, x, batch: Tensor, aggr: str, size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """*_pool_x with configurable aggr method"""
    if batch is None and size is None:
        raise Exception('Must provide at least one of "batch" or "size"')
    #if size is not None and batch is not None:
    #    batch_size = int(batch.max().item()) + 1
    #    return _aggr_pool_x(cluster, x, aggr, batch_size * size), None

    cluster, perm = consecutive_cluster(cluster)
    x = _aggr_pool_x(cluster, x, aggr)
    if batch is not None:
        batch = pool_batch(perm, batch)

    return x, batch
    
class DRN(nn.Module):
    '''
    This model iteratively contracts nearest neighbour graphs 
    until there is one output node.
    The latent space trained to group useful features at each level
    of aggregration.
    This allows single quantities to be regressed from complex point counts
    in a location and orientation invariant way.
    One encoding layer is used to abstract away the input features.

    @param input_dim: dimension of input features
    @param hidden_dim: dimension of hidden layers
    @param output_dim: dimension of output
    @param graph_features: number of high-level features
    
    @param k: size of k-nearest neighbor graphs
    @param aggr: message passing aggregation scheme. 
    @param loop: boolean for presence/absence of self loops in k-nearest neighbor graphs
    @param pool: type of pooling in aggregation layers. Choices are 'add', 'max', 'mean'
    
    @param agg_layers: number of aggregation layers. Must be >=0
    @param mp_layers: number of layers in message passing networks. Must be >=1
    @param in_layers: number of layers in inputnet. Must be >=1
    @param out_layers: number of layers in outputnet. Must be >=1
    '''
    def __init__(self, 
                 input_dim,
                 hidden_dim=64, 
                 output_dim=6, 
                 graph_features = 2,
                 k=16, 
                 aggr='add', 
                 loop=True, 
                 pool='max',
                 agg_layers=6, 
                 mp_layers=3, 
                 in_layers=4, 
                 out_layers=2,
                 activation='ELU'):

        super(DRN, self).__init__()

        activ = getattr(torch.nn, activation)

        self.graph_features = graph_features

        self.loop = loop

        self.k = k

        #construct inputnet
        in_layers_l = []
        in_layers_l += [nn.Linear(input_dim, hidden_dim), activ()]

        for i in range(in_layers-1):
            in_layers_l += [nn.Linear(hidden_dim, hidden_dim), activ()]

        self.inputnet = nn.Sequential(*in_layers_l)

        #construct aggregation layers
        self.agg_layers = nn.ModuleList()
        for i in range(agg_layers):
            #construct message passing network
            mp_layers_l = []

            for j in range(mp_layers-1):
                mp_layers_l += [nn.Linear(2*hidden_dim, 2*hidden_dim), activ()]

            mp_layers_l += [nn.Linear(2*hidden_dim, hidden_dim), activ()]
           
            convnn = nn.Sequential(*mp_layers_l)
            
            self.agg_layers.append(EdgeConv(nn=convnn, aggr=aggr).jittable())

        #construct outputnet
        out_layers_l = []

        for i in range(out_layers-1):
            out_layers_l += [nn.Linear(hidden_dim+self.graph_features, hidden_dim+self.graph_features), 
                    activ()]

        out_layers_l += [nn.Linear(hidden_dim+self.graph_features, output_dim)]
        self.output_dim = output_dim

        self.output = nn.Sequential(*out_layers_l)

        if pool not in {'max', 'mean', 'add'}:
            raise Exception("ERROR: INVALID POOLING")
        
        self.aggr_type = pool

    def forward(self, 
                x: Tensor, 
                graph_x: Tensor, 
                batch: Tensor) -> Tensor:
        torch.manual_seed(0)
        '''
        Push the batch 'data' through the network
        '''
        #torch.use_deterministic_algorithms(True)

        #print("top of forward")
        if len(x) == 0:
            return torch.empty( (0, self.output_dim) )

        graph_x = graph_x.view((-1, self.graph_features))

        points = self.inputnet(x)

        # if there are no aggregation layers just leave x, batch alone
        nAgg = len(self.agg_layers)
        for i, edgeconv in enumerate(self.agg_layers):
            knn = knn_graph(points, self.k, batch, loop=self.loop, flow=edgeconv.flow)
            edge_index = to_undirected(knn)
            adj = SparseTensor(
                row=edge_index[1], col=edge_index[0], 
                value=torch.ones(len(edge_index[0])).to(edge_index.device), 
                sparse_sizes = (points.shape[0], points.shape[0])
            )

            points = edgeconv(points, adj)

            weight = normalized_cut_2d(edge_index, points)
            cluster = graclus_cluster(edge_index[0], edge_index[1], weight, points.size(0))

            if i == nAgg - 1:
                points, batch = aggr_pool_x(cluster, points, batch, self.aggr_type)
            else:
                points, batch = aggr_pool(cluster, points, batch, self.aggr_type)
        
        # this xforms to batch-per-row so no need to return batch
        points = global_pool_aggr(points, batch, self.aggr_type)

        if graph_x is not None:
            points = torch.cat((points, graph_x), 1)

        points = self.output(points).squeeze(-1)

        return points
