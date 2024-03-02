import copy
import numpy as np
from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.utils import scatter
from mlx_graphs.nn import Linear
from . import tasks, layers


class NBFNet(nn.Module):

    def __init__(self, input_dim, hidden_dims, num_relation, message_func="distmult", aggregate_func="pna",
                 short_cut=False, layer_norm=False, activation="relu", concat_hidden=False, num_mlp_layer=2,
                 dependent=True, remove_one_hop=False, **kwargs):
        super(NBFNet, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]

        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut  # whether to use residual connections between GNN layers
        self.concat_hidden = concat_hidden  # whether to compute final states as a function of all layer outputs or last
        self.remove_one_hop = remove_one_hop  # whether to dynamically remove one-hop edges from edge_index

        self.layers = []  # TODO: ModuleList of MLX?
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                self.dims[0], message_func, aggregate_func, layer_norm,
                                                                activation, dependent))

        feature_dim = (sum(hidden_dims) if concat_hidden else hidden_dims[-1]) + input_dim

        # additional relation embedding which serves as an initial 'query' for the NBFNet forward pass
        # each layer has its own learnable relations matrix, so we send the total number of relations, too
        self.query = nn.Embedding(num_relation, input_dim)
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(num_mlp_layer - 1):
            mlp.append(Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def remove_easy_edges(self, data, h_index, t_index, r_index=None):
        # we remove training edges (we need to predict them at training time) from the edge index
        # think of it as a dynamic edge dropout
        h_index_ext = mx.concatenate([h_index, t_index], axis=-1)
        t_index_ext = mx.concatenate([t_index, h_index], axis=-1)
        r_index_ext = mx.concatenate([r_index, r_index + self.num_relation // 2], axis=-1)
        if self.remove_one_hop:
            # we remove all existing immediate edges between heads and tails in the batch
            # TODO: np.digitize does not work properly in this setup, but for inductive datasets
            # remove_one_hop is turned off anyways
            edge_index = data.edge_index
            easy_edge = mx.stack([h_index_ext, t_index_ext]).flatten(1)
            index = tasks.edge_match(edge_index, easy_edge)[0]
            mask = ~index_to_mask(index, data.num_edges)
        else:
            # we remove existing immediate edges between heads and tails in the batch with the given relation
            edge_index = mx.concatenate([data.edge_index, data.edge_type[None, :]])
            # note that here we add relation types r_index_ext to the matching query
            easy_edge = mx.stack([h_index_ext, t_index_ext, r_index_ext]).flatten(1)
            index = tasks.edge_match(edge_index, easy_edge)[0]
            mask = ~index_to_mask(index, data.num_edges)

        data = copy.copy(data)
        # MLX crashes when doing boolean masking with varying shapes, so let's execute this in numpy
        new_ei = np.array(data.edge_index, copy=False)[:, mask]
        new_et = np.array(data.edge_type, copy=False)[mask]

        data.edge_index = mx.array(new_ei)
        data.edge_type = mx.array(new_et)
        return data

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        # is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        is_t_neg = mx.all(h_index == h_index[:, 0][:, None], axis=-1, keepdims=True)
        new_h_index = mx.where(is_t_neg, h_index, t_index)
        new_t_index = mx.where(is_t_neg, t_index, h_index)
        new_r_index = mx.where(is_t_neg, r_index, r_index + self.num_relation // 2)
        return new_h_index, new_t_index, new_r_index

    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        query = self.query(r_index)
        # index = mx.repeat(h_index[:, None], query.shape[1], axis=1)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = mx.zeros((batch_size, data.num_nodes, self.dims[0]))
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary[mx.arange(batch_size), h_index] = query
        # boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (data.num_nodes, data.num_nodes)
        edge_weight = mx.ones(data.num_edges)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = mx.repeat(mx.expand_dims(query, 1), data.num_nodes, axis=1)  # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = mx.concatenate(hiddens + [node_query], axis=-1)
        else:
            output = mx.concatenate([hiddens[-1], node_query], axis=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def __call__(self, data, batch):
        # h_index, t_index, r_index = batch.unbind(-1)
        h_index, t_index, r_index = batch[..., 0], batch[..., 1], batch[..., 2]
        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        assert mx.all(h_index == h_index[:, 0][:, None])
        assert mx.all(r_index == r_index[:, 0][:, None])

        # message passing and updated node representations
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = mx.repeat(t_index[:, :, None], feature.shape[-1], axis=2)
        # extract representations of tail entities from the updated node states (torch.gather())
        feature = mx.take_along_axis(feature, index, axis=1)  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score  #.reshape(shape)


def index_to_mask(index, size):
    index = index.reshape(-1)
    size = index.max().item() + 1 if size is None else size
    mask = mx.zeros(size, dtype=mx.bool_)
    mask[index] = True
    return mask
