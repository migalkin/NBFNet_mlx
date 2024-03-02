from functools import reduce
import numpy as np
#import torch
import mlx.core as mx
from mlx_graphs.utils import scatter

# For reference, the OG algorithm in pytorch
# def edge_match(edge_index, query_index):
#     edge_index = torch.from_numpy(np.array(edge_index))
#     query_index = torch.from_numpy(np.array(query_index))
#     # preparing unique hashing of edges, base: (max_node, max_relation) + 1
#     base = edge_index.max(dim=1)[0] + 1
#     # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
#     # idea: max number of edges = num_nodes * num_relations
#     # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
#     # given a tuple (h, r), we will search for all other existing edges starting from head h
#     assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
#     scale = base.cumprod(0)
#     scale = scale[-1] // scale

#     # hash both the original edge index and the query index to unique integers
#     edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
#     edge_hash, order = edge_hash.sort()
#     query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

#     # matched ranges: [start[i], end[i])
#     start = torch.bucketize(query_hash, edge_hash)
#     end = torch.bucketize(query_hash, edge_hash, right=True)
#     # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
#     num_match = end - start

#     # generate the corresponding ranges
#     offset = num_match.cumsum(0) - num_match
#     range = torch.arange(num_match.sum(), device=edge_index.device)
#     range = range + (start - offset).repeat_interleave(num_match)

#     return order[range], num_match


def edge_match(edge_index, query_index):
    # O((n + q)logn) time
    # O(n) memory
    # edge_index: big underlying graph
    # query_index: edges to match
    edge_index, query_index = np.array(edge_index).astype(np.int64), np.array(query_index).astype(np.int64)
    # preparing unique hashing of edges, base: (max_node, max_relation) + 1
    base = edge_index.max(axis=1) + 1  # OG torch had .max()[0] because returns values and indices
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    #assert reduce(int.__mul__, base.tolist()) < torch.iinfo(mx.int64).max
    #scale = mx.cumprod(base)
    scale = base.cumprod(0)
    scale = scale[-1] // scale

    # hash both the original edge index and the query index to unique integers
    edge_hash_prep = (edge_index * scale[:, None]).sum(axis=0)
    #edge_hash = np.sort(edge_hash_prep)
    order = np.argsort(edge_hash_prep)
    edge_hash = edge_hash_prep[order] 
    query_hash = (query_index * scale[:, None]).sum(axis=0)

    # matched ranges: [start[i], end[i])
    # TODO bucketize is not yet implemented in MLX, run it on CPU with numpy
    #start = torch.bucketize(query_hash, edge_hash)
    #end = torch.bucketize(query_hash, edge_hash, right=True)

    start = np.digitize(query_hash, edge_hash)
    end = np.digitize(query_hash, edge_hash, right=True)

    # start = np.digitize(np.array(query_hash, copy=False), np.array(edge_hash, copy=False))
    # end = np.digitize(np.array(query_hash, copy=False), np.array(edge_hash, copy=False), right=True)

    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    # num_match = end - start
    num_match = start - end  # vice versa for the numpy version 

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    range = np.arange(num_match.sum())  # TODO: add stream
    range = range + (end - offset).repeat(num_match)
    # range = range + (start - offset).repeat(num_match)

    return mx.array(order[mx.array(range)]), mx.array(num_match)


def negative_sampling(data, batch, num_negative, strict=True):
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.T

    # strict negative sampling vs random negative sampling
    if strict:
        t_mask, h_mask = strict_negative_mask(data, batch)
        t_mask = t_mask[:batch_size // 2]
        neg_t_candidate = np.nonzero(np.array(t_mask))[1]
        num_t_candidate = mx.array(t_mask.sum(axis=-1))
        # draw samples for negative tails
        rand = mx.random.uniform(0,1, (len(t_mask), num_negative))
        index = (rand * num_t_candidate[..., None]).astype(mx.int64)
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate)[..., None]
        neg_t_index = mx.array(neg_t_candidate[index])

        h_mask = h_mask[batch_size // 2:]
        neg_h_candidate = np.nonzero(np.array(h_mask))[1]
        num_h_candidate = mx.array(h_mask.sum(axis=-1))
        # draw samples for negative heads
        rand = mx.random.uniform(0, 1, (len(h_mask), num_negative))
        index = (rand * num_h_candidate[..., None]).astype(mx.int64)
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate)[..., None]
        neg_h_index = mx.array(neg_h_candidate[index])
    else:
        neg_index = mx.random.randint(0, data.num_nodes, (batch_size, num_negative))  # TODO add stream
        neg_t_index, neg_h_index = neg_index[:batch_size // 2], neg_index[batch_size // 2:]

    h_index = mx.repeat(pos_h_index[:, None], num_negative + 1, axis=-1)
    t_index = mx.repeat(pos_t_index[:, None], num_negative + 1, axis=-1)
    r_index = mx.repeat(pos_r_index[:, None], num_negative + 1, axis=-1)
    t_index[:batch_size // 2, 1:] = neg_t_index
    h_index[batch_size // 2:, 1:] = neg_h_index

    return mx.stack([h_index, t_index, r_index], axis=-1)


def all_negative(data, batch_og):
    batch = np.array(batch_og)
    pos_h_index, pos_t_index, pos_r_index = batch.transpose()
    r_index = np.repeat(pos_r_index[:, None], data.num_nodes, axis=-1)
    # r_index = pos_r_index.unsqueeze(-1).expand(-1, data.num_nodes)
    # generate all negative tails for this batch
    all_index = np.arange(data.num_nodes)
    h_index, t_index = np.meshgrid(pos_h_index, all_index, indexing='ij')
    t_batch = np.stack([h_index, t_index, r_index], axis=-1)
    # generate all negative heads for this batch
    all_index = np.arange(data.num_nodes)
    t_index, h_index = np.meshgrid(pos_t_index, all_index, indexing='ij')
    h_batch = np.stack([h_index, t_index, r_index], axis=-1)

    return mx.array(t_batch), mx.array(h_batch)


def strict_negative_mask(data, batch):
    # this function makes sure that for a given (h, r) batch we will NOT sample true tails as random negatives
    # similarly, for a given (t, r) we will NOT sample existing true heads as random negatives

    pos_h_index, pos_t_index, pos_r_index = batch.transpose()

    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    edge_index = mx.stack([data.edge_index[0], data.edge_type])
    # edge index of current batch (head, relation) for which we will sample negatives
    query_index = mx.stack([pos_h_index, pos_r_index])
    # search for all true tails for the given (h, r) batch
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    # switched to numpy edge_id and num_t_truth
    # build an index from the found edges
    t_truth_index = data.edge_index[1, edge_id]
    sample_id = np.arange(len(num_t_truth)).repeat(num_t_truth)
    t_mask = np.ones((len(num_t_truth), data.num_nodes), dtype=np.bool_)
    # assign 0s to the mask with the found true tails
    t_mask[sample_id, t_truth_index] = 0
    t_mask[np.arange(len(num_t_truth)), pos_t_index] = 0
    #t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    # part II: sample hard negative heads
    # edge_index[1] denotes tails, so the edge index becomes (t, r)
    edge_index = mx.stack([data.edge_index[1], data.edge_type])
    # edge index of current batch (tail, relation) for which we will sample heads
    query_index = mx.stack([pos_t_index, pos_r_index])
    # search for all true heads for the given (t, r) batch
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = np.arange(len(num_h_truth)).repeat(num_h_truth)
    h_mask = np.ones((len(num_h_truth), data.num_nodes), dtype=np.bool_)
    # assign 0s to the mask with the found true heads
    h_mask[sample_id, h_truth_index] = 0
    h_mask[np.arange(len(num_h_truth)), pos_h_index] = 0
    #h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)

    return mx.array(t_mask, dtype=mx.bool_), mx.array(h_mask, dtype=mx.bool_)


def compute_ranking(pred, target, mask=None):
    pos_pred = pred[mx.arange(pred.shape[0]), target]
    # unsqueeze
    pos_pred = pos_pred[:, None]
    # pos_pred = pred.gather(-1, target.unsqueeze(-1))
    if mask is not None:
        # filtered ranking
        ranking = mx.sum((pos_pred <= pred) & mask, axis=-1) + 1
    else:
        # unfiltered ranking
        ranking = mx.sum(pos_pred <= pred, axis=-1) + 1
    return ranking
