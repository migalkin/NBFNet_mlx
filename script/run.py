import os
import sys
import math
import pprint
from functools import partial

import mlx
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx import optimizers as optim
from mlx_graphs.data.data import GraphData

from mlx.utils import tree_flatten, tree_map

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util, models


separator = ">" * 30
line = "-" * 30

def batch_iterate(batch_size, dataset, shuffle=False):
    num_samples = dataset.shape[0]
    perm = mx.array(np.random.permutation(num_samples)) if shuffle else mx.arange(num_samples)
    for s in range(0, num_samples, batch_size):
        ids = perm[s : s + batch_size]
        yield dataset[ids]


def forward_fn(model, graph, batch, cfg):
    preds = model(graph, batch)
    labels = mx.zeros_like(preds)
    labels[:, 0] = 1
    loss = nn.losses.binary_cross_entropy(preds, labels, reduction="none")
    neg_weight = mx.stop_gradient(mx.ones_like(preds))
    if cfg.task.adversarial_temperature > 0:
        #with torch.no_grad():
        neg_weight[:, 1:] = mx.softmax(preds[:, 1:] / cfg.task.adversarial_temperature, axis=-1)
    else:
        neg_weight[:, 1:] = 1 / cfg.task.num_negative
    loss = (loss * neg_weight).sum(axis=-1) / neg_weight.sum(axis=-1)
    loss = loss.mean()
    # sometimex training with MLX yields sudden NaNs, restarting helps
    if mx.isnan(loss):
        print('nan!')
        raise Exception("nan has been caught, restart training")
    return loss

def train_and_validate(cfg, model, train_data, valid_data, filtered_data=None):
    if cfg.train.num_epoch == 0:
        return

    train_triplets = mx.concatenate([train_data.target_edge_index, train_data.target_edge_type[None, :]]).T

    mx.eval(model.parameters())
    nparams = sum(x.size for k, x in tree_flatten(model.parameters()))
    #num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {nparams}")

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(**cfg.optimizer)

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    for i in range(0, cfg.train.num_epoch, step):
        model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            logger.warning(separator)
            logger.warning("Epoch %d begin" % epoch)

            losses = []

            # Poor man's dataloader
            num_batches = math.ceil(train_triplets.shape[0] / cfg.train.batch_size)
            loader = batch_iterate(batch_size=cfg.train.batch_size, dataset=train_triplets, shuffle=True)
            for _id in range(num_batches):
                batch = next(loader)
                batch = tasks.negative_sampling(train_data, batch, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative)

                loss_and_grad_fn = nn.value_and_grad(model, forward_fn)
                loss, grads = loss_and_grad_fn(model, train_data, batch, cfg)
                
                # eval computation graph
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                #mx.eval(model.state, optimizer.state)

                if batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss.item())
                losses.append(loss.item())
                batch_id += 1

            avg_loss = sum(losses) / len(losses)
            logger.warning(separator)
            logger.warning("Epoch %d end" % epoch)
            logger.warning(line)
            logger.warning("average binary cross entropy: %g" % avg_loss)

        epoch = min(cfg.train.num_epoch, i + step)
        
        logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
        model.save_weights("model_epoch_%d.npz" % epoch)

        logger.warning(separator)
        logger.warning("Evaluate on valid")
        result = test(cfg, model, valid_data, filtered_data=filtered_data)
        if result > best_result:
            best_result = result
            best_epoch = epoch

    logger.warning("Load checkpoint from model_epoch_%d.npz" % best_epoch)
    model.load_weights("model_epoch_%d.npz" % best_epoch)


def test(cfg, model, test_data, filtered_data=None):

    test_triplets = mx.concatenate([test_data.target_edge_index, test_data.target_edge_type[None, :]]).transpose()

    # Test dataloader
    loader = batch_iterate(batch_size=cfg.train.batch_size, dataset=test_triplets, shuffle=False)
    num_batches = math.ceil(test_triplets.shape[0] / cfg.train.batch_size)

    model.eval()
    rankings = []
    num_negatives = []
    for batch_id in range(num_batches):
        batch = next(loader)
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        t_pred = model(test_data, t_batch)
        h_pred = model(test_data, h_batch)

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.transpose()
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(axis=-1)
        num_h_negative = h_mask.sum(axis=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

    all_ranking = mx.concatenate(rankings)
    all_num_negative = mx.concatenate(num_negatives)

    
    for metric in cfg.task.metric:
        if metric == "mr":
            score = all_ranking.astype(mx.float32).mean()
        elif metric == "mrr":
            score = (1 / all_ranking.astype(mx.float32)).mean()
        elif metric.startswith("hits@"):
            values = metric[5:].split("_")
            threshold = int(values[0])
            if len(values) > 1:
                num_sample = int(values[1])
                # unbiased estimation
                fp_rate = (all_ranking - 1).astype(mx.float32) / all_num_negative
                score = 0
                for i in range(threshold):
                    # choose i false positive from num_sample - 1 negatives
                    num_comb = math.factorial(num_sample - 1) / \
                                math.factorial(i) / math.factorial(num_sample - i - 1)
                    score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                score = score.mean()
            else:
                score = (all_ranking <= threshold).astype(mx.float32).mean()
        logger.warning("%s: %g" % (metric, score.item()))
    mrr = (1 / all_ranking.astype(mx.float32)).mean()

    return mrr


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    logger = util.get_root_logger()
    logger.warning("Random seed: %d" % args.seed)
    logger.warning("Config file: %s" % args.config)
    logger.warning(pprint.pformat(cfg))
    is_inductive = cfg.dataset["class"].startswith("Ind")
    dataset = util.build_dataset(cfg)
    cfg.model.num_relation = dataset.num_relations

    # Model creation
    model = models.NBFNet(**cfg.model)

    if "checkpoint" in cfg:
        model.load_weights(cfg.checkpoint)

    # model = util.build_model(cfg)

    train_data, valid_data, test_data = dataset.train_data, dataset.valid_data, dataset.test_data
    if is_inductive:
        # for inductive setting, use only the test fact graph for filtered ranking
        # test filtering graph: inference edges + test edges
        full_inference_edges = mx.concatenate([test_data.edge_index, test_data.target_edge_index], axis=1)
        full_inference_etypes = mx.concatenate([test_data.edge_type, test_data.target_edge_type])
        test_filtered_data = GraphData(edge_index=full_inference_edges, edge_type=full_inference_etypes)

        # validation filtering graph: train edges + validation edges
        val_filtered_data = GraphData(
            edge_index=mx.concatenate([train_data.edge_index, valid_data.target_edge_index], axis=1),
            edge_type=mx.concatenate([train_data.edge_type, valid_data.target_edge_type])
        )
    else:
        # for transductive setting, use the whole graph for filtered ranking
        filtered_data = GraphData(edge_index=dataset.data.target_edge_index, edge_type=dataset.data.target_edge_type)

    train_and_validate(cfg, model, train_data, valid_data, filtered_data=val_filtered_data)
    
    logger.warning(separator)
    logger.warning("Evaluate on valid")
    test(cfg, model, valid_data, filtered_data=val_filtered_data)

    logger.warning(separator)
    logger.warning("Evaluate on test")
    test(cfg, model, test_data, filtered_data=test_filtered_data)