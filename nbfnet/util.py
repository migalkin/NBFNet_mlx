import os
import sys
import ast
import time
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict

import mlx.core as mx
from mlx_graphs.data.data import GraphData

from nbfnet import models, datasets


logger = logging.getLogger(__file__)


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    tree = env.parse(raw)
    vars = meta.find_undeclared_variables(tree)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def literal_eval(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars


def get_root_logger(file=True):
    format = "%(asctime)-10s %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(format=format, datefmt=datefmt)
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    if file:
        handler = logging.FileHandler("log.txt")
        format = logging.Formatter(format, datefmt)
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def create_working_directory(cfg):
    file_name = "working_dir.tmp"

    working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.model["class"], cfg.dataset["class"], time.strftime("%Y-%m-%d-%H-%M-%S"))
    
    with open(file_name, "w") as fout:
        fout.write(working_dir)
    os.makedirs(working_dir)
    os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def build_dataset(cfg):
    cls = cfg.dataset.pop("class")
    # TODO: transductive datasets
    if cls == "FB15k-237":
        raise NotImplementedError
        dataset = RelLinkPredDataset(name=cls, **cfg.dataset)
        data = dataset.data
        train_data = GraphData(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                          target_edge_index=data.train_edge_index, target_edge_type=data.train_edge_type)
        valid_data = GraphData(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                          target_edge_index=data.valid_edge_index, target_edge_type=data.valid_edge_type)
        test_data = GraphData(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                         target_edge_index=data.test_edge_index, target_edge_type=data.test_edge_type)
        dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
    elif cls == "WN18RR":
        raise NotImplementedError 
        dataset = WordNet18RR(**cfg.dataset)
        # convert wn18rr into the same format as fb15k-237
        data = dataset.data
        num_nodes = int(data.edge_index.max()) + 1
        num_relations = int(data.edge_type.max()) + 1
        edge_index = data.edge_index[:, data.train_mask]
        edge_type = data.edge_type[data.train_mask]
        edge_index = mx.concatenate([edge_index, edge_index.flip(0)], dim=-1)
        edge_type = mx.concatenate([edge_type, edge_type + num_relations])
        train_data = GraphData(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                          target_edge_index=data.edge_index[:, data.train_mask],
                          target_edge_type=data.edge_type[data.train_mask])
        valid_data = GraphData(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                          target_edge_index=data.edge_index[:, data.val_mask],
                          target_edge_type=data.edge_type[data.val_mask])
        test_data = GraphData(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                         target_edge_index=data.edge_index[:, data.test_mask],
                         target_edge_type=data.edge_type[data.test_mask])
        dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
        dataset.num_relations = num_relations * 2
    elif cls.startswith("Ind"):
        dataset = datasets.IndRelLinkPredDataset(name=cls, **cfg.dataset)
    else:
        raise ValueError("Unknown dataset `%s`" % cls)

    logger.warning("%s dataset" % cls)
    logger.warning("#train: %d, #valid: %d, #test: %d" %
                    (dataset.train_data.target_edge_index.shape[1], dataset.valid_data.target_edge_index.shape[1],
                    dataset.test_data.target_edge_index.shape[1]))

    return dataset


def build_model(cfg):
    cls = cfg.model.pop('class')
    model = models.NBFNet(**cfg.model)

    if "checkpoint" in cfg:
        model.load_weights(cfg.checkpoint)
        # state = mx.load(cfg.checkpoint)  # TODO add stream?
        # model.load_state_dict(state["model"])

    return model
