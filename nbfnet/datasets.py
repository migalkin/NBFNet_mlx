import os
import os.path as osp

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_graphs.data.data import GraphData
from mlx_graphs.datasets import Dataset
from mlx_graphs.datasets.utils import download


class IndRelLinkPredDataset(Dataset):

    urls = {
        "IndFB15k237": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/valid.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt"
        ],
        "IndWN18RR": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/valid.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt"
        ],
        "IndNELL": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/valid.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s/valid.txt"
        ]
    }

    def __init__(self, root, name, version):
        self.version = version
        assert name in ["IndFB15k237", "IndWN18RR", "IndNELL"]
        assert version in ["v1", "v2", "v3", "v4"]
        super().__init__(base_dir=root, name=f"{name}/{version}")
        #self.data, self.slices = mx.load(self.processed_path[0])

    @property
    def num_relations(self):
        return self.train_data.edge_type.max().item() + 1

    @property
    def raw_dir(self):
        return os.path.join(self._base_dir, self.name, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self._base_dir, self.name, "processed")

    @property
    def processed_file_names(self):
        return "data.npz"

    @property
    def raw_file_names(self):
        return [
            "train_ind.txt", "valid_ind.txt", "test_ind.txt", "train.txt", "valid.txt"
        ]
    
    @property
    def raw_paths(self):
        files = self.raw_file_names
        return [osp.join(self.raw_dir, f) for f in files]

    def download(self):
        for url, path in zip(self.urls[self.name.split("/")[0]], self.raw_paths):
            download_path = download(url % self.version, path)
            #os.rename(osp.join(download_path, url.split("/")[-1]), path)

    def process(self):
        test_files = self.raw_paths[:3]
        train_files = self.raw_paths[3:]

        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                    t = inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)
        triplets = mx.array(triplets)

        edge_index = triplets[:, :2].T
        edge_type = triplets[:, 2]
        num_relations = edge_type.max().item() + 1

        train_fact_slice = slice(None, sum(num_samples[:1]))
        test_fact_slice = slice(sum(num_samples[:2]), sum(num_samples[:3]))
        train_fact_index = edge_index[:, train_fact_slice]
        train_fact_type = edge_type[train_fact_slice]
        test_fact_index = edge_index[:, test_fact_slice]
        test_fact_type = edge_type[test_fact_slice]
        # add flipped triplets for the fact graphs
        train_fact_index = mx.concatenate([train_fact_index, 
                                           mx.stack([train_fact_index[1], train_fact_index[0]])], axis=-1)
        train_fact_type = mx.concatenate([train_fact_type, train_fact_type + num_relations])
        test_fact_index = mx.concatenate([test_fact_index, 
                                          mx.stack([test_fact_index[1], test_fact_index[0]])], axis=-1)
        test_fact_type = mx.concatenate([test_fact_type, test_fact_type + num_relations])

        train_slice = slice(None, sum(num_samples[:1]))
        valid_slice = slice(sum(num_samples[:1]), sum(num_samples[:2]))
        test_slice = slice(sum(num_samples[:3]), sum(num_samples))
        self.train_data = GraphData(edge_index=train_fact_index, edge_type=train_fact_type, #num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, train_slice], target_edge_type=edge_type[train_slice])
        self.valid_data = GraphData(edge_index=train_fact_index, edge_type=train_fact_type, #num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, valid_slice], target_edge_type=edge_type[valid_slice])
        self.test_data = GraphData(edge_index=test_fact_index, edge_type=test_fact_type, #num_nodes=len(inv_test_entity_vocab),
                         target_edge_index=edge_index[:, test_slice], target_edge_type=edge_type[test_slice])

        # TODO GraphData is un-picklable in MLX yet, can't save
        # mx.savez_compressed(osp.join(self.processed_dir, self.processed_file_names), 
        #                     train_data=tree_flatten(train_data), valid_data=valid_data, test_data=test_data)
        

    def __repr__(self):
        return "%s()" % self.name
