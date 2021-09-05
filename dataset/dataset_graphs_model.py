"""
Here is the implementation of helpful class and functions that handle the given dataset.
"""

import json
from scipy.stats import zscore
import scipy.sparse as sp
from torch import Tensor
from torch.nn import ConstantPad2d
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from feature_calculators import FeatureMeta
from features_processor import FeaturesProcessor, log_norm
from graph_features import GraphFeatures
from loggers import PrintLogger
from multi_graph import MultiGraph
from dataset.dataset_external_data import ExternalData
import os
import pandas as pd
import networkx as nx
import pickle
import numpy as np
from vertices.betweenness_centrality import BetweennessCentralityCalculator
from vertices.bfs_moments import BfsMomentsCalculator
from sklearn.preprocessing import MinMaxScaler


# some important shortenings
PKL_DIR = "pkl"
NORM_REDUCED = "NORM_REDUCED"
NORM_REDUCED_SYMMETRIC = "NORM_REDUCED_SYMMETRIC"
IDENTITY = "IDENTITY"
RAW_FORM = "RAW_FORM"
DEG = "DEG"
IN_DEG = "IN_DEG"
OUT_DEG = "OUT_DEG"
CENTRALITY = ("betweenness_centrality", FeatureMeta(BetweennessCentralityCalculator, {"betweenness"}))
BFS = ("bfs_moments", FeatureMeta(BfsMomentsCalculator, {"bfs"}))


class GraphsDataset(Dataset):
    def __init__(self, params, external_data: ExternalData = None):
        self._params = params if type(params) is dict else json.load(open(params, "rt"))
        self._dataset_name = self._params["dataset_name"]
        self._params = self._params["graphs_data"]
        self._logger = PrintLogger("logger")
        if self._params["directed"] == "False":
            self._params["directed"] = False
        else:
            self._params["directed"] = True
        # path to base directory
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0], "..")

        self._external_data = external_data
        # init ftr_meta dictionary and other ftr attributes
        self._init_ftrs()
        self._src_file_path = os.path.join(self._params["file_path"])
        self._multi_graph, self._labels, self._label_to_idx, self._idx_to_label = self._build_multi_graph()
        self._data, self._idx_to_name = self._build_data()

    @property
    def all_labels(self):
        return self._idx_to_label

    @property
    def label_count(self):
        return Counter([v[3] for name, v in self._data.items()])

    def label(self, idx):
        return self._data[self._idx_to_name[idx]][3]

    @property
    def len_features(self):
        return self._data[self._idx_to_name[0]][1].shape[1]

    def _init_ftrs(self):
        self._deg, self._in_deg, self._out_deg, self._is_ftr, self._ftr_meta = False, False, False, False, {}
        self._is_external_data = False if self._external_data is None else True
        # params.FEATURES contains string and list of two elements (matching to key: value)
        # should Deg/In-Deg/Out-Deg be calculated
        for ftr in self._params["features"]:
            ftr = globals()[ftr]
            if ftr == DEG:
                self._deg = True
            elif ftr == IN_DEG:
                self._in_deg = True
            elif ftr == OUT_DEG:
                self._out_deg = True
            else:
                self._ftr_meta[ftr[0]] = ftr[1]

        # add directories for pickles
        if len(self._ftr_meta) > 0:
            self._ftr_path = os.path.join(self._base_dir, PKL_DIR, "ftr", self._dataset_name)
            if not os.path.exists(self._ftr_path):
                os.makedirs(self._ftr_path)
            self._is_ftr = True

    """
    build multi graph according to csv 
    each community is a single graph, no consideration to time
    """
    def _build_multi_graph(self):
        path_pkl = os.path.join(self._base_dir, PKL_DIR, self._dataset_name + "_split_" +
                                str(self._params["percentage"]) + "_mg.pkl")
        if os.path.exists(path_pkl):
            return pickle.load(open(path_pkl, "rb"))
        multi_graph_dict = {}
        labels = {}
        label_to_idx = {}
        # open basic data csv (with all edges of all times)
        data_df = pd.read_csv(self._src_file_path)
        stop = data_df.shape[0] * self._params["percentage"]

        for index, edge in data_df.iterrows():
            if index > stop:
                break
            # write edge to dictionary
            graph_id = str(edge[self._params["graph_col"]])
            src = str(edge[self._params["src_col"]])
            dst = str(edge[self._params["dst_col"]])
            multi_graph_dict[graph_id] = multi_graph_dict.get(graph_id, []) + [(src, dst)]
            label = edge[self._params["label_col"]]
            label_to_idx[label] = len(label_to_idx) if label not in label_to_idx else label_to_idx[label]
            labels[graph_id] = label_to_idx[label]

        mg = MultiGraph(self._dataset_name, graphs_source=multi_graph_dict,
                        directed=self._params["directed"], logger=self._logger)
        idx_to_label = [l for l in sorted(label_to_idx, key=lambda x: label_to_idx[x])]
        mg.suspend_logger()
        os.makedirs(os.path.join(self._base_dir, PKL_DIR), exist_ok=True)
        pickle.dump((mg, labels, label_to_idx, idx_to_label), open(path_pkl, "wb"))
        mg.wake_logger()
        return mg, labels, label_to_idx, idx_to_label

    """
    returns a vector x for gnx 
    basic version returns degree for each node
    """
    def _gnx_vec(self, gnx_id, gnx: nx.Graph, node_order):
        final_vec = []
        if self._deg:
            degrees = gnx.degree(gnx.nodes)
            final_vec.append(np.matrix([np.log(degrees[d] + 1e-3) for d in node_order]).T)
        if self._in_deg:
            degrees = gnx.in_degree(gnx.nodes)
            final_vec.append(np.matrix([np.log(degrees[d] + 1e-3) for d in node_order]).T)
        if self._out_deg:
            degrees = gnx.out_degree(gnx.nodes)
            final_vec.append(np.matrix([np.log(degrees[d] + 1e-3) for d in node_order]).T)
        if self._is_external_data and self._external_data.is_continuous:
            final_vec.append(np.matrix([self._external_data.continuous_feature(gnx_id, d) for d in node_order]))
        if self._is_ftr:
            name = str(gnx_id)
            gnx_dir_path = os.path.join(self._ftr_path, name)
            if not os.path.exists(gnx_dir_path):
                os.mkdir(gnx_dir_path)
            raw_ftr = GraphFeatures(gnx, self._ftr_meta, dir_path=gnx_dir_path, is_max_connected=False,
                                    logger=PrintLogger("logger"))
            raw_ftr.build(should_dump=True)  # build features
            final_vec.append(FeaturesProcessor(raw_ftr).as_matrix(norm_func=log_norm))

        return np.hstack(final_vec)

    def _degree_matrix(self, gnx, nodelist):
        degrees = gnx.degree(gnx.nodes)
        return np.diag([1/degrees[d] for d in nodelist])

    def _standardize_data(self, data):
        all_data_continuous_vec = []                # stack all vectors for all graphs
        key_to_idx_map = []                     # keep ordered list (g_id, num_nodes) according to stack order

        # stack
        for g_id, (A, gnx_vec, embed_vec, label) in data.items():
            all_data_continuous_vec.append(gnx_vec)
            key_to_idx_map.append((g_id, gnx_vec.shape[0]))  # g_id, number of nodes ... ordered
        all_data_continuous_vec = np.vstack(all_data_continuous_vec)

        # z-score data
        if self._params["standardization"] == "zscore":
            standardized_data = zscore(all_data_continuous_vec, axis=0)
        elif self._params["standardization"] == "scale":
            pass
        elif self._params["standardization"] == "min_max":
            scalar = MinMaxScaler()
            standardized_data = scalar.fit_transform(all_data_continuous_vec)

        # rebuild data to original form -> split stacked matrix according to <list: (g_id, num_nodes)>
        new_data_dict = {}
        start_idx = 0
        for g_id, num_nodes in key_to_idx_map:
            new_data_dict[g_id] = (data[g_id][0], standardized_data[start_idx: start_idx+num_nodes],
                                   data[g_id][2], data[g_id][3])
            start_idx += num_nodes

        return new_data_dict

    def _norm_adjacency(self, A, gnx, node_order):
        if self._params["adjacency_norm"] == NORM_REDUCED:
            # D^-0.5 A D^-0.5
            D = self._degree_matrix(gnx, nodelist=node_order)
            D_sqrt = np.matrix(np.sqrt(D))
            adjacency = D_sqrt * np.matrix(A) * D_sqrt

        elif self._params["adjacency_norm"] == NORM_REDUCED_SYMMETRIC:
            # D^-0.5 [A + A.T + I] D^-0.5
            D = self._degree_matrix(gnx, nodelist=node_order)
            D_sqrt = np.matrix(np.sqrt(D))
            adjacency = D_sqrt * np.matrix(A + A.T + np.identity(A.shape[0])) * D_sqrt
        elif self._params["adjacency_norm"] == IDENTITY:
            adjacency = np.identity(A.shape[0])
        elif self._params["adjacency_norm"] == RAW_FORM:
            adjacency = A
        else:
            print("Error in adjacency_norm: " + self._params["adjacency_norm"] + "is not a valid option")
            exit(1)
        return adjacency

    """
    builds a data dictionary
    { ... graph_name: ( A = Adjacency_matrix, x = graph_vec, label ) ... }  
    """
    def _build_data(self):
        ext_data_id = "None" if not self._is_external_data else "_embed_ftr_" + "_".join(self._external_data.embed_headers)\
                                                                + "_continuous_ftr_" + "_".join(self._external_data.continuous_headers) \
                                                                + "standardization_" + self._params["standardization"]
        pkl_path = os.path.join(self._base_dir, PKL_DIR, self._dataset_name + ext_data_id + "_data.pkl")
        if os.path.exists(pkl_path):
            return pickle.load(open(pkl_path, "rb"))
        data = {}
        idx_to_name = []

        for gnx_id, gnx in zip(self._multi_graph.graph_names(), self._multi_graph.graphs()):
            # if gnx.number_of_nodes() < 5:
            #     continue
            node_order = sorted(list(gnx.nodes()))
            idx_to_name.append(gnx_id)
            
            if self._params["adjacency_norm"] == NORM_REDUCED_SYMMETRIC:
                adjacency = nx.adjacency_matrix(gnx, nodelist=node_order).todense()
                #adjacency = self._norm_adjacency(nx.adjacency_matrix(gnx, nodelist=node_order).todense(), gnx, node_order)
                adj = sp.coo_matrix(adjacency)
                adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
                adj = normalize(adj + sp.eye(adj.shape[0]))
                adjacency = adj.toarray()
            else:
                adjacency = self._norm_adjacency(nx.adjacency_matrix(gnx, nodelist=node_order).todense(), gnx, node_order)

            gnx_vec = self._gnx_vec(gnx_id, gnx, node_order)
            embed_vec = [self._external_data.embed_feature(gnx_id, d) for d in node_order] \
                if self._is_external_data and self._external_data.is_embed else None

            data[gnx_id] = (adjacency, gnx_vec, embed_vec, self._labels[gnx_id])

        data = self._standardize_data(data)
        pickle.dump((data, idx_to_name), open(pkl_path, "wb"))
        return data, idx_to_name

    def collate_fn(self, batch):
        lengths_sequences = []
        # calculate max word len + max char len
        for A, x, e, l in batch:
            lengths_sequences.append(A.shape[0])

        # in order to pad all batch to a single dimension max length is needed
        seq_max_len = np.max(lengths_sequences)

        # new batch variables
        adjacency_batch = []
        x_batch = []
        embeddings_batch = []
        labels_batch = []
        for A, x, e, l in batch:
            # pad word vectors
            adjacency_pad = ConstantPad2d((0, seq_max_len - A.shape[0], 0, seq_max_len - A.shape[0]), 0)
            adjacency_batch.append(adjacency_pad(A).tolist())
            vec_pad = ConstantPad2d((0, 0, 0, seq_max_len - A.shape[0]), 0)
            x_batch.append(vec_pad(x).tolist())
            embeddings_batch.append(vec_pad(e).tolist() if self._is_external_data and self._external_data.is_embed else e)
            labels_batch.append(l)

        return Tensor(adjacency_batch), Tensor(x_batch), Tensor(embeddings_batch).long(), Tensor(labels_batch).long()

    def __getitem__(self, index):
        gnx_id = self._idx_to_name[index]
        A, x, embed, label = self._data[gnx_id]
        embed = 0 if embed is None else Tensor(embed).long()
        return Tensor(A), Tensor(x), embed, label

    def __len__(self):
        return len(self._idx_to_name)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


if __name__ == "__main__":
    ext_train = ExternalData("../params/default_multiclass_params.json")
    ds = GraphsDataset("../params/default_multiclass_params.json", external_data=ext_train)
    # ds = BilinearDataset(AidsDatasetTestParams())
    dl = DataLoader(
        dataset=ds,
        collate_fn=ds.collate_fn,
        batch_size=64,
    )
    p = []
    for i, (A, x, e, l) in enumerate(dl):
        print(i, A, x, e, l)
    e = 0
