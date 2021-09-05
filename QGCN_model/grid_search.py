import sys
import os
f = open("curr_pwd", "wt")
cwd = os.getcwd()
f.write(cwd)
f.close()

sys.path.insert(1, os.path.join(cwd, ".."))
sys.path.insert(1, os.path.join(cwd, "..", "..", "graph-measures"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "graph-measures", "features_algorithms"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "graph-measures", "graph_infra"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "graph-measures", "features_infra"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "graph-measures", "features_meta"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "graph-measures", "features_algorithms", "vertices"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "graphs-package", "features_processor"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "graphs-package", "multi_graph"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "graphs-package", "temporal_graphs"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "graphs-package", "features_processor", "motif_variations"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "graphs-package"))

from QGCN_model.qgcn_activator import QGCNActivator
from QGCN_model.QGCN import QGCN
from dataset.dataset_external_data import ExternalData
from dataset.dataset_graphs_model import GraphsDataset
from datetime import datetime
import logging
import json
import ast
import pytorch_lightning as pl

pl.seed_everything(0)


def create_params_dict(params_file, features, lr, dropout, l2, gcn_layers, standardization):
    dict_params = json.load(open(params_file, "rt"))
    dict_params["graphs_data"]["features"] = features
    dict_params["graphs_data"]["standardization"] = standardization
    dict_params["model"]["dropout"] = dropout
    dict_params["model"]["lr"] = lr
    dict_params["model"]["L2_regularization"] = l2
    dict_params["model"]["GCN_layers"] = ast.literal_eval(gcn_layers)
    return dict_params
    

def create_params_dict2(params_file, adj_norm, lr, dropout, l2, gcn_layers, standardization):
    dict_params = json.load(open(params_file, "rt"))
    dict_params["graphs_data"]["standardization"] = standardization
    dict_params["model"]["dropout"] = dropout
    dict_params["model"]["lr"] = lr
    dict_params["model"]["L2_regularization"] = l2
    dict_params["graphs_data"]["adjacency_norm"] = adj_norm
    dict_params["model"]["GCN_layers"] = ast.literal_eval(gcn_layers)
    return dict_params


def main1(params_file, device, name, fr, i):
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    csv_file = "../grid_results/{}/{} grid search results_{} {}.csv".format(name, name, i, curr_time)
    csv_columns = ["features", "lr", "dropout", "l2", "layer1", "layer2", "standardization", "acc"]
    
    f = open(csv_file, "wt")
    header = csv_columns
    f.write(",".join(header) + "\n")
    f.close()
    
    #features = [["DEG", "CENTRALITY", "BFS"], ["DEG", "BFS"], ["BFS", "CENTRALITY"], ["DEG", "CENTRALITY"]]
    lrrr = [5e-3, 5e-4, 1e-4]
    dropout = [0.2, 0.5]
    l2 = [5e-3, 1e-4, 5e-4]
    gcn_layers = ["[{\"in_dim\": \"None\", \"out_dim\": 250},{\"in_dim\": 250, \"out_dim\": 100}]", "[{\"in_dim\": \"None\", \"out_dim\": 500},{\"in_dim                        \": 500, \"out_dim\": 250}]"]
    standardization = ["min_max", "zscore"]
    batch_size = 64
    optimizer = "ADAM_"
    activation = "tanh_"
    adjacency_norm = "NORM_REDUCED_SYMMETRIC"
    
    for lr in lrrr:
        for regr in l2:
            for d in dropout:
                for gl in gcn_layers:
                    for std in standardization:
                        dict_params = create_params_dict(params_file, fr, lr, d, regr, gl, std)
                        if name == "mutagen" or name=="grec":
                            ext_train = ExternalData(params_file)
                            ds = GraphsDataset(params_file, external_data=ext_train)
                            model = QGCN(params_file, ds.len_features, ext_train.len_embed())
                        else:
                            ds = GraphsDataset(params_file, external_data=None)
                            model = QGCN(params_file, ds.len_features, [10])
                        activator = QGCNActivator(model, params_file, ds, device=device, grid=True)
                        acc = activator.train()
                        f = open(csv_file, "a")
                        layer1 = gl.split(":")[2].split(")")[0].split("}")[0].split(" ")[1]
                        layer2 = gl.split(":")[4].split("}")[0].split(" ")[1]
                        my_f = "".join(fr[i] + ":"for i in range(len(fr)))
                        data = {"features": my_f, "lr": lr, "dropout": d, "l2": regr, "layer1": layer1, "layer2": layer2, "standardization": std, "acc":                                     acc}
                        f.write(",".join([str(x) for x in list(data.values())]) + "\n")  # parameters
                        f.close()
                        

def main2(params_file, device, name, lr, i):
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    csv_file = "../grid_results/{}/{} grid search results_{} {}.csv".format(name, name, i, curr_time)
    csv_columns = ["adj_norm", "lr", "dropout", "l2", "layer1", "layer2", "standardization", "acc"]
    
    f = open(csv_file, "wt")
    header = csv_columns
    f.write(",".join(header) + "\n")
    f.close()
    
    #lrrr = [5e-3, 5e-4, 1e-4]
    dropout = [0.2, 0.5]
    l2 = [5e-3, 1e-4, 5e-4]
    gcn_layers = ["[{\"in_dim\": \"None\", \"out_dim\": 250},{\"in_dim\": 250, \"out_dim\": 250}]", "[{\"in_dim\": \"None\", \"out_dim\": 200},{\"in_dim                        \": 200, \"out_dim\": 100}]"]
    standardization = ["min_max", "zscore"]
    batch_size = 64
    optimizer = "ADAM_"
    activation = "tanh_"
    adjacency_norm = ["NORM_REDUCED_SYMMETRIC", "NORM_REDUCED"]
    
    for regr in l2:
        for d in dropout:
            for gl in gcn_layers:
                for std in standardization:
                    for adj_norm in adjacency_norm:
                        dict_params = create_params_dict2(params_file, adj_norm, lr, d, regr, gl, std)
                        if name == "mutagen" or name=="grec":
                            ext_train = ExternalData(params_file)
                            ds = GraphsDataset(params_file, external_data=ext_train)
                            model = QGCN(params_file, ds.len_features, ext_train.len_embed())
                        else:
                            ds = GraphsDataset(params_file, external_data=None)
                            model = QGCN(params_file, ds.len_features, [10])
                        activator = QGCNActivator(model, params_file, ds, device=device, grid=True)
                        acc = activator.train()
                        f = open(csv_file, "a")
                        layer1 = gl.split(":")[2].split(")")[0].split("}")[0].split(" ")[1]
                        layer2 = gl.split(":")[4].split("}")[0].split(" ")[1]
                        data = {"adj_norm": adj_norm, "lr": lr, "dropout": d, "l2": regr, "layer1": layer1, "layer2": layer2, "standardization": std,                                           "acc": acc}
                        f.write(",".join([str(x) for x in list(data.values())]) + "\n")  # parameters
                        f.close()
                        

def main(params_file, device, name, i):
    features = [["DEG", "CENTRALITY", "BFS"], ["DEG", "BFS"], ["BFS", "CENTRALITY"], ["DEG", "CENTRALITY"]]
    #lr = [5e-3, 5e-4, 1e-4]
    lr = [1e-3, 5e-3, 8e-5]
    if name == "mutagen":
        main1(params_file, device, name, features[i], i)
    else:
        main2(params_file, device, name, lr[i], i)
    



if __name__ == "__main__":
    device = "cuda:4"
    params_file = "../params/NCI1_params.json"
    main(params_file, device, "NCI1", 2)
