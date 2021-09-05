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
from nni import get_next_parameter
import logging
import json
import ast
import pytorch_lightning as pl

pl.seed_everything(0)

logger = logging.getLogger("NNI_logger")
NONE = None


def create_params_dict(params_file, nni_params):
    dict_params = json.load(open(params_file, "rt"))
    dict_params["graphs_data"]["features"] = ast.literal_eval(nni_params["features"])
    dict_params["graphs_data"]["adjacency_norm"] = nni_params["adjacency_norm"]
    dict_params["graphs_data"]["standardization"] = nni_params["standardization"]
    dict_params["model"]["activation"] = nni_params["activation"]
    dict_params["model"]["f"] = nni_params["f"]
    dict_params["model"]["dropout"] = nni_params["dropout"]
    dict_params["model"].update({"dropout_gat": nni_params["dropout_gat"]})
    dict_params["model"].update({"init_layers": nni_params["init_layers"]})
    dict_params["model"]["lr"] = nni_params["lr"]
    dict_params["model"]["optimizer"] = "ADAM_"
    dict_params["model"]["L2_regularization"] = nni_params["L2_regularization"]
    dict_params["model"]["GCN_layers"] = ast.literal_eval(nni_params["GCN_layers"])
    dict_params["activator"]["epochs"] = 350
    dict_params["activator"]["batch_size"] = nni_params["batch_size"]
    dict_params["activator"].update({"wait": nni_params["wait"]})
    dict_params["activator"].update({"mul": nni_params["mul"]})
    dict_params["activator"].update({"stop_sign": nni_params["stop_sign"]})
    dict_params["activator"].update({"norm_type": nni_params["norm_type"]})
    dict_params["activator"].update({"wd": nni_params["wd"]})
    
    return dict_params


def run_trial(dict_params):
    device = "cuda:2"
    
    ext_train = ExternalData(dict_params)
    ds = GraphsDataset(dict_params, external_data=ext_train)
    model = QGCN(dict_params, ds.len_features, ext_train.len_embed(), init_layers=dict_params["model"]["init_layers"])
    #model = QGCN(dict_params, ds.len_features, ext_train.len_embed(), d=dict_params["model"]["dropout_gat"])
    activator = QGCNActivator(model, dict_params, ds, nni=True, device=device)
    activator.train(show_plot=False, early_stop=False, wait=dict_params["activator"]["wait"], mul=dict_params["activator"]["mul"], stop_sign=dict_params["activator"]["stop_sign"], norm_type=dict_params["activator"]["norm_type"], wd=dict_params["activator"]["wd"])
    
    #ds = GraphsDataset(dict_params, external_data=None)
    #model = QGCN(dict_params, ds.len_features, [10])
    #model = QGCN(dict_params, ds.len_features, [10], d=dict_params["model"]["dropout_gat"])
    #activator = QGCNActivator(model, dict_params, ds, nni=True, device=device)
    #activator.train(show_plot=False, early_stop=False)


def main(params_file):
    try:
        # get parameters form tuner
        params = get_next_parameter()
        logger.debug(params)
        params_dict = create_params_dict(params_file, params)
        run_trial(params_dict)
    except Exception as exception:
        logger.error(exception)
        raise


if __name__ == "__main__":
    params_file = "../params/mutagen_params.json"
    main(params_file)
