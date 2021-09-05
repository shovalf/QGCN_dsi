import json
from torch.nn import Module, Linear, Dropout, Embedding, ModuleList, Sequential, LogSoftmax
import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

ADAM_ = Adam
SGD_ = SGD
relu_ = torch.relu
sigmoid_ = torch.sigmoid
tanh_ = torch.tanh


"""
given A, x0 : A=Adjacency_matrix, x0=nodes_vec
First_model => x1(n x k) = sigma( A(n x n) * x0(n x d) * W1(d x k) )
Bilinear_model => x2(1 x 1) = sigma( W2(1 x k) * trans(x1)(k x n) * A(n x n) * x0(n x d) * W1(d x 1) )
"""

class GCN(Module):
    def __init__(self, in_dim, out_dim, activation, dropout, init_layers="reset"):
        super(GCN, self).__init__()
        # useful info in forward function
        self._linear = Linear(in_dim, out_dim)
        if init_layers == "reset":
            self.reset_parameters(out_dim)
        else:
            self.init()
        self.reset_parameters(out_dim)
        self._activation = activation
        self._dropout = Dropout(p=dropout)
        self._gpu = True

    def reset_parameters(self, out_dim):
        stdv = 1. / math.sqrt(out_dim)
        self._linear.weight.data.uniform_(-stdv, stdv)

    def init(self):
        final_weight_tensor = torch.nn.init.xavier_normal_(self._linear.weight.data)
        self._linear.weight.data = final_weight_tensor

    def forward(self, A, x0):
        # Dropout layer
        x0 = self._dropout(x0)
        # tanh( A(n x n) * x0(n x d) * W1(d x k) )
        Ax = torch.matmul(A, x0)
         
        x = self._linear(Ax)
        x1 = self._activation(x) - 2/(x**2+1)

        #x = x = self._linear(Ax)

        #x1 = torch.log(torch.abs(self._linear(Ax)) + 1e-4)
        #x1 = self._activation(x)
        return x1


class QGCNLastLayer(Module):
    def __init__(self, left_in_dim, right_in_dim, out_dim, init_layers="reset"):
        super(QGCNLastLayer, self).__init__()
        # useful info in forward function
        self._left_linear = Linear(left_in_dim, 1)
        self._right_linear = Linear(right_in_dim, out_dim)
        if init_layers == "reset":
            self.reset_parameters(out_dim)
        else:
            self.init()
        self._gpu = False

    def reset_parameters(self, out_dim):
        stdv = 1. / math.sqrt(1)
        self._left_linear.weight.data.uniform_(-stdv, stdv)
        stdv_r = 1. / math.sqrt(out_dim)
        self._right_linear.weight.data.uniform_(-stdv_r, stdv_r)

    def init(self):
        final_weight_tensor_l = torch.nn.init.xavier_uniform_(self._left_linear.weight.data)
        self._left_linear.weight.data = final_weight_tensor_l
        final_weight_tensor_r = torch.nn.init.uniform_(self._right_linear.weight.data)
        self._right_linear.weight.data = final_weight_tensor_r

    def forward(self, A, x0, x1):
        # sigmoid( W2(1 x k) * trans(x1)(k x n) * A(n x n) * x0(n x d) * W1(d x 1) )
        x1_A = torch.matmul(x1.permute(0, 2, 1), A)
        W2_x1_A = self._left_linear(x1_A.permute(0, 2, 1))
        W2_x1_A_x0 = torch.matmul(W2_x1_A.permute(0, 2, 1), x0)
        W2_x1_A_x0_W3 = self._right_linear(W2_x1_A_x0)
        return W2_x1_A_x0_W3.squeeze(dim=1)


class QGCN(Module):
    """
    first linear layer is executed numerous times
    """

    def __init__(self, params, in_dim, embed_vocab_dim, init_layers="reset"):
        super(QGCN, self).__init__()
        self._params = params["model"] if type(params) is dict else json.load(open(params, "rt"))["model"]

        # add embedding layers
        self._is_binary = True if self._params["label_type"] == "binary" else False
        self._is_embed = True if self._params["use_embeddings"] == "True" else False

        qgcn_layers_dim = [{"in": layer["in_dim"], "out": layer["out_dim"]} for layer in self._params["GCN_layers"]]
        qgcn_layers_dim[0]["in"] = in_dim

        if self._is_embed:
            # embeddings are added to ftr vector -> update dimensions of relevant weights
            qgcn_layers_dim[0]["in"] += sum(self._params["embeddings_dim"])

            # add embedding layers
            self._embed_layers = []
            for vocab_dim, embed_dim in zip(embed_vocab_dim, self._params["embeddings_dim"]):
                self._embed_layers.append(Embedding(vocab_dim, embed_dim))
            self._embed_layers = ModuleList(self._embed_layers)
            
        values_out = [qgcn_layers_dim[i]["out"] for i in range(len(qgcn_layers_dim))]
        values_out.append(qgcn_layers_dim[0]["in"])
        sum_layers_shape = sum(values_out)

        self._num_layers = len(self._params["GCN_layers"])
        self._linear_layers = []
        # create linear layers
        self._linear_layers = ModuleList([GCN(qgcn_layers_dim[i]["in"], qgcn_layers_dim[i]["out"],
                                              globals()[self._params["activation"]], self._params["dropout"], 
                                              init_layers=init_layers) for i in range(self._num_layers)])
        # def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
        # self._linear_layers = ModuleList([GCNII(qgcn_layers_dim[i]["in"], 64, qgcn_layers_dim[0]["out"], qgcn_layers_dim[1]["out"],
        #                                       self._params["dropout"], 0.5, 0.1, True)
        #                                   for i in range(1)])
        if self._params["f"] == "x1_x0":
            qgcn_right_in = in_dim + sum(self._params["embeddings_dim"])
            qgcn_left_in = qgcn_layers_dim[-1]["out"]
        elif self._params["f"] == "x1_x1":
            qgcn_right_in = qgcn_layers_dim[-1]["out"]
            qgcn_left_in = qgcn_layers_dim[-1]["out"]
        elif self._params["f"] == "c_x1":
            qgcn_right_in = qgcn_layers_dim[-1]["out"]
            qgcn_left_in = sum_layers_shape
        else:
            qgcn_right_in = in_dim + sum(self._params["embeddings_dim"])
            qgcn_left_in = qgcn_layers_dim[-1]["out"]

        self._qgcn_last_layer = QGCNLastLayer(left_in_dim=qgcn_left_in, right_in_dim=qgcn_right_in,
                                              out_dim=1 if self._is_binary else self._params["num_classes"], 
                                              init_layers=init_layers)
        self._softmax = LogSoftmax(dim=1)
        self.optimizer = self.set_optimizer(self._params["lr"], globals()[self._params["optimizer"]], self._params["L2_regularization"])

    def set_optimizer(self, lr, opt, weight_decay):
        return opt(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _fix_shape(self, input):
        return input if len(input.shape) == 3 else input.unsqueeze(dim=0)

    def forward(self, A, x0, embed, test=False):
        if self._is_embed:
            list_embed = []
            for i, embedding in enumerate(self._embed_layers):
                list_embed.append(embedding(embed[:, :, i]))
            x0 = torch.cat([x0] + list_embed, dim=2)

        x1 = x0
        H_layers = [x1]
        for i in range(self._num_layers):
            x1 = self._linear_layers[i](A, x1)
            H_layers.append(x1)

        c = torch.cat(H_layers, dim=2)

        if self._params["f"] == "x1_x0":
            x2 = self._qgcn_last_layer(A, x0, x1)
        elif self._params["f"] == "x1_x1":
            x2 = self._qgcn_last_layer(A, x1, x1)
        elif self._params["f"] == "c_x1":
            x2 = self._qgcn_last_layer(A, x1, c)
        else: # like "x1_x0"
            x2 = self._qgcn_last_layer(A, x0, x1)

        # x2 = self._qgcn_last_layer(A, x0, x1)
        if test:
            if self._is_binary:
                x2 = torch.sigmoid(x2)
            else:
                x2 = self._softmax(x2)
        return x2


if __name__ == "__main__":
    from dataset.dataset_external_data import ExternalData
    from dataset.dataset_graphs_model import GraphsDataset

    params_file = "../params/default_binary_params.json"
    ext_train = ExternalData(params_file)
    ds = GraphsDataset(params_file, external_data=ext_train)
    dl = DataLoader(
        dataset=ds,
        collate_fn=ds.collate_fn,
        batch_size=64,
    )

    module = QGCN(params_file, ds.len_features, ext_train.len_embed())
    # module = BilinearModule(BilinearModuleParams())
    for i, (_A, _D, _x0, _l) in enumerate(dl):
        _x2 = module(_A, _D, _x0)
        print(i, "/", len(dl), _x2.shape)
        e = 0
