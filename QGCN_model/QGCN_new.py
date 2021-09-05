import json
from torch.nn import Module, Linear, Dropout, Embedding, ModuleList, Sequential, LogSoftmax
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader


ADAM_ = Adam
SGD_ = SGD
relu_ = torch.relu
sigmoid_ = torch.sigmoid
tanh_ = torch.tanh
leaky_ = nn.LeakyReLU(0.2)


"""
given A, x0 : A=Adjacency_matrix, x0=nodes_vec
First_model => x1(n x k) = sigma( A(n x n) * x0(n x d) * W1(d x k) )
Bilinear_model => x2(1 x 1) = sigma( W2(1 x k) * trans(x1)(k x n) * A(n x n) * x0(n x d) * W1(d x 1) )
"""


class GraphAttentionLayer(Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, activation, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.activation = activation

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        my_input = torch.matmul(a_input, self.a).squeeze(2)
        e = self.activation(my_input) - 2/(my_input**2+1)

        zero_vec = -9e15*torch.ones_like(e)
        #print(zero_vec.shape)
        #print(adj.shape)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime) - 2/(h_prime**2+1)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
        

class GAT(Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, activation):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, activation=activation, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, activation=activation, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x
    


class GCN(Module):
    def __init__(self, in_dim, out_dim, activation, dropout):
        super(GCN, self).__init__()
        # useful info in forward function
        self._linear = Linear(in_dim, out_dim)
        self._activation = activation
        self._dropout = Dropout(p=dropout)
        self._gpu = True

    def forward(self, A, x0):
        # Dropout layer
        x0 = self._dropout(x0)
        # tanh( A(n x n) * x0(n x d) * W1(d x k) )
        Ax = torch.matmul(A, x0)
         
        x = self._linear(Ax)
        x1 = self._activation(x) - 2/(x**2+1)

        #x = x = self._linear(Ax)

        #x = torch.log(torch.abs(self._linear(Ax)) + 1e-4)
        #x1 = self._activation(x)
        return x1


class QGCNLastLayer(Module):
    def __init__(self, left_in_dim, right_in_dim, out_dim):
        super(QGCNLastLayer, self).__init__()
        # useful info in forward function
        self._left_linear = Linear(left_in_dim, 1)
        self._right_linear = Linear(right_in_dim, out_dim)
        self._gpu = False

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

    def __init__(self, params, in_dim, embed_vocab_dim, d):
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
        self._linear_layers = GAT(qgcn_layers_dim[0]["in"], 8, qgcn_layers_dim[-1]["out"], d, 0.2, 8, globals()[self._params["activation"]])
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
                                              out_dim=1 if self._is_binary else self._params["num_classes"])
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
        
        x0 = x0.reshape(x0.shape[0]*x0.shape[1], x0.shape[2])
        A = A.reshape(A.shape[0]*A.shape[1], A.shape[2])
        #print(x0.shape)
        #print(A.shape)
        #my_list = []
        #batch_size, num_nodes = A.shape[0], A.shape[1]
        #for j in range(A.shape[0]):
         #   adj = A[j, :, :]
          #  x0_new = x0[j, :, :]
           # x1 = self._linear_layers(x0_new, adj)
           # my_list.append(x1)
        #x1 = torch.cat((my_list), axis=0)
        #hid = x1.shape[1]
        #x1 = x1.reshape(batch_size, num_nodes, hid)
            
            
        x1 = self._linear_layers(x0, A)
        H_layers = [x1]
        x1 = torch.unsqueeze(x1, 0)
        #c = torch.cat(H_layers, dim=2)

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
            x2 = torch.sigmoid(x2) if self._is_binary else self._softmax(x2)
        else:
            if not self._is_binary:
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
