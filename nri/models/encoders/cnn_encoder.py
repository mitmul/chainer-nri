import logging

import chainer
import chainer.functions as F
import chainer.links as L

from nri.models import cnn
from nri.models import mlp


class CNNEncoder(chainer.Chain):
    """ CNNEncoder module.

    Args:
        n_in (int): The dimension of feature.
        n_hid (int): Number of channels of 1D convolutions.
        n_out (int): The dimension of output vectors.
        do_prob (float): The dropout ratio.
        factor (bool): Flag for factor graph CNN encoder. The default if True.

    """

    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(CNNEncoder, self).__init__()
        self.dropout_prob = do_prob
        self.factor = factor

        # Xavier normal = Glorot normal
        w = chainer.initializers.GlorotNormal()


        with self.init_scope():
            self.cnn = cnn.CNN(n_in * 2, n_hid, n_hid, do_prob)
            self.mlp1 = mlp.MLP(n_hid, n_hid, n_hid, do_prob)
            self.mlp2 = mlp.MLP(n_hid, n_hid, n_hid, do_prob)
            self.mlp3 = mlp.MLP(n_hid * 3, n_hid, n_hid, do_prob)
            self.fc_out = L.Linear(n_hid, n_out, initialW=w, initial_bias=0.1)

        logger = logging.getLogger(__name__)
        if self.factor:
            logger.info("Using factor graph CNN encoder.")
        else:
            logger.info("Using CNN encoder.")

    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        # inputs: [batch_size, num_nodes, num_timesteps, feature_dims]
        x = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)
        # inputs: [batch_size, num_nodes, num_timesteps * feature_dims]
        # rel_rec, rel_send: [num_edges, num_nodes]
        receivers = F.matmul(rel_rec, x)
        senders = F.matmul(rel_send, x)
        # receivers: [batch_size, num_edges, num_timesteps * feature_dims]
        # senders: [batch_size, num_edges, num_timesteps * feature_dims]
        num_edges = rel_rec.shape[0]
        batch_size, num_nodes, num_timesteps, feature_dims = inputs.shape
        shape = (batch_size, num_edges, num_timesteps, feature_dims)
        receivers = F.reshape(receivers, shape)
        senders = F.reshape(senders, shape)
        # receivers: [batch_size, num_edges, num_timesteps, feature_dims]
        # senders: [batch_size, num_edges, num_timesteps, feature_dims]
        edges = F.concat([receivers, senders], axis=3)
        return edges

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        # x: [batch_size, num_edges, feature_dims * num_timesteps]
        incoming = F.matmul(rel_rec.T, x)
        return incoming / incoming.shape[1]

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = F.matmul(rel_rec, x)
        senders = F.matmul(rel_send, x)
        edges = F.concat([receivers, senders], axis=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # inputs: [batch_size, num_nodes, num_timesteps, feature_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
        edges = F.transpose(edges, (0, 1, 3, 2))
        # edges: [batch_size, num_edges, feature_dims * 2, num_timesteps]
        batch_size, num_edges, feature_dims, num_timesteps = edges.shape
        edges = F.reshape(
            edges, (batch_size * num_edges, feature_dims, num_timesteps))
        x = self.cnn(edges)
        x = F.reshape(x, (batch_size, num_edges, -1))
        # x: [batch_size, num_edges, n_hid]
        x = self.mlp1(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp2(x)

            x = self.node2edge(x, rel_rec, rel_send)
            x = F.concat((x, x_skip), axis=2)  # Skip connection
            x = self.mlp3(x)

        # x: [batch_size, num_edges, n_hid]
        x = F.reshape(x, (batch_size * num_edges, -1))
        x = self.fc_out(x)
        x = F.reshape(x, (batch_size, num_edges, -1))

        return x
