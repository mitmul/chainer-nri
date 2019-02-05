import chainer
import chainer.links as L
import chainer.functions as F

import math
class CNN(chainer.Chain):
    """ CNN module.

    The dimension of input sequences is reduced by 1D convolutions and 1d max pooling.

    """

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(CNN, self).__init__()
        self.dropout_prob = do_prob

        # w = self._conv1d_weight_initializer
        w = chainer.initializers.GlorotNormal()

        with self.init_scope():
            self.conv1 = L.Convolution1D(n_in, n_hid, ksize=5, stride=1, pad=0, initialW=w, initial_bias=0.1)
            self.bn1 = L.BatchNormalization(n_hid)
            self.conv2 = L.Convolution1D(n_hid, n_hid, ksize=5, stride=1, pad=0, initialW=w, initial_bias=0.1)
            self.bn2 = L.BatchNormalization(n_hid)
            self.conv_predict = L.Convolution1D(n_hid, n_out, ksize=1, initialW=w, initial_bias=0.1)
            self.conv_attention = L.Convolution1D(n_hid, 1, ksize=1, initialW=w, initial_bias=0.1)

    def _conv1d_weight_initializer(self, array):
        n_hid, n_in, ksize = array.shape
        return chainer.initializers.Normal(math.sqrt(2. / (ksize * n_hid)))

    def forward(self, inputs):
        # inputs: [batch_size * num_edges, feature_dims, num_timesteps]
        x = self.bn1(F.relu(self.conv1(inputs)))
        x = F.dropout(x, self.dropout_prob)
        x = F.max_pooling_1d(x, 2)
        x = self.bn2(F.relu(self.conv2(x)))
        pred = self.conv_predict(x)
        # pred: [batch_size * num_edges, n_out, processed_timesteps]
        attention = F.softmax(self.conv_attention(x), axis=2)
        edge_prob = F.mean(pred * attention, axis=2)
        # edge_prob: [batch_size * num_edges, n_out]
        return edge_prob
