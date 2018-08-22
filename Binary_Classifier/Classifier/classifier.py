import os
from tfcore.interfaces.IModel import IModel, IModel_Params
from tfcore.core.layer import *
from tfcore.core.activations import *
from tfcore.core.loss import *
from tfcore.utilities.utils import pad_borders, get_patches


class Classifier_Params(IModel_Params):
    """
    Parameter class for ExampleModel
    """

    def __init__(self,
                 activation='relu',
                 normalization='IN',
                 scope='Classifier',
                 name='Classifier'):
        super().__init__(scope=scope, name=name)

        self.activation = activation
        self.normalization = normalization
        self.path = os.path.realpath(__file__)


class Classifier_Model(IModel):
    """
    Example of a simple 3 layer generator model for super-resolution
    """

    def __init__(self, sess, params, global_steps, is_training):
        """
        Init of Example Class

        # Arguments
            sess: Tensorflow-Session
            params: Instance of ExampleModel_Params
            global_steps: Globel steps for optimizer
        """
        super().__init__(sess, params, global_steps)
        self.model_name = self.params.name
        self.activation = get_activation(name='relu')
        self.normalization = get_normalization(self.params.normalization)
        self.is_training = is_training

    def build_model(self, input, is_train=False, reuse=False):
        """
        Build model and create summary

        # Arguments
            input: Input-Tensor
            is_train: Bool
            reuse: Bool

        # Return
            Tensor of dimension 4D
        """
        self.reuse = reuse
        super().build_model(input, is_train, reuse)

        return self.probs

    def model(self, net, is_train=False, reuse=False):
        """
        Create generator model

        # Arguments
            input: Input-Tensor
            is_train: Bool
            reuse: Bool

        # Return
            Tensor of dimension 4D
        """

        def down_block(net, scope, conv_count, f_out, k_size=3, activation=tf.nn.relu, normalization=None, is_training=False):
            with tf.variable_scope(scope):
                for i in range(conv_count):
                    net = conv2d(net,
                                 f_out=f_out,
                                 k_size=k_size,
                                 stride=1,
                                 activation=activation,
                                 normalization=normalization,
                                 padding='VALID',
                                 is_training=is_training,
                                 reuse=self.reuse,
                                 use_weight_decay=True,
                                 use_bias=True,
                                 name='conv_' + str(i+1))

                net = max_pool(net)
                net = dropout(net, 0.5, train=self.is_training)
            return net

        with tf.variable_scope(self.params.scope, reuse=tf.AUTO_REUSE):

            net = conv2d(net,
                         f_out=16,
                         k_size=5,
                         stride=1,
                         activation=self.activation,
                         normalization=self.normalization,
                         padding='VALID',
                         is_training=self.is_training,
                         reuse=self.reuse,
                         use_weight_decay=True,
                         use_bias=True,
                         name='conv_in')
            net = max_pool(net)
            net = dropout(net, 0.5, train=self.is_training)

            layer = [1,1,1]
            f_out = [16, 32,64]
            k_size = [5,5,3,3]
            for n in range(len(layer)):
                net = down_block(net,
                                 scope='stage' + str(n + 1),
                                 conv_count=layer[n],
                                 f_out=f_out[n],
                                 k_size=k_size[n],
                                 activation=self.activation,
                                 normalization=self.normalization,
                                 is_training=self.is_training)

            net = tf.layers.flatten(net)

            net = linear_layer(net,
                               512,
                               activation=self.activation,
                               use_weight_decay=True,
                               is_training=self.is_training,
                               scope='linear_1')
            net = dropout(net, 0.5, train=self.is_training)

            net = linear_layer(net,
                               512,
                               activation=self.activation,
                               use_weight_decay=True,
                               is_training=self.is_training,
                               scope='linear_2')
            net = dropout(net, 0.5, train=self.is_training)


            net = linear_layer(net, 2, scope='linear_out')

            self.logits = net
            self.probs = tf.nn.softmax(net)

        print(' [*] ResNet loaded...')
        return self.logits

    def loss(self, Y, normalize=False, name='MSE'):

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=self.logits))

        regularizer_L2 = tf.add_n(tf.get_collection('losses'))
        self.total_loss = tf.reduce_mean(loss + 0.001 * regularizer_L2)

        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(self.probs, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.summary.append(tf.summary.scalar("accuracy_train", accuracy))
        self.summary_val.append(tf.summary.scalar("accuracy_test", accuracy))

        self.summary.append(tf.summary.scalar("cross_entropy_train", loss))
        self.summary_val.append(tf.summary.scalar("cross_entropy_test", loss))
        self.summary.append(tf.summary.scalar("Learning rate", self.learning_rate))

        self.summary.append(tf.summary.scalar("total_loss_train", self.total_loss))
        self.summary_val.append(tf.summary.scalar("total_loss_test", self.total_loss))

        return self.total_loss
