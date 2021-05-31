import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope
from recognition.lpr_util import NUM_CHARS

class LPRNet(object):

    def __init__(self, training):
        self._training = training

    def build_net(self, inputs, version='v2'):
        return self.get_train_model_multitask_v3(inputs)


    def conv_layer(self, inputs, filter, kernel, stride=1, padding='SAME', layer_name="conv"):
        with tf.variable_scope(layer_name):
            network = tf.layers.conv2d(inputs=inputs, use_bias=False, filters=filter, 
                                       kernel_size=kernel, strides=stride, padding=padding)
            return network

    def batch_normalization(self, x, training, scope):
        with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
            return tf.cond(tf.constant(training),
                           lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                           lambda : batch_norm(inputs=x, is_training=training, reuse=True))

    def drop_out(self, x, rate, training) :
        return tf.layers.dropout(inputs=x, rate=rate, training=training)

    def sigmoid(self, x):
        return tf.nn.sigmoid(x)
    def relu(self, x):
        return tf.nn.relu(x)

    def average_pooling(self, x, pool_size=[2,2], stride=[1,1], padding='VALID'):
        return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

    def max_pooling(self, x, pool_size=[3,3], stride=[1,1], padding='VALID'):
        return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

    def concatenation(self, layers) :
        return tf.concat(layers, axis=3)

    def linear(self, x, units) :
        return tf.layers.dense(inputs=x, units=units, name='linear')

    def fully_connected(self, x, units, activation, layer_name='fully_connected'):
        with tf.variable_scope(layer_name) :
            return tf.layers.dense(inputs=x, units=units, activation=activation, use_bias=True)


    def small_inception_block(self, x, om, scope='incep_block'):
        #参考squeezenet的fire module，先squeeze，然后expand
        with tf.variable_scope(scope): 
            x = self.conv_layer(x, int(om/4), kernel=[1,1], layer_name = scope+'_conv1')
            x = self.relu(x)
            #参考inception v3
            #branch1
            x1 = self.conv_layer(x, int(om/4), kernel=[1,1], layer_name = scope+'_conv1_1')
            x1 = self.relu(x1)
            #branch2
            x2 = self.conv_layer(x, int(om/4), kernel=[3,1], layer_name = scope+'_conv2_1')
            x2 = self.relu(x2)
            x2 = self.conv_layer(x2, int(om/4), kernel=[1,3], layer_name = scope+'_conv2_2')
            x2 = self.relu(x2)
            #branch3
            x3 = self.conv_layer(x, int(om/4), kernel=[5,1], layer_name = scope+'_conv3_1')
            x3 = self.relu(x3)
            x3 = self.conv_layer(x3, int(om/4), kernel=[1,5], layer_name = scope+'_conv3_2')
            x3 = self.relu(x3)
            #branch4
            x4 = self.conv_layer(x, int(om/4), kernel=[7,1], layer_name = scope+'_conv4_1')
            x4 = self.relu(x4)
            x4 = self.conv_layer(x4, int(om/4), kernel=[1,7], layer_name = scope+'_conv4_2')
            x4 = self.relu(x4)
            
            x = self.concatenation([x1,x2,x3,x4])
            x = self.conv_layer(x, om, kernel=[1,1], layer_name='conv5')
            return x

    #1014642个变量
    def get_train_model_multitask_v3(self, inputs):

        with tf.variable_scope('base'): 
            #输入：96*36*3
            x = inputs
            #卷积核：3*3*3*64，输出：96*36*64
            x = self.conv_layer(x, 32, kernel=[3,3], layer_name='conv1')
            x = self.batch_normalization(x, training=self._training, scope='bn1')
            x = self.relu(x)
            x1 = x

            x = self.max_pooling(x, pool_size=[3,3], stride=1, padding='SAME')
            
            #输出：96*36*128
            x = self.small_inception_block(x, 64, scope='incep_block1')
            x = self.batch_normalization(x, training=self._training, scope='bn2')
            x = self.relu(x)
            x2 = x

            #输出：48*36*64
            x = self.max_pooling(x, pool_size=[3,3], stride=[2,1], padding='SAME')

            #输出：48*36*256
            x = self.small_inception_block(x, 128, scope='incep_block2')
            x = self.batch_normalization(x, training=self._training, scope='bn3')
            x = self.relu(x)
            x3 = x

            #输出：48*36*256
            x = self.small_inception_block(x, 256, scope='incep_block3')
            x = self.batch_normalization(x, training=self._training, scope='bn4')
            x = self.relu(x)
            x4 = x

            #输出：24*36*256
            x = self.max_pooling(x, pool_size=[3,3], stride=[2,1], padding='SAME')
            x = self.drop_out(x, rate=0.3, training=self._training)

            x_classify = x

            #卷积核：4*1*256*256，输出：24*36*256
            x = self.conv_layer(x, 256, kernel=[4,1], layer_name='conv2')
            x = self.drop_out(x, rate=0.3, training=self._training)
            x = self.batch_normalization(x, training=self._training, scope='bn5')
            x = self.relu(x)

            #卷积核：1*13*256*67，输出：24*36*67
            x = self.conv_layer(x, NUM_CHARS+1, kernel=[1,13], layer_name='conv3')
            x = self.drop_out(x, rate=0.3, training=self._training)
            x = self.batch_normalization(x, training=self._training, scope='bn6')
            x = self.relu(x)

            #池化：输入：96*36*3，输出：x0 = 24*36*16
            x0 = self.average_pooling(inputs, pool_size=[4,1], stride=[4,1], padding='SAME')
            x0 = self.conv_layer(x0, 32, kernel=[1,1], layer_name='conv_x0')
            x0 = self.batch_normalization(x0, training=self._training, scope='bn_x0')
            x0 = self.relu(x0)

            #池化：输入：96*36*32，输出：x1 = 24*36*32
            x1 = self.average_pooling(x1, pool_size=[4,1], stride=[4,1], padding='SAME')
            #x1 = self.conv_layer(x1, 32, kernel=[1,1], layer_name='conv_x1')
            #x1 = self.batch_normalization(x1, training=self._training, scope='bn_x1')
            #x1 = self.relu(x1)

            #池化：输入：96*36*64，输出：x2 = 24*36*64
            x2 = self.average_pooling(x2, pool_size=[4,1], stride=[4,1], padding='SAME')
            #x2 = self.conv_layer(x2, 32, kernel=[1,1], layer_name='conv_x2')
            #x2 = self.batch_normalization(x2, training=self._training, scope='bn_x2')
            #x2 = self.relu(x2)

            #池化：输入：48*36*256，输出：x3 = 24*36*128
            x3 = self.average_pooling(x3, pool_size=[2,1], stride=[2,1], padding='SAME')
            #x3 = self.conv_layer(x3, 64, kernel=[1,1], layer_name='conv_x3')
            #x3 = self.batch_normalization(x3, training=self._training, scope='bn_x3')
            #x3 = self.relu(x3)

            #池化：输入：48*36*256，输出：x3 = 24*36*256
            x4 = self.average_pooling(x4, pool_size=[2,1], stride=[2,1], padding='SAME')
            #x4 = self.conv_layer(x4, 128, kernel=[1,1], layer_name='conv_x4')
            #x4 = self.batch_normalization(x4, training=self._training, scope='bn_x4')
            #x4 = self.relu(x4)

            #通道合并：输入24*36*（67+16+16+32+64+128），输出：24*36*512
            x = self.concatenation([x, x0, x1, x2, x3, x4])
            x_twoline = x

            #卷积核：1*1*454*67，输出：24*36*67
            x = self.conv_layer(x, NUM_CHARS+1, kernel=[1,1], layer_name='conv4')
            #降维：输入：b*24*36*67，输出：b*24*67
            logits = tf.reduce_mean(x, axis=2)

        with tf.variable_scope('classify'): 
            #输出：24*36*64
            x_classify = self.conv_layer(x_classify, 32, kernel=[1,1], layer_name='conv1')
            #输出：12*12*32
            x_classify = self.max_pooling(x_classify, pool_size=[3,3], stride=[2,3], padding='SAME')
            #输出：10*10*32
            x_classify = self.conv_layer(x_classify, 32, kernel=[3,3], padding='VALID', layer_name='conv2')
            #输出：5*5*32
            x_classify = self.max_pooling(x_classify, pool_size=[3,3], stride=[2,2], padding='SAME')
            #输出：800
            cl_shape = x_classify.get_shape().as_list()
            #nodes = cl_shape[1]*cl_shape[2]*cl_shape[3]
            x_classify = tf.reshape(x_classify, [-1, cl_shape[1]*cl_shape[2]*cl_shape[3]])
            dense = self.fully_connected(x_classify, units=128, activation=tf.nn.relu, layer_name='fully_connected1')
            dense = self.fully_connected(dense, units=32, activation=tf.nn.relu, layer_name='fully_connected2')
            logits_classify = self.fully_connected(dense, units=1, activation=tf.nn.sigmoid, layer_name='fully_connected3')

        with tf.variable_scope('two_line'): 
            x_shape = x_twoline.get_shape().as_list()
            x_up = tf.slice(x_twoline, [0, 0, 0, 0], [x_shape[0], x_shape[1], int(x_shape[2]/3+2), x_shape[3]])
            x_down = tf.slice(x_twoline, [0, 0, int(x_shape[2]/3), 0], [x_shape[0], x_shape[1], int(x_shape[2]/3*2), x_shape[3]])

            x_up = self.conv_layer(x_up, NUM_CHARS+1, kernel=[1,1], layer_name='conv_up')
            x_down = self.conv_layer(x_down, NUM_CHARS+1, kernel=[1,1], layer_name='conv_down')

            logits_up = tf.reduce_mean(x_up, axis=2)
            logits_down = tf.reduce_mean(x_down, axis=2)

        #返回值：logits:(b*24*67), inputs(b*96*36*3), targets(1), seq_len(n)
        return logits, logits_up, logits_down, logits_classify
