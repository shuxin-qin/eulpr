import tensorflow as tf
from recognition.lpr_util import NUM_CHARS

class LPRNet(object):

    def __init__(self, training=False):
        self._training = training

    def build_net(self, inputs, version='v2'):
        if version == 'v1':
            return self.get_train_model_multitask(inputs)
        elif version == 'v2':
            return self.get_train_model_multitask_v2(inputs)
        elif version == 'v3':
            return self.get_train_model_multitask_v3(inputs)

    def small_inception_block(self, x, im, om, scope='incep_block'):
        #参考squeezenet的fire module，先squeeze，然后expand
        with tf.variable_scope(scope): 
            x = self.conv(x,im,int(om/4),ksize=[1,1])
            x = tf.nn.relu(x)
            #参考inception v3
            #branch1
            x1 = self.conv(x, int(om/4), int(om/4), ksize=[1,1], layer_name='conv1')
            x1 = tf.nn.relu(x1)
            #branch2
            x2 = self.conv(x, int(om/4), int(om/4), ksize=[3,1], pad='SAME', layer_name='conv2_1')
            x2 = tf.nn.relu(x2)
            x2 = self.conv(x2, int(om/4), int(om/4), ksize=[1,3], pad='SAME', layer_name='conv2_2')
            x2 = tf.nn.relu(x2)
            #branch3
            x3 = self.conv(x, int(om/4), int(om/4), ksize=[5,1], pad='SAME', layer_name='conv3_1')
            x3 = tf.nn.relu(x3)  
            x3 = self.conv(x3, int(om/4), int(om/4), ksize=[1,5], pad='SAME', layer_name='conv3_2')
            x3 = tf.nn.relu(x3)
            #branch4
            x4 = self.conv(x, int(om/4), int(om/4), ksize=[7,1], pad='SAME', layer_name='conv4_1')
            x4 = tf.nn.relu(x4)  
            x4 = self.conv(x4, int(om/4), int(om/4), ksize=[1,7], pad='SAME', layer_name='conv4_2')
            x4 = tf.nn.relu(x4)
            #x4 = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
            
            x = tf.concat([x1,x2,x3,x4], 3)
            x = self.conv(x, om, om, ksize=[1,1], layer_name='conv5')
            return x

    def conv(self, x, im, om, ksize, stride=[1,1,1,1], pad='SAME', layer_name='conv'):
        with tf.variable_scope(layer_name): 
            conv_weights = tf.Variable(
                tf.truncated_normal([ksize[0], ksize[1], im, om], stddev=0.1,
                                    seed=None, dtype=tf.float32), name='weight')
            conv_biases = tf.Variable(tf.zeros([om], dtype=tf.float32), name='biase')
            out = tf.nn.conv2d(x, conv_weights, strides=stride, padding=pad)
            relu = tf.nn.bias_add(out, conv_biases)
            return relu

    #1107417个变量
    def get_train_model_multitask(self, inputs):
        #输入：96*36*3
        x = inputs
        #卷积核：3*3*3*64，输出：96*36*64
        x = self.conv(x, 3, 64, ksize=[3,3])
        x = tf.layers.batch_normalization(x, training=self._training)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        #输出：96*36*128
        x = self.small_inception_block(x, 64, 128)
        x2 = x

        x = tf.layers.batch_normalization(x, training=self._training)
        x = tf.nn.relu(x)

        #输出：48*36*64
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 1, 1], padding='SAME')

        #输出：48*36*256
        x = self.small_inception_block(x, 128, 256)
        x = tf.layers.batch_normalization(x, training=self._training)
        x = tf.nn.relu(x)

        #输出：48*36*256
        x = self.small_inception_block(x, 256, 256)
        x3 = x
        x = tf.layers.batch_normalization(x, training=self._training)
        x = tf.nn.relu(x)

        #输出：24*36*256
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 1, 1], padding='SAME')
        x = tf.layers.dropout(x)

        x_classify = x
        #输出：24*36*64
        x_classify = self.conv(x_classify, 256, 32, ksize=[1, 1])
        #输出：12*12*32
        x_classify = tf.nn.max_pool(x_classify, ksize=[1, 3, 3, 1], strides = [1, 2, 3, 1], padding='SAME')
        #输出：10*10*32
        x_classify = self.conv(x_classify, 32, 32, ksize=[3, 3], stride=[1,1,1,1], pad='VALID')
        #输出：5*5*32
        x_classify = tf.nn.max_pool(x_classify, ksize=[1, 3, 3, 1], strides = [1, 2, 2, 1], padding='SAME')
        #输出：800
        cl_shape = x_classify.get_shape().as_list()
        #nodes = cl_shape[1]*cl_shape[2]*cl_shape[3]
        x_classify = tf.reshape(x_classify, [-1, cl_shape[1]*cl_shape[2]*cl_shape[3]])
        dense = tf.layers.dense(inputs=x_classify, units=128, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        dense = tf.layers.dense(inputs=dense, units=32, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        logits_classify = tf.layers.dense(inputs=dense, units=1, activation=tf.nn.sigmoid)

        #卷积核：4*1*256*256，输出：24*36*256
        x = self.conv(x, 256, 256, ksize=[4, 1])
        #函数默认的drop rate=0.5
        x = tf.layers.dropout(x)
        x = tf.layers.batch_normalization(x, training=self._training)
        x = tf.nn.relu(x)

        #卷积核：1*13*256*67，输出：24*36*67
        x = self.conv(x,256,NUM_CHARS+1,ksize=[1,13],pad='SAME')
        x = tf.nn.relu(x)
        cx = tf.reduce_mean(tf.square(x))
        x = tf.div(x,cx)

        #池化：输入：96*36*3，输出：x1 = 24*36*3
        x1 = tf.nn.avg_pool(inputs, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')
        cx1 = tf.reduce_mean(tf.square(x1))
        x1 = tf.div(x1, cx1)

        #池化：输入：96*36*128，输出：x2 = 24*36*128
        x2 = tf.nn.avg_pool(x2, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')
        cx2 = tf.reduce_mean(tf.square(x2))
        x2 = tf.div(x2, cx2)

        #池化：输入：48*36*256，输出：x3 = 24*36*256
        x3 = tf.nn.avg_pool(x3, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')
        cx3 = tf.reduce_mean(tf.square(x3))
        x3 = tf.div(x3, cx3)

        #通道合并：输入24*36*（67+3+128+256），输出：24*36*454
        x = tf.concat([x,x1,x2,x3],3)

        #卷积核：1*1*454*67，输出：24*36*67
        x = self.conv(x, x.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1))

        x_shape = x.get_shape().as_list()
        x_up = tf.slice(x, [0, 0, 0, 0], [x_shape[0], x_shape[1], int(x_shape[2]/3), x_shape[3]])
        x_down = tf.slice(x, [0, 0, int(x_shape[2]/3), 0], [x_shape[0], x_shape[1], int(x_shape[2]/3*2), x_shape[3]])

        #降维：输入：b*24*36*67，输出：b*24*67
        logits = tf.reduce_mean(x, axis=2)
        logits_up = tf.reduce_mean(x_up, axis=2)
        logits_down = tf.reduce_mean(x_down, axis=2)

        #返回值：logits:(b*24*67), inputs(b*96*36*3), targets(1), seq_len(n)
        return logits, logits_up, logits_down, logits_classify

    #1168387个变量
    def get_train_model_multitask_v2(self, inputs):

        #输入：96*36*3
        x = inputs
        #卷积核：3*3*3*64，输出：96*36*64
        x = self.conv(x, 3, 64, ksize=[3,3], layer_name='conv1')
        x = tf.layers.batch_normalization(x, training=self._training)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        #输出：96*36*128
        x = self.small_inception_block(x, 64, 128, scope='incep_block1')
        x2 = x

        x = tf.layers.batch_normalization(x, training=self._training)
        x = tf.nn.relu(x)

        #输出：48*36*64
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 1, 1], padding='SAME')

        #输出：48*36*256
        x = self.small_inception_block(x, 128, 256, scope='incep_block2')
        x = tf.layers.batch_normalization(x, training=self._training)
        x = tf.nn.relu(x)

        #输出：48*36*256
        x = self.small_inception_block(x, 256, 256, scope='incep_block3')
        x3 = x
        x = tf.layers.batch_normalization(x, training=self._training)
        x = tf.nn.relu(x)

        #输出：24*36*256
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 1, 1], padding='SAME')
        x = tf.layers.dropout(inputs=x, rate=0.3, training=self._training)

        with tf.variable_scope('classify'): 
            x_classify = x
            #输出：24*36*64
            x_classify = self.conv(x_classify, 256, 32, ksize=[1, 1], layer_name='conv1')
            #输出：12*12*32
            x_classify = tf.nn.max_pool(x_classify, ksize=[1, 3, 3, 1], strides = [1, 2, 3, 1], padding='SAME')
            #输出：10*10*32
            x_classify = self.conv(x_classify, 32, 32, ksize=[3, 3], stride=[1,1,1,1], pad='VALID', layer_name='conv2')
            #输出：5*5*32
            x_classify = tf.nn.max_pool(x_classify, ksize=[1, 3, 3, 1], strides = [1, 2, 2, 1], padding='SAME')
            #输出：800
            cl_shape = x_classify.get_shape().as_list()
            #nodes = cl_shape[1]*cl_shape[2]*cl_shape[3]
            x_classify = tf.reshape(x_classify, [-1, cl_shape[1]*cl_shape[2]*cl_shape[3]])
            dense = tf.layers.dense(inputs=x_classify, units=128, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
            dense = tf.layers.dense(inputs=dense, units=32, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
            logits_classify = tf.layers.dense(inputs=dense, units=1, activation=tf.nn.sigmoid)

        #卷积核：4*1*256*256，输出：24*36*256
        x = self.conv(x, 256, 256, ksize=[4, 1], layer_name='conv2')
        #函数默认的drop rate=0.2
        x = tf.layers.dropout(inputs=x, rate=0.3, training=self._training)
        x = tf.layers.batch_normalization(x, training=self._training)
        x = tf.nn.relu(x)

        #卷积核：1*13*256*67，输出：24*36*67
        x = self.conv(x,256,NUM_CHARS+1,ksize=[1,13],pad='SAME', layer_name='conv3')
        x = tf.nn.relu(x)
        cx = tf.reduce_mean(tf.square(x))
        x = tf.div(x,cx)

        #池化：输入：96*36*3，输出：x1 = 24*36*3
        x1 = tf.nn.avg_pool(inputs, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')
        cx1 = tf.reduce_mean(tf.square(x1))
        x1 = tf.div(x1, cx1)

        #池化：输入：96*36*128，输出：x2 = 24*36*128
        x2 = tf.nn.avg_pool(x2, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')
        cx2 = tf.reduce_mean(tf.square(x2))
        x2 = tf.div(x2, cx2)

        #池化：输入：48*36*256，输出：x3 = 24*36*256
        x3 = tf.nn.avg_pool(x3, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')
        cx3 = tf.reduce_mean(tf.square(x3))
        x3 = tf.div(x3, cx3)

        #通道合并：输入24*36*（67+3+128+256），输出：24*36*454
        x = tf.concat([x,x1,x2,x3],3)

        with tf.variable_scope('two_line'): 
            x_shape = x.get_shape().as_list()
            x_up = tf.slice(x, [0, 0, 0, 0], [x_shape[0], x_shape[1], int(x_shape[2]/3), x_shape[3]])
            x_down = tf.slice(x, [0, 0, int(x_shape[2]/3), 0], [x_shape[0], x_shape[1], int(x_shape[2]/3*2), x_shape[3]])
            x_up = self.conv(x_up, x_up.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1), layer_name='conv1')
            x_down = self.conv(x_down, x_down.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1), layer_name='conv2')

        #卷积核：1*1*454*67，输出：24*36*67
        x = self.conv(x, x.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1), layer_name='conv4')
        #降维：输入：b*24*36*67，输出：b*24*67
        logits = tf.reduce_mean(x, axis=2)
        logits_up = tf.reduce_mean(x_up, axis=2)
        logits_down = tf.reduce_mean(x_down, axis=2)

        #返回值：logits:(b*24*67), inputs(b*96*36*3), targets(1), seq_len(n)
        return logits, logits_up, logits_down, logits_classify

    #1180648个变量
    def get_train_model_multitask_v3(self, inputs):

        with tf.variable_scope('base'): 
            #输入：96*36*3
            x = inputs
            #卷积核：3*3*3*64，输出：96*36*64
            x = self.conv(x, 3, 64, ksize=[3,3], layer_name='conv1')
            x = tf.layers.batch_normalization(x, training=self._training)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
            x1 = x
            #输出：96*36*128
            x = self.small_inception_block(x, 64, 128, scope='incep_block1')
            x = tf.layers.batch_normalization(x, training=self._training)
            x = tf.nn.relu(x)
            x2 = x

            #输出：48*36*64
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 1, 1], padding='SAME')

            #输出：48*36*256
            x = self.small_inception_block(x, 128, 256, scope='incep_block2')
            x = tf.layers.batch_normalization(x, training=self._training)
            x = tf.nn.relu(x)

            #输出：48*36*256
            x = self.small_inception_block(x, 256, 256, scope='incep_block3')
            x = tf.layers.batch_normalization(x, training=self._training)
            x = tf.nn.relu(x)
            x3 = x

            #输出：24*36*256
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 1, 1], padding='SAME')
            x = tf.layers.dropout(inputs=x, rate=0.3, training=self._training)

            x_classify = x

            #卷积核：4*1*256*256，输出：24*36*256
            x = self.conv(x, 256, 256, ksize=[4, 1], layer_name='conv2')

            x = tf.layers.dropout(inputs=x, rate=0.3, training=self._training)
            x = tf.layers.batch_normalization(x, training=self._training)
            x = tf.nn.relu(x)

            #卷积核：1*13*256*67，输出：24*36*67
            x = self.conv(x, 256, NUM_CHARS+1, ksize=[1,13], pad='SAME', layer_name='conv3')
            x = tf.nn.relu(x)

            #池化：输入：96*36*3，输出：x1 = 24*36*3
            x1 = tf.nn.avg_pool(x1, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')

            #池化：输入：96*36*128，输出：x2 = 24*36*128
            x2 = tf.nn.avg_pool(x2, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')

            #池化：输入：48*36*256，输出：x3 = 24*36*256
            x3 = tf.nn.avg_pool(x3, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

            #通道合并：输入24*36*（67+64+128+256），输出：24*36*515
            x = tf.concat([x,x1,x2,x3],3)
            x_twoline = x

            #卷积核：1*1*454*67，输出：24*36*67
            x = self.conv(x, x.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1), layer_name='conv4')
            #降维：输入：b*24*36*67，输出：b*24*67
            logits = tf.reduce_mean(x, axis=2)

        with tf.variable_scope('classify'): 
            #输出：24*36*64
            x_classify = self.conv(x_classify, 256, 32, ksize=[1, 1], layer_name='conv1')
            #输出：12*12*32
            x_classify = tf.nn.max_pool(x_classify, ksize=[1, 3, 3, 1], strides = [1, 2, 3, 1], padding='SAME')
            #输出：10*10*32
            x_classify = self.conv(x_classify, 32, 32, ksize=[3, 3], stride=[1,1,1,1], pad='VALID', layer_name='conv2')
            #输出：5*5*32
            x_classify = tf.nn.max_pool(x_classify, ksize=[1, 3, 3, 1], strides = [1, 2, 2, 1], padding='SAME')
            #输出：800
            cl_shape = x_classify.get_shape().as_list()
            #nodes = cl_shape[1]*cl_shape[2]*cl_shape[3]
            x_classify = tf.reshape(x_classify, [-1, cl_shape[1]*cl_shape[2]*cl_shape[3]])
            dense = tf.layers.dense(inputs=x_classify, units=128, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
            dense = tf.layers.dense(inputs=dense, units=32, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
            logits_classify = tf.layers.dense(inputs=dense, units=1, activation=tf.nn.sigmoid)

        with tf.variable_scope('two_line'): 
            x_shape = x_twoline.get_shape().as_list()
            x_up = tf.slice(x_twoline, [0, 0, 0, 0], [x_shape[0], x_shape[1], int(x_shape[2]/3), x_shape[3]])
            x_down = tf.slice(x_twoline, [0, 0, int(x_shape[2]/3), 0], [x_shape[0], x_shape[1], int(x_shape[2]/3*2), x_shape[3]])

            x_up = self.conv(x_up, x_up.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1), layer_name='conv1')
            x_down = self.conv(x_down, x_down.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1), layer_name='conv2')

            logits_up = tf.reduce_mean(x_up, axis=2)
            logits_down = tf.reduce_mean(x_down, axis=2)

        #返回值：logits:(b*24*67), inputs(b*96*36*3), targets(1), seq_len(n)
        return logits, logits_up, logits_down, logits_classify
