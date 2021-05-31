import tensorflow as tf
from recognition.lpr_util import NUM_CHARS

def small_inception_block(x, im, om, scope='incep_block'):
    #参考squeezenet的fire module，先squeeze，然后expand
    with tf.variable_scope(scope): 
        x = conv(x,im,int(om/4),ksize=[1,1])
        x = tf.nn.relu(x)
        #参考inception v3
        #branch1
        x1 = conv(x, int(om/4), int(om/4), ksize=[1,1], layer_name='conv1')
        x1 = tf.nn.relu(x1)
        #branch2
        x2 = conv(x, int(om/4), int(om/4), ksize=[3,1], pad='SAME', layer_name='conv2_1')
        x2 = tf.nn.relu(x2)
        x2 = conv(x2, int(om/4), int(om/4), ksize=[1,3], pad='SAME', layer_name='conv2_2')
        x2 = tf.nn.relu(x2)
        #branch3
        x3 = conv(x, int(om/4), int(om/4), ksize=[5,1], pad='SAME', layer_name='conv3_1')
        x3 = tf.nn.relu(x3)  
        x3 = conv(x3, int(om/4), int(om/4), ksize=[1,5], pad='SAME', layer_name='conv3_2')
        x3 = tf.nn.relu(x3)
        #branch4
        x4 = conv(x, int(om/4), int(om/4), ksize=[7,1], pad='SAME', layer_name='conv4_1')
        x4 = tf.nn.relu(x4)  
        x4 = conv(x4, int(om/4), int(om/4), ksize=[1,7], pad='SAME', layer_name='conv4_2')
        x4 = tf.nn.relu(x4)
        #x4 = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        
        x = tf.concat([x1,x2,x3,x4], 3)
        x = conv(x, om, om, ksize=[1,1], layer_name='conv5')
        return x

def small_basic_block(x, im, om, scope='bas_block'):
    '''提取小模块'''
    with tf.variable_scope(scope): 
        x = conv(x,im,int(om/4),ksize=[1,1])
        x = tf.nn.relu(x)
        x1 = conv(x, int(om/4), int(om/4), ksize=[3,1], pad='SAME')
        x2 = conv(x, int(om/4), int(om/4), ksize=[5,1], pad='SAME')
        x3 = conv(x, int(om/4), int(om/4), ksize=[7,1], pad='SAME')
        x1 = tf.nn.relu(x1)
        x2 = tf.nn.relu(x2)
        x3 = tf.nn.relu(x3)
        x1 = conv(x1, int(om/4), int(om/4), ksize=[1,3], pad='SAME')
        x2 = conv(x2, int(om/4), int(om/4), ksize=[1,5], pad='SAME')
        x3 = conv(x3, int(om/4), int(om/4), ksize=[1,7], pad='SAME')
        x1 = tf.nn.relu(x1)
        x2 = tf.nn.relu(x2)
        x3 = tf.nn.relu(x3)
        x = tf.concat([x1,x2,x3], 3)
        x = conv(x,int(om/4*3),om,ksize=[1,1])
        return x

#深度可分离网络
def depth_sep_conv(x, im, om, ksize, stride=[1,1,1,1], pad='SAME', training=False, scope = 'sep_conv'):
    with tf.variable_scope(scope): 
        conv_weights_d = tf.Variable(tf.truncated_normal([ksize[0], ksize[1], im, 1], stddev=0.1,seed=None, dtype=tf.float32))
        conv_depthwise = tf.nn.depthwise_conv2d(x, conv_weights_d, strides=stride, padding=pad)
        #print('conv_depthwise shape is :', conv_depthwise.get_shape().as_list())
        conv_depthwise = tf.layers.batch_normalization(conv_depthwise, training)
        conv_depthwise = tf.nn.relu(conv_depthwise)
        
        conv_weights_s = tf.Variable(tf.truncated_normal([1, 1, im, om], stddev=0.1,seed=None, dtype=tf.float32))
        conv_biases = tf.Variable(tf.zeros([om], dtype=tf.float32))
        conv_sep = tf.nn.conv2d(conv_depthwise, conv_weights_s, strides=stride, padding=pad)
        out = tf.nn.bias_add(conv_sep, conv_biases)

        out = tf.layers.batch_normalization(out, training)
        out = tf.nn.relu(out)
        #print("out:", out.get_shape().as_list())
        return out

def conv(x,im,om,ksize,stride=[1,1,1,1], pad='SAME', layer_name='conv'):
    with tf.variable_scope(layer_name): 
        conv_weights = tf.Variable(
            tf.truncated_normal([ksize[0], ksize[1], im, om], stddev=0.1,
                                seed=None, dtype=tf.float32), name='weight')
        conv_biases = tf.Variable(tf.zeros([om], dtype=tf.float32), name='biase')
        out = tf.nn.conv2d(x, conv_weights, strides=stride, padding=pad)
        relu = tf.nn.bias_add(out, conv_biases)
        return relu

#893128个变量
def get_train_model(num_channels, batch_size, img_size, training=False):

    inputs = tf.placeholder(tf.float32, shape=(batch_size, img_size[0], img_size[1], num_channels))

    # 定义ctc_loss需要的稀疏矩阵
    targets = tf.sparse_placeholder(tf.int32)

    # 1维向量 序列长度 [batch_size,]
    seq_len = tf.placeholder(tf.int32, [None])
    #输入：94*24*3
    x = inputs

    #卷积核：3*3*3*64，输出：94*24*64
    x = conv(x,num_channels,64,ksize=[3,3])
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x,
                          ksize=[1, 3, 3, 1],
                          strides=[1, 1, 1, 1],
                          padding='SAME')

    #输出：94*24*64
    x = small_basic_block(x,64,64)
    x2=x
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)

    #输出：47*24*64
    x = tf.nn.max_pool(x,
                          ksize=[1, 3, 3, 1],
                          strides=[1, 2, 1, 1],
                          padding='SAME')

    #输出：47*24*256
    x = small_basic_block(x, 64,256)
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)

    #输出：47*24*256
    x = small_basic_block(x, 256, 256)
    x3 = x
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)

    #输出：24*24*256
    x = tf.nn.max_pool(x,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 1, 1],
                       padding='SAME')
    x = tf.layers.dropout(x)

    #卷积核：4*1*256*256，输出：24*24*256
    x = conv(x, 256, 256, ksize=[4, 1])
    #函数默认的drop rate=0.5
    x = tf.layers.dropout(x)
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)

    #卷积核：1*13*256*67，输出：24*24*67
    x = conv(x,256,NUM_CHARS+1,ksize=[1,13],pad='SAME')
    x = tf.nn.relu(x)
    cx = tf.reduce_mean(tf.square(x))
    x = tf.div(x,cx)

    #池化：输入：94*24*3，输出：x1 = 24*24*3
    x1 = tf.nn.avg_pool(inputs,
                       ksize=[1, 4, 1, 1],
                       strides=[1, 4, 1, 1],
                       padding='SAME')
    cx1 = tf.reduce_mean(tf.square(x1))
    x1 = tf.div(x1, cx1)

    #池化：输入：94*24*64，输出：x1 = 24*24*64
    x2 = tf.nn.avg_pool(x2,
                        ksize=[1, 4, 1, 1],
                        strides=[1, 4, 1, 1],
                        padding='SAME')
    cx2 = tf.reduce_mean(tf.square(x2))
    x2 = tf.div(x2, cx2)

    #池化：输入：47*24*256，输出：x1 = 24*24*256
    x3 = tf.nn.avg_pool(x3,
                        ksize=[1, 2, 1, 1],
                        strides=[1, 2, 1, 1],
                        padding='SAME')
    cx3 = tf.reduce_mean(tf.square(x3))
    x3 = tf.div(x3, cx3)

     #通道合并：输入24*24*（67+3+64+256），输出：24*24*390
    x = tf.concat([x,x1,x2,x3],3)

    #卷积核：1*1*390*67，输出：24*24*67
    x = conv(x, x.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1))

    #降维：输入：b*24*24*67，输出：b*24*67
    logits = tf.reduce_mean(x, axis=2)

    #返回值：logits:(b*24*67), inputs(b*94*24*3), targets(1), seq_len(n)
    return logits, inputs, targets, seq_len

#981336个变量
def get_train_model_new(num_channels, batch_size, img_size, training=False):

    inputs = tf.placeholder(tf.float32, shape=(batch_size, img_size[0], img_size[1], num_channels))

    # 定义ctc_loss需要的稀疏矩阵
    targets = tf.sparse_placeholder(tf.int32)

    # 1维向量 序列长度 [batch_size,]
    seq_len = tf.placeholder(tf.int32, [None])
    #输入：94*24*3
    x = inputs

    #卷积核：3*3*3*64，输出：94*24*64
    x = conv(x,num_channels,64,ksize=[3,3])
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x,
                          ksize=[1, 3, 3, 1],
                          strides=[1, 1, 1, 1],
                          padding='SAME')

    #输出：94*24*128
    x = small_inception_block(x, 64, 128)
    x2 = x

    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)

    #输出：47*24*64
    x = tf.nn.max_pool(x,
                          ksize=[1, 3, 3, 1],
                          strides=[1, 2, 1, 1],
                          padding='SAME')

    #输出：47*24*256
    x = small_inception_block(x, 128, 256)
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)

    #输出：47*24*256
    x = small_inception_block(x, 256, 256)
    x3 = x
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)

    #输出：24*24*256
    x = tf.nn.max_pool(x,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 1, 1],
                       padding='SAME')
    x = tf.layers.dropout(x)

    #卷积核：4*1*256*256，输出：24*24*256
    x = conv(x, 256, 256, ksize=[4, 1])
    #函数默认的drop rate=0.5
    x = tf.layers.dropout(x)
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)

    #卷积核：1*13*256*67，输出：24*24*67
    x = conv(x,256,NUM_CHARS+1,ksize=[1,13],pad='SAME')
    x = tf.nn.relu(x)
    cx = tf.reduce_mean(tf.square(x))
    x = tf.div(x,cx)

    #池化：输入：94*24*3，输出：x1 = 24*24*3
    x1 = tf.nn.avg_pool(inputs,
                       ksize=[1, 4, 1, 1],
                       strides=[1, 4, 1, 1],
                       padding='SAME')
    cx1 = tf.reduce_mean(tf.square(x1))
    x1 = tf.div(x1, cx1)

    #池化：输入：94*24*128，输出：x1 = 24*24*128
    x2 = tf.nn.avg_pool(x2,
                        ksize=[1, 4, 1, 1],
                        strides=[1, 4, 1, 1],
                        padding='SAME')
    cx2 = tf.reduce_mean(tf.square(x2))
    x2 = tf.div(x2, cx2)

    #池化：输入：47*24*256，输出：x1 = 24*24*256
    x3 = tf.nn.avg_pool(x3,
                        ksize=[1, 2, 1, 1],
                        strides=[1, 2, 1, 1],
                        padding='SAME')
    cx3 = tf.reduce_mean(tf.square(x3))
    x3 = tf.div(x3, cx3)

    #通道合并：输入24*24*（67+3+128+256），输出：24*24*454
    x = tf.concat([x,x1,x2,x3],3)

    #卷积核：1*1*454*67，输出：24*24*67
    x = conv(x, x.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1))

    #降维：输入：b*24*24*67，输出：b*24*67
    logits = tf.reduce_mean(x, axis=2)

    #返回值：logits:(b*24*67), inputs(b*94*24*3), targets(1), seq_len(n)
    return logits, inputs, targets, seq_len

#1107417个变量
def get_train_model_multitask(inputs, num_channels, batch_size, img_size, training=False):

    #输入：96*36*3
    x = inputs

    #卷积核：3*3*3*64，输出：96*36*64
    x = conv(x,num_channels,64,ksize=[3,3])
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

    #输出：96*36*128
    x = small_inception_block(x, 64, 128)
    x2 = x

    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)

    #输出：48*36*64
    x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 1, 1], padding='SAME')

    #输出：48*36*256
    x = small_inception_block(x, 128, 256)
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)

    #输出：48*36*256
    x = small_inception_block(x, 256, 256)
    x3 = x
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)

    #输出：24*36*256
    x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 1, 1], padding='SAME')
    x = tf.layers.dropout(x)

    x_classify = x
    #输出：24*36*64
    x_classify = conv(x_classify, 256, 32, ksize=[1, 1])
    #输出：12*12*32
    x_classify = tf.nn.max_pool(x_classify, ksize=[1, 3, 3, 1], strides = [1, 2, 3, 1], padding='SAME')
    #输出：10*10*32
    x_classify = conv(x_classify, 32, 32, ksize=[3, 3], stride=[1,1,1,1], pad='VALID')
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
    x = conv(x, 256, 256, ksize=[4, 1])
    #函数默认的drop rate=0.5
    x = tf.layers.dropout(x)
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)

    #卷积核：1*13*256*67，输出：24*36*67
    x = conv(x,256,NUM_CHARS+1,ksize=[1,13],pad='SAME')
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
    x = conv(x, x.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1))

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
def get_train_model_multitask_v2(inputs, num_channels, batch_size, img_size, training=False):

    #输入：96*36*3
    x = inputs
    #卷积核：3*3*3*64，输出：96*36*64
    x = conv(x,num_channels,64,ksize=[3,3], layer_name='conv1')
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

    #输出：96*36*128
    x = small_inception_block(x, 64, 128, scope='incep_block1')
    x2 = x

    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)

    #输出：48*36*64
    x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 1, 1], padding='SAME')

    #输出：48*36*256
    x = small_inception_block(x, 128, 256, scope='incep_block2')
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)

    #输出：48*36*256
    x = small_inception_block(x, 256, 256, scope='incep_block3')
    x3 = x
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)

    #输出：24*36*256
    x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 1, 1], padding='SAME')
    x = tf.layers.dropout(inputs=x, rate=0.3, training=training)

    with tf.variable_scope('classify'): 
        x_classify = x
        #输出：24*36*64
        x_classify = conv(x_classify, 256, 32, ksize=[1, 1], layer_name='conv1')
        #输出：12*12*32
        x_classify = tf.nn.max_pool(x_classify, ksize=[1, 3, 3, 1], strides = [1, 2, 3, 1], padding='SAME')
        #输出：10*10*32
        x_classify = conv(x_classify, 32, 32, ksize=[3, 3], stride=[1,1,1,1], pad='VALID', layer_name='conv2')
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
    x = conv(x, 256, 256, ksize=[4, 1], layer_name='conv2')
    #函数默认的drop rate=0.2
    x = tf.layers.dropout(inputs=x, rate=0.3, training=training)
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)

    #卷积核：1*13*256*67，输出：24*36*67
    x = conv(x,256,NUM_CHARS+1,ksize=[1,13],pad='SAME', layer_name='conv3')
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
        x_up = conv(x_up, x_up.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1), layer_name='conv1')
        x_down = conv(x_down, x_down.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1), layer_name='conv2')

    #卷积核：1*1*454*67，输出：24*36*67
    x = conv(x, x.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1), layer_name='conv4')
    #降维：输入：b*24*36*67，输出：b*24*67
    logits = tf.reduce_mean(x, axis=2)
    logits_up = tf.reduce_mean(x_up, axis=2)
    logits_down = tf.reduce_mean(x_down, axis=2)

    #返回值：logits:(b*24*67), inputs(b*96*36*3), targets(1), seq_len(n)
    return logits, logits_up, logits_down, logits_classify

#1183364个变量
def get_train_model_multitask_v3(inputs, num_channels, batch_size, img_size, training):

    with tf.variable_scope('base'): 
        #输入：96*36*3
        x = inputs
        #卷积核：3*3*3*64，输出：96*36*64
        x = conv(x, num_channels, 64, ksize=[3,3], layer_name='conv1')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        x1 = x
        #输出：96*36*128
        x = small_inception_block(x, 64, 128, scope='incep_block1')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x2 = x

        #输出：48*36*64
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 1, 1], padding='SAME')

        #输出：48*36*256
        x = small_inception_block(x, 128, 256, scope='incep_block2')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)

        #输出：48*36*256
        x = small_inception_block(x, 256, 256, scope='incep_block3')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x3 = x

        #输出：24*36*256
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 1, 1], padding='SAME')
        x = tf.layers.dropout(inputs=x, rate=0.3, training=training)

        x_classify = x

        #卷积核：4*1*256*256，输出：24*36*256
        x = conv(x, 256, 256, ksize=[4, 1], layer_name='conv2')

        x = tf.layers.dropout(inputs=x, rate=0.3, training=training)
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)

        #卷积核：1*13*256*67，输出：24*36*67
        x = conv(x, 256, NUM_CHARS+1, ksize=[1,13], pad='SAME', layer_name='conv3')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)

        #池化：输入：96*36*3，输出：x0 = 24*36*16
        x0 = tf.nn.avg_pool(inputs, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')
        x0 = conv(x0, x0.get_shape().as_list()[3], 16, ksize=(1, 1), layer_name='conv_x0')
        x0 = tf.layers.batch_normalization(x0, training=training)
        x0 = tf.nn.relu(x0)

        #池化：输入：96*36*64，输出：x1 = 24*36*32
        x1 = tf.nn.avg_pool(x1, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')
        x1 = conv(x1, x1.get_shape().as_list()[3], 32, ksize=(1, 1), layer_name='conv_x1')
        x1 = tf.layers.batch_normalization(x1, training=training)
        x1 = tf.nn.relu(x1)

        #池化：输入：96*36*128，输出：x2 = 24*36*64
        x2 = tf.nn.avg_pool(x2, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')
        x2 = conv(x2, x2.get_shape().as_list()[3], 64, ksize=(1, 1), layer_name='conv_x2')
        x2 = tf.layers.batch_normalization(x2, training=training)
        x2 = tf.nn.relu(x2)

        #池化：输入：48*36*256，输出：x3 = 24*36*128
        x3 = tf.nn.avg_pool(x3, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')
        x3 = conv(x3, x3.get_shape().as_list()[3], 128, ksize=(1, 1), layer_name='conv_x3')
        x3 = tf.layers.batch_normalization(x3, training=training)
        x3 = tf.nn.relu(x3)

        #通道合并：输入24*36*（67+16+32+64+128），输出：24*36*307
        x = tf.concat([x, x0, x1, x2, x3], 3)
        x_twoline = x

        #卷积核：1*1*454*67，输出：24*36*67
        x = conv(x, x.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1), layer_name='conv4')
        #降维：输入：b*24*36*67，输出：b*24*67
        logits = tf.reduce_mean(x, axis=2)

    with tf.variable_scope('classify'): 
        #输出：24*36*64
        x_classify = conv(x_classify, 256, 32, ksize=[1, 1], layer_name='conv1')
        #输出：12*12*32
        x_classify = tf.nn.max_pool(x_classify, ksize=[1, 3, 3, 1], strides = [1, 2, 3, 1], padding='SAME')
        #输出：10*10*32
        x_classify = conv(x_classify, 32, 32, ksize=[3, 3], stride=[1,1,1,1], pad='VALID', layer_name='conv2')
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

        x_up = conv(x_up, x_up.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1), layer_name='conv1')
        x_down = conv(x_down, x_down.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1), layer_name='conv2')

        logits_up = tf.reduce_mean(x_up, axis=2)
        logits_down = tf.reduce_mean(x_down, axis=2)

    #返回值：logits:(b*24*67), inputs(b*96*36*3), targets(1), seq_len(n)
    return logits, logits_up, logits_down, logits_classify
