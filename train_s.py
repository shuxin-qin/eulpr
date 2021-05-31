import tensorflow as tf 
import numpy as np 
from recognition.lpr_net_1 import LPRNet
from recognition.text_image_generator import TextImageGenerator
from recognition.lpr_util import decode_sparse_tensor

import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
#训练最大轮次
num_epochs = 100

#初始化学习速率
INITIAL_LEARNING_RATE = 1e-5
DECAY_STEPS = 3000
LEARNING_RATE_DECAY_FACTOR = 0.95  # The learning rate decay factor

#输出字符串结果的步长间隔
REPORT_STEPS = 200
#训练集的数量
BATCH_SIZE = 128
TRAIN_SIZE = 105265
BATCHES = TRAIN_SIZE//BATCH_SIZE

#统计测试集和训练集大小 ls -l | grep "^-" | wc -l。nohup python -u test.py > nohup.out 2>&1 &

train_dir = '/home/public/share/01Datasets/LP_recognition/generate/train_s'
test_dir = '/home/public/share/01Datasets/LP_recognition/generate/test_s'
model_save_dir = '../model_v3/LPR_model_v3.ckpt'
log_dir = '../logs/log_v3'

img_size = [112, 36]
num_channels = 3

def train(checkpoint = None, mode = 0):
#开始一个新的训练
    train_gen = TextImageGenerator(img_dir=train_dir, batch_size=BATCH_SIZE, img_size=img_size, num_channels=num_channels)
    val_gen = TextImageGenerator(img_dir=test_dir, batch_size=BATCH_SIZE, img_size=img_size, num_channels=num_channels)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, DECAY_STEPS, LEARNING_RATE_DECAY_FACTOR, staircase=True)

    #定义输入张量
    inputs = tf.placeholder(tf.float32, shape=(BATCH_SIZE, img_size[0], img_size[1], num_channels), name='inputs')
    # 定义ctc_loss需要的稀疏矩阵
    targets = tf.sparse_placeholder(tf.int32)
    targets_up = tf.sparse_placeholder(tf.int32)
    targets_down = tf.sparse_placeholder(tf.int32)
    targets_cla = tf.placeholder(tf.float32)

    # 1维向量 序列长度 [batch_size,]
    seq_len = tf.constant(np.ones(BATCH_SIZE, dtype=np.int32) * 24)

    #构建网络，获取logit
    logits, logits_up, logits_down, logits_cla = LPRNet(training = True).build_net(inputs)
    logits = tf.transpose(logits, (1, 0, 2))
    logits_up = tf.transpose(logits_up, (1, 0, 2))
    logits_down = tf.transpose(logits_down, (1, 0, 2))

    #计算Loss
    loss_base = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
    loss_up = tf.nn.ctc_loss(labels=targets_up, inputs=logits_up, sequence_length=seq_len)
    loss_down = tf.nn.ctc_loss(labels=targets_down, inputs=logits_down, sequence_length=seq_len)
    loss_cla = - (targets_cla * tf.log(tf.clip_by_value(logits_cla, 1e-10, 1.0)) + (1-targets_cla)*tf.log(tf.clip_by_value((1-logits_cla), 1e-10, 1.0)))

    #shap由nx1转换成n
    loss_cla_r = tf.reshape(loss_cla, [BATCH_SIZE])
    targets_cla_r = tf.reshape(targets_cla, [BATCH_SIZE])
    lamda = 50
    loss_total = (1 - targets_cla_r) * loss_base + targets_cla_r * (loss_up + loss_down) + lamda * loss_cla_r
    cost_total = tf.reduce_mean(loss_total)

    loss_base = (1 - targets_cla_r) * loss_base + lamda * loss_cla_r
    loss_twoline = targets_cla_r * (loss_up + loss_down)
    cost_base = tf.reduce_mean(loss_base)
    cost_twoline = tf.reduce_mean(loss_twoline)

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    print('Trainable Variables: ', np.sum([np.prod(v.get_shape().as_list()) for v in var_list]))

    base_var_list = [v for v in var_list if 'two_line' not in v.name]
    twoline_var_list = [v for v in var_list if 'two_line' in v.name]
    print('One_line Variables: ', np.sum([np.prod(v.get_shape().as_list()) for v in base_var_list]))
    print('Two_line Variables: ', np.sum([np.prod(v.get_shape().as_list()) for v in twoline_var_list]))

    # 通知Tensorflow在训练时要更新均值和方差的分布
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimize_base = optimizer.minimize(loss_base, global_step=global_step, var_list=base_var_list)
        optimize_twoline = optimizer.minimize(loss_twoline, global_step=global_step, var_list=twoline_var_list)
        optimize_total = optimizer.minimize(loss_total, global_step=global_step, var_list=var_list)
        #optimize_global = optimizer.minimize(loss_total, global_step=global_step, var_list=g_list)

    decoded, _ = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    decoded_up, _ = tf.nn.ctc_beam_search_decoder(logits_up, seq_len, merge_repeated=False)
    decoded_down, _ = tf.nn.ctc_beam_search_decoder(logits_down, seq_len, merge_repeated=False)

    #编辑距离
    acc_base = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
    acc_up = tf.reduce_mean(tf.edit_distance(tf.cast(decoded_up[0], tf.int32), targets_up))
    acc_down = tf.reduce_mean(tf.edit_distance(tf.cast(decoded_down[0], tf.int32), targets_down))
    #acc = (1 - targets_cla_r) * acc_base + targets_cla_r * (acc_up + acc_down)

    # 增加tensorboard显示
    tf.summary.scalar('loss_total', cost_total)
    tf.summary.scalar('loss_base', cost_base)
    tf.summary.scalar('loss_twoline', cost_twoline)
    tf.summary.scalar('acc_base', acc_base)
    tf.summary.scalar('acc_up', acc_up)
    tf.summary.scalar('acc_down', acc_down)
    tf.summary.scalar('learning_rate', learning_rate)

    init = tf.global_variables_initializer()

    def do_batch(train_gen, val_gen, session, merged):
        '''
        if mode == 1:
            optimize_run = optimize_base
            cost_run = cost_base
        elif mode == 2:
            optimize_run = optimize_twoline
            cost_run = cost_twoline
        else:
            optimize_run = optimize_total
            cost_run = cost_total
        '''
        train_inputs, train_targets, train_targets_up, train_targets_down, train_targets_cla = train_gen.next_batch()
        feed = {inputs: train_inputs, targets: train_targets, targets_up: train_targets_up, 
                targets_down: train_targets_down, targets_cla: train_targets_cla}
        b_cost, steps, _, lr, summary = session.run([cost_total, global_step, optimize_total, learning_rate, merged], feed)

        if steps > 0 and steps % 50 == 0:
            print('steps = {}, train loss = {}, learning rate = {}'.format(steps, b_cost, lr))

        if steps > 0 and steps % REPORT_STEPS == 0:
            print('save model', steps)
            saver.save(session, model_save_dir, global_step=steps)

        return b_cost, steps, lr, summary

    with tf.Session() as session:
        session.run(init)

        saver = tf.train.Saver(var_list=var_list, max_to_keep=10)
        if checkpoint != None:
            saver.restore(session, checkpoint)
            print("load model: ", checkpoint)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_dir, session.graph)

        for curr_epoch in range(num_epochs):
            print("Epoch...............", curr_epoch+1)
            start = time.time()
            train_cost = 0
            for batch in range(BATCHES):
                c, steps, lr, summary = do_batch(train_gen, val_gen, session, merged)
                train_cost += c

                if steps > 0 and steps % 50 == 0:
                    writer.add_summary(summary, steps)

            train_cost /= BATCHES

            log = "Epoch {}/{}, train_cost = {:.6f}, learning_rate = {:.6f}, time = {:.3f}s"
            print(log.format(curr_epoch + 1, num_epochs, train_cost, lr, time.time() - start))
        writer.close()

if __name__ == "__main__":

    checkpoint = "../model_v3/back/LPR_model_v3.ckpt-28000"
    train(checkpoint)