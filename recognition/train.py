import tensorflow as tf 
import numpy as np 
from lpr_model import get_train_model, get_train_model_new
from text_image_generator import TextImageGenerator
from lpr_util import decode_sparse_tensor

import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = " "
#训练最大轮次
num_epochs = 50

#初始化学习速率
INITIAL_LEARNING_RATE = 1e-4
DECAY_STEPS = 2000
LEARNING_RATE_DECAY_FACTOR = 0.95  # The learning rate decay factor
#MOMENTUM = 0.9

#输出字符串结果的步长间隔
REPORT_STEPS = 200
#训练集的数量
BATCH_SIZE = 128
TRAIN_SIZE = 60000
BATCHES = TRAIN_SIZE//BATCH_SIZE
test_num = 1

#统计测试集和训练集大小 ls -l | grep "^-" | wc -l。

train_dir = '/home/store-1-img/xiajinpeng/LPRNET/data/generate/train'
print('train:', train_dir)#训练集位置
test_dir = '/home/store-1-img/xiajinpeng/LPRNET/data/generate/test'
print('test:', test_dir)#验证集位置

model_save_dir = '../model0610/LPR_model.ckpt'
model_best_dir = '../modelbest/LPR_model_best.ckpt'

img_size = [96, 36]
num_channels = 3


def report_accuracy(decoded_list, test_targets):
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)
    true_numer = 0

    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length desn't match")
        return
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        #print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        if hit:
            true_numer = true_numer + 1
        else:
            print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
    print("Test Accuracy:", true_numer * 1.0 / len(original_list))
    return true_numer * 1.0 / len(original_list)


#开始一个新的训练
def train():

    train_gen = TextImageGenerator(img_dir=train_dir,
                                   batch_size=BATCH_SIZE,
                                   img_size=img_size,
                                   num_channels=num_channels)

    val_gen = TextImageGenerator(img_dir=test_dir,
                                 batch_size=BATCH_SIZE,
                                 img_size=img_size,
                                 num_channels=num_channels)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                               global_step,
                                               DECAY_STEPS,
                                               LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)
    #构建网络模型
    logits, inputs, targets, seq_len = get_train_model_new(num_channels, BATCH_SIZE, img_size, training=True)
    logits = tf.transpose(logits, (1, 0, 2))
    # tragets是一个稀疏矩阵
    loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)

    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=MOMENTUM).minimize(cost, global_step=global_step)
    # 通知Tensorflow在训练时要更新均值和方差的分布
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    # 前面说的划分块之后找每块的类属概率分布，ctc_beam_search_decoder方法,是每次找最大的K个概率分布
    # 还有一种贪心策略是只找概率最大那个，也就是K=1的情况ctc_ greedy_decoder
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

    #编辑距离
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    # 增加tensorboard显示
    tf.summary.scalar('loss', cost)
    tf.summary.scalar('acc', acc)
    tf.summary.scalar('learning_rate', learning_rate)

    init = tf.global_variables_initializer()
   
    def do_report(val_gen, session, num):
        sumacc = 0
        for i in range(num):
            test_inputs, test_targets, _, _, test_seq_len, _, _ = val_gen.next_batch()
            test_feed = {inputs: test_inputs, targets: test_targets, seq_len: test_seq_len}
            st =time.time()
            
            dd, v_cost= session.run([decoded[0], cost], test_feed)
            tim = time.time() -st
            print('time:%s, cost:%f'%(tim/BATCH_SIZE, v_cost))
            accu = report_accuracy(dd, test_targets)
            sumacc += accu
        print(sumacc/num)

    def do_batch(train_gen, val_gen, session, merged):
        train_inputs, train_targets, _, _, train_seq_len, _, _ = train_gen.next_batch()

        feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}

        b_loss, b_targets, b_logits, b_seq_len, b_cost, steps, _, lr, summary = session.run(
            [loss, targets, logits, seq_len, cost, global_step, optimizer, learning_rate, merged], feed)
        
        if steps > 0 and steps % 50 == 0:
            print('steps = {}, train loss = {}'.format(steps, b_cost))

        if steps > 0 and steps % REPORT_STEPS == 0:
            print('save model', steps)
            saver.save(session, model_save_dir, global_step=steps)
            do_report(val_gen, session, test_num)

        return b_cost, steps, summary

    with tf.Session() as session:
        session.run(init)

        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        saver = tf.train.Saver(var_list=var_list, max_to_keep=10)
        #saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

        print('Trainable Variables: ', np.sum([np.prod(v.get_shape().as_list()) for v in var_list]))

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs', session.graph)

        bestdic = 100
        beststep = 0
        for curr_epoch in range(num_epochs):
            print("Epoch.......", curr_epoch)
            start = time.time()
            train_cost = 0
            for batch in range(BATCHES):
                c, steps, summary = do_batch(train_gen, val_gen, session, merged)
                train_cost += c * BATCH_SIZE

                if steps > 0 and steps % 50 == 0:
                    writer.add_summary(summary, steps)

            train_cost /= TRAIN_SIZE

            #通过测试集数据验证当前epoch的训练效果
            val_cost = 0
            val_accurate = 0
            for i in range(test_num):
                val_inputs, val_targets, _, _, val_seq_len, _, _ = val_gen.next_batch()
                val_feed = {inputs: val_inputs, targets: val_targets, seq_len: val_seq_len}

                val_cos, val_acc, lr = session.run([cost, acc, learning_rate], feed_dict=val_feed)
                val_cost += val_cos
                val_accurate += val_acc

            val_cost = val_cost/test_num
            val_accurate = val_accurate/test_num
            if val_accurate < bestdic:
                bestdic = val_accurate
                saver.save(session, model_best_dir)
                beststep = steps
                print('save best:',steps)

            log = "Epoch {}/{}, steps = {}, train_cost = {:.6f}, val_cost = {:.6f}, val_ler = {:.6f}, time = {:.3f}s, learning_rate = {:.6f}"
            print(log.format(curr_epoch + 1, num_epochs, steps, train_cost, val_cost, val_accurate, time.time() - start, lr))
        
        print(bestdic, beststep)

        writer.close()

#基于已训练的checkpoint，加载参数，继续进行训练或调优
def train_on_checkpoint(checkpoint):
    train_gen = TextImageGenerator(img_dir=train_dir,
                                   batch_size=BATCH_SIZE,
                                   img_size=img_size,
                                   num_channels=num_channels)

    val_gen = TextImageGenerator(img_dir=test_dir,
                                 batch_size=BATCH_SIZE,
                                 img_size=img_size,
                                 num_channels=num_channels)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                               global_step,
                                               DECAY_STEPS,
                                               LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)
    logits, inputs, targets, seq_len = get_train_model_new(num_channels, BATCH_SIZE, img_size, training=True)
    logits = tf.transpose(logits, (1, 0, 2))
    # tragets是一个稀疏矩阵
    loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)

    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=MOMENTUM).minimize(cost, global_step=global_step)
    # 通知Tensorflow在训练时要更新均值和方差的分布
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    # 前面说的划分块之后找每块的类属概率分布，ctc_beam_search_decoder方法,是每次找最大的K个概率分布
    # 还有一种贪心策略是只找概率最大那个，也就是K=1的情况ctc_ greedy_decoder
    #decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len, merge_repeated=False)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    # 增加tensorboard显示
    tf.summary.scalar('loss', cost)
    tf.summary.scalar('acc', acc)
    tf.summary.scalar('learning_rate', learning_rate)

    init = tf.global_variables_initializer()
   
    def do_report(val_gen, session, num):
        sumacc = 0
        for i in range(num):
            test_inputs, test_targets, _, _, test_seq_len, _, _ = val_gen.next_batch()
            test_feed = {inputs: test_inputs, targets: test_targets, seq_len: test_seq_len}
            st =time.time()
            
            dd, v_cost= session.run([decoded[0], cost], test_feed)
            tim = time.time() -st
            print('time:%s, cost:%f'%(tim/BATCH_SIZE, v_cost))
            accu = report_accuracy(dd, test_targets)
            sumacc += accu
        print(sumacc/num)

    def do_batch(train_gen, val_gen, session, merged):
        train_inputs, train_targets, _, _, train_seq_len, _, _ = train_gen.next_batch()

        feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}

        b_loss, b_targets, b_logits, b_seq_len, b_cost, steps, _, lr, summary = session.run(
            [loss, targets, logits, seq_len, cost, global_step, optimizer, learning_rate, merged], feed)

        if steps > 0 and steps % 50 == 0:
            print('steps = {}, train loss = {}'.format(steps, b_cost))

        if steps > 0 and steps % REPORT_STEPS == 0:
            print('save model', steps)
            saver.save(session, model_save_dir, global_step=steps)
            do_report(val_gen, session, test_num)

        return b_cost, steps, summary

    with tf.Session() as session:
        session.run(init)

        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        saver = tf.train.Saver(var_list=var_list, max_to_keep=10)
        #saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
        saver.restore(session, checkpoint)

        print('Trainable Variables: ', np.sum([np.prod(v.get_shape().as_list()) for v in var_list]))

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs', session.graph)

        bestdic = 100
        beststep = 0
        for curr_epoch in range(num_epochs):
            print("Epoch.......", curr_epoch)
            start = time.time()
            train_cost = 0
            for batch in range(BATCHES):
                c, steps, summary = do_batch(train_gen, val_gen, session, merged)
                train_cost += c * BATCH_SIZE

                if steps > 0 and steps % 50 == 0:
                    writer.add_summary(summary, steps)

            train_cost /= TRAIN_SIZE

            #通过测试集数据验证当前epoch的训练效果
            val_cost = 0
            val_accurate = 0
            for i in range(test_num):
                val_inputs, val_targets, _, _, val_seq_len, _, _ = val_gen.next_batch()
                val_feed = {inputs: val_inputs, targets: val_targets, seq_len: val_seq_len}

                val_cos, val_acc, lr = session.run([cost, acc, learning_rate], feed_dict=val_feed)
                val_cost += val_cos
                val_accurate += val_acc

            val_cost = val_cost/test_num
            val_accurate = val_accurate/test_num
            if val_accurate < bestdic:
                bestdic = val_accurate
                saver.save(session, model_best_dir)
                beststep = steps
                print('save best:',steps)

            log = "Epoch {}/{}, steps = {}, train_cost = {:.6f}, val_cost = {:.6f}, val_ler = {:.6f}, time = {:.3f}s, learning_rate = {:.6f}"
            print(log.format(curr_epoch + 1, num_epochs, steps, train_cost, val_cost, val_accurate, time.time() - start, lr))
        
        print(bestdic, beststep)

        writer.close()

if __name__ == "__main__":
        
    checkpoint = "../model0529/LPR_model.ckpt-11200"
    train_on_checkpoint(checkpoint)
    #train()