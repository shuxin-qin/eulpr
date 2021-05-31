import tensorflow as tf 
import numpy as np 
from lpr_model import get_train_model, get_train_model_new
from text_image_generator import TextImageGenerator
from lpr_util import decode_sparse_tensor, is_legal_lpnumber

import time
import os
import math

os.environ["CUDA_VISIBLE_DEVICES"] = " "

#训练集的数量
BATCH_SIZE = 64

img_size = [96, 36]
num_channels = 3
label_len = 7

#查找所有解码序列，寻找符合规则且概率最高的序列作为识别结果
def pl_regularization_old(bs_decodes, log_prob):
    #所有通过beam search返回的序列
    detec_lists = []
    #识别结果序列
    detected_list = []
    #detected_list_prob = []
    #解码所有beam search序列
    
    for j in range(len(bs_decodes)):
        d_batch = decode_sparse_tensor(bs_decodes[j])
        detec_lists.append(d_batch)

    for b in range(BATCH_SIZE):
        detect = detec_lists[0][b]
        #detect_prob = log_prob[b][0]
        for i in range(len(detec_lists)):
            d = detec_lists[i][b]
            if is_legal_lpnumber(d, [7, 8]):
                detect = d
                #detect_prob = math.exp(log_prob[b][j])
                break
        detected_list.append(detect)
        #detected_list_prob.append(detect_prob)
    return detected_list


#查找所有解码序列，寻找符合规则且概率最高的序列作为识别结果
def pl_regularization(bs_decodes, log_prob):
    #识别结果序列
    detected_list = []
    #detected_list_prob = []
    length = len(decode_sparse_tensor(bs_decodes[0]))
    #print(len(bs_decodes), length)
    detec_lists = [[] for i in range(length)]

    #解码所有beam search序列
    for i in range(len(bs_decodes)):
        d_batch = decode_sparse_tensor(bs_decodes[i])
        for j in range(len(d_batch)):
            detec_lists[j].append(d_batch[j])
            
    for i in range(len(detec_lists)):
        detects = detec_lists[i]
        detect = detec_lists[i][0]
        for j in range(len(detects)):
            d = detects[j]
            if is_legal_lpnumber(d, [7, 8]):
                detect = d
                break
        detected_list.append(detect)
        #detected_list_prob.append(detect_prob)
    return detected_list

    
def test(model_dir, img_dir, test_file_num=0):

    global_step = tf.Variable(0, trainable=False)

    logits, inputs, _, _ = get_train_model_new(num_channels, BATCH_SIZE, img_size, training=False)
    logits = tf.transpose(logits, (1, 0, 2))

    seq_len = tf.placeholder(tf.int32, [None])
    seq_len_feed = np.ones(BATCH_SIZE) * 24
    # 前面说的划分块之后找每块的类属概率分布，ctc_beam_search_decoder方法,是每次找最大的K个概率分布
    # 还有一种贪心策略是只找概率最大那个，也就是K=1的情况ctc_ greedy_decoder
    #decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len, merge_repeated=False)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, top_paths=20, merge_repeated=False)

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars

        #saver = tf.train.Saver(var_list, max_to_keep=10)
        saver = tf.train.Saver(var_list, max_to_keep=20)
        
        saver.restore(session, model_dir)

        print('Trainable Variables: ', np.sum([np.prod(v.get_shape().as_list()) for v in var_list]))
        test_gen = TextImageGenerator(img_dir=img_dir, batch_size=BATCH_SIZE, img_size=img_size, num_channels=num_channels, isClassify=False)
        #统计文件夹下测试样本的数量
        file_num = len(os.listdir(img_dir))
        print("all images: ", file_num)

        if test_file_num == 0 or test_file_num > file_num:
            test_file_num = file_num

        print("test images: ", test_file_num)

        true_number = 0
        num_batch = test_file_num//BATCH_SIZE
        test_file_num = num_batch*BATCH_SIZE

        start =time.time()
        for i in range(num_batch):
            test_inputs, test_targets, _, _, _ = test_gen.next_batch()
            test_feed = {inputs: test_inputs, seq_len: seq_len_feed}
            
            decodes, lg  = session.run([decoded, log_prob], test_feed)

            detected_list = pl_regularization(decodes, lg)

            original_list = decode_sparse_tensor(test_targets)
            #detected_list = decode_sparse_tensor(decodes[0])
                        
            for idx, number in enumerate(original_list):
                detect_number = detected_list[idx]
                hit = (number == detect_number)
                #print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
                if hit:
                    true_number = true_number + 1
                else:
                    print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
            #endt = time.time() -startt
            #print('time: %s'%endt)
            print(true_number,'/',(i+1)*BATCH_SIZE)
        print("Test Accuracy: {}, time per sample: {}".format(true_number*1.0/test_file_num, (time.time()-start)/test_file_num))

def test_beam_search():
    global_step = tf.Variable(0, trainable=False)
    batchsize = 4
    logits, inputs, targets, seq_len = get_train_model_new(num_channels, label_len, batchsize, img_size)
    logits = tf.transpose(logits, (1, 0, 2))

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, top_paths=20, merge_repeated=False)

    init = tf.global_variables_initializer()

    img_dir='/home/store-1-img/xiajinpeng/LPRNET/data/temp'
    model_dir = '../model0529/LPR_model.ckpt-46000'

    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
        
        saver.restore(session, model_dir)

        test_gen = TextImageGenerator(img_dir=img_dir,
                                      batch_size=batchsize,
                                      img_size=img_size,
                                      num_channels=num_channels)
        true_number = 0
        num_batch = 1
        for i in range(num_batch):
            test_inputs, test_targets, test_seq_len = test_gen.next_batch()
            test_feed = {inputs: test_inputs, targets: test_targets, seq_len: test_seq_len}
            
            decodes, lg  = session.run([decoded, log_prob], test_feed)

            detected_list = []
            #查找所有解码序列，寻找第一个（概率最高的）符合规则的序列
            detec_lists = []
            for j in range(len(decodes)):
                d_batch = decode_sparse_tensor(decodes[j])
                detec_lists.append(d_batch)

            for b in range(batchsize):
                detect = detec_lists[0][b]
                for j in range(len(detec_lists)):
                    d = detec_lists[j][b]
                    lg_p = lg[b][j]
                    print(b, d, math.exp(lg_p))
                    if len(d) == 7:
                        detect = d
                        break
                detected_list.append(detect)
            original_list = decode_sparse_tensor(test_targets)

            for idx, number in enumerate(original_list):
                detect_number = detected_list[idx]
                hit = (number == detect_number)
                print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")


if __name__ == '__main__':

    #model_dir = '../model3573/LPRtf3.ckpt'
    #model_dir = '../model3575/LPRtf5.ckpt-50000'
    #model_dir = '../model3574/LPRtf4.ckpt-75000'   99.501%
    #model_dir = '../model3574/LPRtf4.ckpt-80600'   99.495%
    #model_dir = '../model3574/LPRtf4.ckpt-64600'   99.513%
    #model_dir = '../model3574/LPRtf4.ckpt-96600'   99.418%
    #model_dir = '../model0524/LPR_model.ckpt-23000'  99.6911%
    #model_dir = '../model0524/LPR_model.ckpt-23400'  99.7666%
    #model_dir = '../model0524/LPR_model.ckpt-24000'  99.7671%
    #model_dir = '../model0524/LPR_model.ckpt-25000'  99.7696%
    #model_dir = '../model0524/LPR_model.ckpt-25400'  99.7321%
    #model_dir = '../model0524/LPR_model.ckpt-26000'  99.7471%
    model_dir = '../model0610/LPR_model.ckpt-23400'

    #img_dir='/home/store-1-img/xiajinpeng/LPRNET/data/base/test'
    #img_dir='/home/store-1-img/xiajinpeng/LPRNET/data/weather'
    #img_dir='/home/store-1-img/xiajinpeng/LPRNET/data/tilt'
    #img_dir='/home/store-1-img/xiajinpeng/LPRNET/data/rotate'
    #img_dir='/home/store-1-img/xiajinpeng/LPRNET/data/fn'
    #img_dir='/home/store-1-img/xiajinpeng/LPRNET/data/db'
    #img_dir='/home/store-1-img/xiajinpeng/LPRNET/data/challenge'
    img_dir='/home/store-1-img/xiajinpeng/LPRNET/data/blur'
    #img_dir='/home/store-1-img/xiajinpeng/LPRNET/data/32_plates_test'
    
    #img_dir='/home/store-1-img/xiajinpeng/LPRNET/data/base/base_detection'

    #img_dir='/home/store-1-img/xiajinpeng/LPRNET/data/generate/test'
    #img_dir='/home/store-1-img/xiajinpeng/LPRNET/data/temp'

    test(model_dir, img_dir)
    #test_beam_search()
