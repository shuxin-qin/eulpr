import tensorflow as tf 
import numpy as np 
from recognition.lpr_net_1 import LPRNet
from recognition.text_image_generator import TextImageGenerator
from recognition.lpr_util import decode_sparse_tensor, is_legal_lpnumber

import time
import os
import math

os.environ["CUDA_VISIBLE_DEVICES"] = " "


BATCH_SIZE = 1

img_size = [112, 36]
num_channels = 3

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

    #定义输入张量
    inputs = tf.placeholder(tf.float32, shape=(BATCH_SIZE, img_size[0], img_size[1], num_channels))
    seq_len = tf.constant(np.ones(BATCH_SIZE, dtype=np.int32) * 24)

    #构建网络，获取logit
    logits, logits_up, logits_down, logits_cla = LPRNet(training = False).build_net(inputs)
    logits = tf.transpose(logits, (1, 0, 2))
    logits_up = tf.transpose(logits_up, (1, 0, 2))
    logits_down = tf.transpose(logits_down, (1, 0, 2))

    #init = tf.global_variables_initializer()
    decoded, _ = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    decoded_up, _ = tf.nn.ctc_beam_search_decoder(logits_up, seq_len, merge_repeated=False)
    decoded_down, _ = tf.nn.ctc_beam_search_decoder(logits_down, seq_len, merge_repeated=False)

    with tf.Session() as session:
        #session.run(init)

        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars

        saver = tf.train.Saver(var_list, max_to_keep=10)
        saver.restore(session, model_dir)

        print('Trainable Variables: ', np.sum([np.prod(v.get_shape().as_list()) for v in var_list]))
        test_gen = TextImageGenerator(img_dir=img_dir, batch_size=BATCH_SIZE, img_size=img_size, num_channels=num_channels)
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
            test_feed = {inputs : test_inputs}
            dd, dd_up, dd_down, classify = session.run(
                [decoded[0], decoded_up[0], decoded_down[0], logits_cla], test_feed)

            #根据分类结果提取LP的识别结果序列
            dd_list = decode_sparse_tensor(dd)
            dd_up_list = decode_sparse_tensor(dd_up)
            dd_down_list = decode_sparse_tensor(dd_down)

            or_list = decode_sparse_tensor(test_targets)

            detect_class = classify.reshape(-1).tolist()

            for idx, det_cla in enumerate(detect_class):
                #print('class:', original_class[idx], "<----->", det_cla)
                if det_cla < 0.5:
                    detect_number = dd_list[idx]
                    clas = 0
                else:
                    detect_number = dd_up_list[idx] + dd_down_list[idx]
                    clas = 1
                hit = (detect_number == or_list[idx])
                if hit:
                    true_number += 1
                else:
                    print(hit, or_list[idx], "(", len(or_list[idx]), ") <-------> ",clas,detect_number, "(", len(detect_number), ")")

            print(true_number,'/',(i+1)*BATCH_SIZE)
        print("Test Accuracy: {}, time per sample: {}".format(true_number*1.0/test_file_num, (time.time()-start)/test_file_num))


if __name__ == '__main__':

    model_dir = '../model_v3/LPR_model_v3.ckpt-56000'

    #img_dir='/home/public/share/01Datasets/LP_recognition/generate/test_s'
    #img_dir='../data/generate/twoline' #94.05%
    img_dir='/home/public/share/01Datasets/LP_recognition/generate/twoline' #97.9%
    #img_dir='../data/generate/oneline_8'  #96.93%
    #img_dir='/home/public/share/01Datasets/LP_recognition/base/base_detection_s'

    test(model_dir, img_dir)