# -*- coding: utf-8 -*-

import os
import random
import shutil
from lpr_util import DICT

#将一个文件夹下面的文件按照percent指定的比例随机抽取并移动到另一个目录下
def move_file(src_dir, dst_dir, percent=0, pick_num=0):

    #获取目录下所有图片的文件名
    file_names = os.listdir(src_dir)
    file_num = len(file_names)
    if percent != 0:
        pick_num = int(file_num*percent)

    if pick_num > file_num:
        pick_num = file_num
    print('total files to move:', pick_num)
    sample = random.sample(file_names, pick_num)
    #print(sample)
    i = 0
    for file_name in sample:
        shutil.move(src_dir+file_name, dst_dir+file_name)
        print(i, file_name)
        i = i+1
    return

#将一个文件夹下面的文件按照percent指定的比例随机抽取并复制到另一个目录下
def copy_file(src_dir, dst_dir, percent=0, pick_num=0):

    #获取目录下所有图片的文件名
    file_names = os.listdir(src_dir)
    file_num = len(file_names)
    if percent != 0:
        pick_num = int(file_num*percent)
    elif pick_num == 0:
        pick_num = file_num
    if pick_num > file_num:
        pick_num = file_num
    print('total files to copy:', pick_num)
    sample = random.sample(file_names, pick_num)
    #print(sample)
    i = 0
    for file_name in sample:
        i = i+1
        #file_name_d = file_name[0:12] + '_0.jpg'
        shutil.copy(src_dir+file_name, dst_dir+file_name)
        print(i, file_name)

    return

#复制文件，排除名称中含关键字key（首个字）的文件
def copy_file_opt(src_dir, dst_dir, key):
    #用于保存车牌号码和对应的数量
    #lpnumber_dict = {}
    #获取目录下所有图片的文件名
    file_names = os.listdir(src_dir)
    #file_names_d = os.listdir(dst_dir)
    #for name in file_names_d:
    #    lp = name[0:11]
    #    num = name[11]
    #    lpnumber_dict[lp] = int(num)
    i = 0
    for file_name in file_names:
        if DICT[file_name[0:3]] != key:
            i += 1
            #count = 0
            #lp_number = file_name[0:11]
            #if lp_number in lpnumber_dict.keys():
            #    count = lpnumber_dict[lp_number] + 1
            #lpnumber_dict[lp_number] = count
            file_name_d = file_name[0:12] + '_0.jpg'

            shutil.copy(src_dir+file_name, dst_dir+file_name_d)
            print(i, file_name_d)
    return

if __name__ == '__main__':

    src_dir = '/home/qinshuxin/datasets/ccpd_recog_mix/tilt/'
    dst_dir = '/home/qinshuxin/datasets/ccpd_recog_mix/train_mix/'

    move_file(src_dir, dst_dir, pick_num=15000)
    #key = '皖'
    #copy_file_opt(src_dir, dst_dir, key)













