# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from recognition.lpr_util import sparse_tuple_from, DICT, decode_sparse_tensor


dict2 = {value:key for key, value in DICT.items()}

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

class DataGenerator:
    """docstring for DataGenerator"""
    def __init__(self, img_dir, batch_size=1, img_size=[0, 0], num_channels=3):
        
        self._img_dir = img_dir
        self._batch_size = batch_size
        self._img_w, self._img_h = img_size
        self._num_channels = num_channels

        self._num_examples = 0
        self._next_index = 0

        self._filenames = []
        self._labels = []

        self.filenames = []
        self.labels = []

        self.init()

    def init(self):
        self.filenames = self.get_data_list()
        self._num_examples = len(self.filenames)
        for filename in self.filenames:
            fn, _ = os.path.splitext(filename)  #0_0_22_27_27_33_16
            if len(fn) < 7:
                self.labels.append(0)
            elif '\u4e00' <= fn[0]<= '\u9fff':
                self.labels.append(fn)
            else:
                lp_number_encoder = fn.split('-')[4].split('_')
                #lp_number_name = 'S01_AY33909S_0.jpg'
                lp_number = self.decode_lpnumber(lp_number_encoder)
                lp_len = len(lp_number)
                self.labels.append(DICT[lp_number[:3]] + lp_number[4:lp_len])

        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._filenames = [self.filenames[i] for i in perm]
        self._labels = np.array(self.labels)[perm]

    def next_batch(self, mode='recognition'):    #mode = ['detection' or 'recognition']
        start = self._next_index
        end = start + self._batch_size
        if end > self._num_examples:
            raise Exception('There are no enough data left for a batch!')
        self._next_index = end
        labels = []
        for j, i in enumerate(range(start, end)):
            fname = self._filenames[i]
            #cv2.imread()按照（H,W,C）格式返回numpy.ndarray对象，通道顺序为BGR
            #img = cv2.imread(os.path.join(self._img_dir, fname))
            file_path = os.path.join(self._img_dir, fname)
            #print(file_path)
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            ratio = float(max(img.shape[:2])) / min(img.shape[:2])
            side = int(ratio * 288.)
            bound_dim = min(side + (side % (2 ** 4)), 608)

            I = self.im2single(img)
            min_dim_img = min(I.shape[:2])
            factor = float(bound_dim)/min_dim_img
            net_step = 16
            w,h = (np.array(I.shape[1::-1],dtype=float)*factor).astype(int).tolist()
            w += (w%net_step!=0)*(net_step - w%net_step)
            h += (h%net_step!=0)*(net_step - h%net_step)
            #print(w, h)
            Iresized = cv2.resize(I,(w,h))

            T = Iresized.copy()
            T = T.reshape((1, T.shape[0],T.shape[1],T.shape[2]))

            if mode == 'detection' and len(fname) > 10:
                pt4 = file_path.split('-')[-4].split('_')
                pt4 = np.array(pt4)[[2, 3, 0, 1]]
                w, h = img.shape[1], img.shape[0]
                ps1 = np.array([self.divi(pt.split('&'), [w, h]) for pt in pt4])
                labels.append(ps1)
        if mode == 'recognition':
            labels = self._labels[start:end, ...]
            labels = [list(i) for i in labels]

        return I, Iresized, T, labels, fname

    #读取指定目录下的所有图片，返回图片完整路径（包含文件名）列表
    def get_data_list(self):
        '''获取目录下的文件'''
        img_files = []
        exts = ['jpg', 'png', 'jpeg', 'JPG']
        for parent, dirnames, filenames in os.walk(self._img_dir):
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        img_files.append(filename)
                        break
        return img_files

    def decode_lpnumber(self, names):
        name = []
        pro = dict2[provinces[int(names[0])]]
        alp = alphabets[int(names[1])]
        for i in names[2: ]:
            name.append(ads[int(i)])
        adss = ''.join(name)
        lp_number = pro + '_' + alp + adss
        return lp_number

    def im2single(self, I):
        assert(I.dtype == 'uint8')
        return I.astype('float32')/255.

    def has_chinese(self, str):
        for ch in str:
            if '\u4e00' <= ch<= '\u9fff':
                return True
        return False

    def divi(self, list1, list2):
        return [round(float(list1[0]) / list2[0], 6), round(float(list1[1]) / list2[1], 6)]