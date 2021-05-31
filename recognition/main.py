# -*- coding: utf-8 -*-

import tensorflow as tf 
import numpy as np
from detection.dataset import PrDataset
from detection.wpod import WpodNet
from lpr_model import get_train_model_multitask_v2
from text_image_generator import TextImageGenerator
from lpr_util import decode_sparse_tensor, is_legal_lpnumber

import time
import os
import math

os.environ["CUDA_VISIBLE_DEVICES"] = " "

#default parameters
DefaultParam = {
    "mode": "total",   #[detection, recognition, total]
    "data_dir": "../Data/test",
    "detect_model_dir": "model/detection",
    "recog_model_dir": "model/recognition",
    "log_dir": "logs/log_test",
    "batch_size": 64,
    "det_img_size": [208, 208],
    "rec_img_size": [96, 36]
}


class LPDetecAndRecog:
    def __init__(self, param):

        self._mode = param["mode"]
        self._data_dir = param["data_dir"]
        self._det_model_dir = param["detect_model_dir"]
        self._rec_model_dir = param["recog_model_dir"]
        self._log_dir = param["log_dir"]
        self._batch_size = param["batch_size"]
        self._det_img_size = param["det_img_size"]
        self._rec_img_size = param["rec_img_size"]

        self._det_data_manager = None
        self._rec_data_manager = None
        self._det_model = None
        self._rec_model = None

        self._det_session = None
        self._rec_session = None
        self._det_graph = None
        self._rec_graph = None

        self.load_model()
        self.load_data()

    def load_model():
        if self._mode not in ["total", "detection", "recognition"]:
            raise Exception('got a unexpected mode,options :{total, detection, recognition}')

        if self._mode == "total":
            
            self._det_model = WpodNet(cfg.NETCFG)
            self._rec_model =


    def load_data():
        if self._mode == "recognition":
            self._rec_data_manager = TextImageGenerator(img_dir=self._data_dir, batch_size=self._batch_size, img_size=self._rec_img_size, num_channels=3)
        else:
            self._det_data_manager = PrDataset.load_from_path(img_dir=self._data_dir).make_initializable_iterator()

    def detection():
        

    def recogniton():
        

    def visualization():
        

    def run():
        self._det_session = tf.Session()
        self._rec_session = tf.Session()


if __name__ == '__main__':

    LPDetecAndRecog(DefaultParam).run()
