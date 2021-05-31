import os
import numpy as np
import cv2
from recognition.lpr_util import sparse_tuple_from, encode_label_multi, DICT, decode_sparse_tensor

#读取图片和label,产生batch
class TextImageGenerator:
    '''读取图片'''
    def __init__(self, img_dir, batch_size, img_size, num_channels=3, isClassify=True):
        self._img_dir = img_dir
        self._batch_size = batch_size
        self._num_channels = num_channels
        self._img_w, self._img_h = img_size

        self._num_examples = 0
        self._next_index = 0
        self._num_epoches = 0

        self._filenames = []
        self._labels = []
        self._labels_up = []
        self._labels_down = []
        self._labels_class = []

        self.filenames = []
        self.labels = []
        self.labels_up = []
        self.labels_down = []
        self.labels_class = []
        self.isClassify = isClassify

        self.init()

    def init(self):
        fs = os.listdir(self._img_dir)
        for filename in fs:
            self.filenames.append(filename)

        for filename in self.filenames:
            lp = filename.split('_')

            if '\u4e00' <= filename[0]<= '\u9fff':
                lp_len = len(lp[0])
                label = filename[:lp_len]
            else:
                lp_len = 4 + len(lp[1])
                label = DICT[filename[:3]] + filename[4:lp_len]

            if self.isClassify:
                flag = filename[lp_len+3]
                if not flag.isdigit():
                    flag = 0
            else:
                flag = 0

            label, label_up, label_down = encode_label_multi(label)
            self.labels.append(label)
            self.labels_up.append(label_up)
            self.labels_down.append(label_down)
            self.labels_class.append(int(flag))
            self._num_examples += 1

        # Shuffle the data
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._filenames = [self.filenames[i] for i in perm]
        self._labels = np.array(self.labels)[perm]
        self._labels_up = np.array(self.labels_up)[perm]
        self._labels_down = np.array(self.labels_down)[perm]
        self._labels_class = np.array(self.labels_class)[perm]


    def next_batch(self):
        batch_size = self._batch_size
        start = self._next_index
        end = self._next_index + batch_size
        if end > self._num_examples:
            self._num_epoches += 1            
            self._next_index = 0

            # Shuffle the data again for a new epoch start
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._filenames = [self.filenames[i] for i in perm]
            self._labels = np.array(self.labels)[perm]
            self._labels_up = np.array(self.labels_up)[perm]
            self._labels_down = np.array(self.labels_down)[perm]
            self._labels_class = np.array(self.labels_class)[perm]

            #update start/end index
            start = self._next_index
            end = self._next_index + batch_size

        #update next index
        self._next_index = end

        images = np.zeros([batch_size, self._img_h, self._img_w, self._num_channels])
        #images = np.zeros([batch_size, self._img_w, self._img_h, self._num_channels])
        # labels = np.zeros([batch_size, self._label_len])

        for j, i in enumerate(range(start, end)):
            fname = self._filenames[i]
            #cv2.imread()按照（H,W,C）格式返回numpy.ndarray对象，通道顺序为BGR
            img = cv2.imread(os.path.join(self._img_dir, fname))
            #cv2.resize()
            img = cv2.resize(img, (self._img_w, self._img_h), interpolation=cv2.INTER_CUBIC)
            images[j, ...] = img
        #将images矩阵调整为[batch_size, self._img_w, self._img_h, self._num_channels]----[b,94,24,3]
        images = np.transpose(images, axes=[0, 2, 1, 3])

        labels = self._labels[start:end, ...]
        labels_up = self._labels_up[start:end, ...]
        labels_down = self._labels_down[start:end, ...]
        labels_class = self._labels_class[start:end, ...]

        targets = [np.asarray(i) for i in labels]
        targets_up = [np.asarray(i) for i in labels_up]
        targets_down = [np.asarray(i) for i in labels_down]
        targets_class = [[i] for i in labels_class]
        targets_class = np.array(targets_class)

        sparse_labels = sparse_tuple_from(targets)
        sparse_labels_up = sparse_tuple_from(targets_up)
        sparse_labels_down = sparse_tuple_from(targets_down)
        # input_length = np.zeros([batch_size, 1])

        return images, sparse_labels, sparse_labels_up, sparse_labels_down, targets_class

if __name__ == "__main__":
    img_dir = '../data/generate/train_s'
    gen = TextImageGenerator(img_dir=img_dir, batch_size=50, img_size=[96, 36], num_channels=3)
    
    file_num = len(os.listdir(img_dir))
    print("all images: ", file_num)
    n_batch = file_num//50
    print("all images: ", file_num, n_batch)
    for i in range(n_batch):
        _, sparse_labels, sparse_labels_up, sparse_labels_down, labels_class = gen.next_batch()
        #print(decode_sparse_tensor(sparse_labels))
        #print(decode_sparse_tensor(sparse_labels_up))
        
        dec = decode_sparse_tensor(sparse_labels)
        dec1 = decode_sparse_tensor(sparse_labels_up)
        dec2 = decode_sparse_tensor(sparse_labels_down)

        for j in range(len(dec)):
            print(i*50+j, len(dec[j]), len(dec1[j]), len(dec2[j]), labels_class[j])
            if len(dec2[j]) > 6:
                print("........................................")