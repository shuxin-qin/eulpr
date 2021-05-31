#从CCPD数据集中提取车牌部分图片并进行矩阵变换，生成训练和测试样本，用于对车牌识别部分进行单独训练和测试
# -*- coding: utf-8 -*- 
import os
import cv2
import numpy as np
from lpr_util import DICT

dict2 = {value:key for key, value in DICT.items()}

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

#用于保存车牌号码和对应的数量
lpnumber_dict = {}

#读取指定目录下的所有图片，返回图片完整路径（包含文件名）列表
def get_data_list(imgpath):
    '''获取impath目录下的文件'''
    img_files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    print('getlist')
    for parent, dirnames, filenames in os.walk(imgpath):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    img_files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(img_files)))
    #for test in img_files:
    #    print(test)
    return img_files

#解析CCPD数据集的图片的命名格式。输出：name：车牌号; [x1,x2,x3,x4]：四个角点坐标
#参考样例："025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
def get_name_4vertices(path):
    #os.path.split将文件名和路径分开
    _, fn = os.path.split(path)
    #os.path.splitext将文件名和扩展名分开
    fn, _ = os.path.splitext(fn)
    #number_encoder = ['0', '0', '22', '27', '27', '33', '16']
    lp_number_encoder = fn.split('-')[4].split('_')
    #lp_number_name = 'S01_AY33909S_0.jpg'
    lp_number = decode_lpnumber(lp_number_encoder)

    count = 0

    if lp_number in lpnumber_dict.keys():
        count = lpnumber_dict[lp_number] + 1
    
    lpnumber_dict[lp_number] = count

    lp_number_name = lp_number + '_' + str(count) + '.jpg'
    x1, x2, x3, x4 = [[int(eel) for eel in el.split('&')] for el in fn.split('-')[3].split('_')]
    return lp_number_name, [x1, x2, x3, x4]

#解析CCPD数据集的图片的命名格式。输出：name：车牌号
def get_lpnumber_name(path):
    #os.path.split将文件名和路径分开
    _, fn = os.path.split(path)
    #os.path.splitext将文件名和扩展名分开
    fn, _ = os.path.splitext(fn)
    #number_encoder = ['0', '0', '22', '27', '27', '33', '16']
    lp_number_encoder = fn.split('-')[4].split('_')
    #lp_number_name = 'S01_AY33909S_0.jpg'
    lp_number = decode_lpnumber(lp_number_encoder)

    count = 0

    if lp_number in lpnumber_dict.keys():
        count = lpnumber_dict[lp_number] + 1
    
    lpnumber_dict[lp_number] = count

    lp_number_name = lp_number + '_' + str(count) + '.jpg'
    
    return lp_number_name

#解析CCPD数据集的图片的命名格式。输出：name：车牌号；box：bounding box的坐标(left-up and the right-bottom vertices)
def get_name_bbox(path):
    _, fn = os.path.split(path)
    fn, _ = os.path.splitext(fn)

    lp_number_encoder = fn.split('-')[4].split('_')
    lp_number_name = decode_lpnumber(lp_number_encoder)

    box = [[int(eel) for eel in el.split('&')] for el in fn.split('-')[2].split('_')]
    return lp_number_name, box

def decode1(names):
    pro = dict2[names[0]]
    newname = pro+'_'+names[1:]+'_'+'0.jpg'
    return newname

#解析车牌号，从ccpd的方式到车牌, 例如：输入nammes=['0', '0', '22', '27', '27', '33', '16']，输出newname='S01_AY339S'
def decode_lpnumber(names):
    name = []
    pro = dict2[provinces[int(names[0])]]
    alp = alphabets[int(names[1])]
    for i in names[2: ]:
        name.append(ads[int(i)])
    adss = ''.join(name)
    #lp_number_name = pro + '_' + alp + adss + '_' + '0.jpg'
    lp_number = pro + '_' + alp + adss
    return lp_number

#从源目录转化图片，并保存到目标目录中，num为要转化的数目，如果num为0，则全部转换，转换方法是采用bounding box
def generate_by_bbox(srcpath, dstpath, num=0):
    image_list = np.array(get_data_list(srcpath))
    print('{} training images in {}'.format(image_list.shape[0], DATA_FOLDER))
    index = np.arange(0, image_list.shape[0])
    np.random.shuffle(index)
    count = 0
    while count < num:
        path = image_list[index[count]]
        im = cv2.imread(path)
        name, box = get_name_bbox(path)
        dst = np.array(im)[box[0][1]: box[1][1], box[0][0]: box[1][0]]
        if cv2.imwrite(dstpath + name, dst):
            print('write success'+' '+name+' '+str(count))
        else:
            print('wrong')
        count += 1
        if count == num:
            cv2.imshow('lastone',dst)

#从源目录转化图片，并保存到目标目录中，num为要转化的数目，如果num为0，则全部转换，转换方法是采用4个顶点坐标并进行透视变换
def generate_by_4vertices(srcpath, dstpath, num=0):

    image_list = np.array(get_data_list(srcpath))
    print('{} training images in {}'.format(image_list.shape[0], srcpath))

    if num == 0 or num > image_list.shape[0]:
        num = image_list.shape[0]

    print('the number of images to generate is: ', num)

    index = np.arange(0, image_list.shape[0])
    np.random.shuffle(index)

    count =0
    while count < num:
        path = image_list[index[count]]
        img = cv2.imread(path)
        name, vertices = get_name_4vertices(path)
        points_src = np.float32(vertices)
        points_dst = np.float32([[134, 45], [0, 45], [0, 0], [134, 0]])
        #points_src = np.array(vertices, dtype = 'float32')
        #points_dst = np.array([[134, 45], [0, 45], [0, 0], [134, 0]], dtype = 'float32')
        #rows, cols, ch = img.shape
        #计算透视变换矩阵，返回由源图像中矩形到目标图像矩形变换的矩阵
        mat = cv2.getPerspectiveTransform(points_src, points_dst)
        #通过矩阵变换，得到目标图像，返回图像的大小为(135, 46)
        img_dst = cv2.warpPerspective(img, mat, (135, 46))
        #将生成的图片保存到特定的目录，构建训练集
        if cv2.imwrite(dstpath + name, img_dst):
            print('write success'+' '+name+' '+str(count))
        else:
            print('wrong')
        count += 1

#从源目录转化图片，并按照percent指定的比例保存到train、test目录中
#num为要转化的数目，如果num为0，则全部转换，转换方法是采用4个顶点坐标并进行透视变换
def generate_by_4vertices(srcpath, trainpath, testpath, num=0, percent=0.7):

    image_list = np.array(get_data_list(srcpath))
    print('{} images in {}'.format(image_list.shape[0], srcpath))

    if num == 0 or num > image_list.shape[0]:
        num = image_list.shape[0]

    print('the number of images to generate is: ', num)

    index = np.arange(0, image_list.shape[0])
    np.random.shuffle(index)

    lpnumber_dict.clear()
    count =0
    train_num = int(percent*num)
    dstpath = trainpath
    while count < num:
        path = image_list[index[count]]
        img = cv2.imread(path)
        name, vertices = get_name_4vertices(path)
        points_src = np.float32(vertices)
        points_dst = np.float32([[134, 45], [0, 45], [0, 0], [134, 0]])
        #points_src = np.array(vertices, dtype = 'float32')
        #points_dst = np.array([[134, 45], [0, 45], [0, 0], [134, 0]], dtype = 'float32')
        #rows, cols, ch = img.shape
        #计算透视变换矩阵，返回由源图像中矩形到目标图像矩形变换的矩阵
        mat = cv2.getPerspectiveTransform(points_src, points_dst)
        #通过矩阵变换，得到目标图像，返回图像的大小为(135, 46)
        img_dst = cv2.warpPerspective(img, mat, (135, 46))

        if count > train_num:
            dstpath = testpath

        #将生成的图片保存到特定的目录，构建训练集
        if cv2.imwrite(dstpath + name, img_dst):
            print(count, name)
        else:
            print('failed')
        count += 1
    print('total number of images:', count)

#解析检测结果图片名称，保存为'S01_AY339S'格式
def generate_by_detection(srcpath, dstpath, num=0):
    image_list = np.array(get_data_list(srcpath))
    print('{} images in {}'.format(image_list.shape[0], srcpath))

    if num == 0 or num > image_list.shape[0]:
        num = image_list.shape[0]

    index = np.arange(0, image_list.shape[0])
    np.random.shuffle(index)

    lpnumber_dict.clear()
    count =0
    while count < num:
        path = image_list[index[count]]
        img = cv2.imread(path)
        name = get_lpnumber_name(path)
        #(135, 46)
        img_dst = cv2.resize(img, (135, 46), interpolation=cv2.INTER_AREA)
        #将生成的图片保存到特定的目录
        if cv2.imwrite(dstpath + name, img_dst):
            print(count, name)
        else:
            print('failed')
        count += 1
    print('total number of images:', count)

if __name__ == '__main__':

    #srcpath = '/home/store-1-img/xiajinpeng/CCPD2019/ccpd_weather'
    #trainpath = '/home/store-1-img/xiajinpeng/LPRNET/data/weather/'
    #testpath = '/home/store-1-img/xiajinpeng/LPRNET/data/weather/'
    #generate_by_4vertices(srcpath, trainpath, testpath, percent=1)

    srcpath = '/home/store-1-img/xiajinpeng/LPRNET/data/base/base_detection_src'
    dstpath = '/home/store-1-img/xiajinpeng/LPRNET/data/base/base_detection/'
    generate_by_detection(srcpath, dstpath)
