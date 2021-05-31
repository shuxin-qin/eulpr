from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
import os
import math
from matplotlib import pyplot as plt

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', '_'
         ]
DICT = {'A01': '京', 'A02': '津', 'A03': '沪', 'B02': '蒙',
        'S01': '皖', 'S02': '闽', 'S03': '粤', 'S04': '甘',
        'S05': '贵', 'S06': '鄂', 'S07': '冀', 'S08': '黑', 'S09': '湘',
        'S10': '豫', 'S12': '吉', 'S13': '苏', 'S14': '赣', 'S15': '辽',
        'S17': '川', 'S18': '鲁', 'S22': '浙',
        'S30': '渝', 'S31': '晋', 'S32': '桂', 'S33': '琼', 'S34': '云', 'S35': '藏',
        'S36': '陕', 'S37': '青', 'S38': '宁', 'S39': '新'}

dict2 = {value: key for key, value in DICT.items()}
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}

NUM_CHARS = len(CHARS)

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
         "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
         "Y", "Z"
         ]

#用于保存车牌号码和对应的数量
lpnumber_dict = {}

#添加污迹
def add_smudginess(img, Smu):
    rows = r(Smu.shape[0] - 50)
    cols = r(Smu.shape[1] - 50)
    adder = Smu[rows:rows + 50, cols:cols + 50]
    adder = cv2.resize(adder, (50, 50))
    # adder = cv2.bitwise_not(adder)
    img = cv2.resize(img, (50, 50))
    img = cv2.bitwise_not(img)
    img = cv2.bitwise_and(adder, img)
    img = cv2.bitwise_not(img)
    return img

#添加仿射畸变，img 输入图像，angel畸变的参数，size为目标图片的尺寸
def add_affine(img, angel, shape, max_angel, borderValue=(0, 0, 0)):
    size_o = [shape[1], shape[0]]
    size = (shape[1] + int(shape[0] * math.cos((float(max_angel) / 180) * math.pi)), shape[0])
    interval = abs(int(math.sin((float(angel) / 180) * math.pi) * shape[0]));
    pts1 = np.float32([[0, 0], [0, size_o[1]], [size_o[0], 0]])
    if (angel > 0):
        pts2 = np.float32([[interval, 0], [0, size[1]], [size[0], 0]])
    else:
        pts2 = np.float32([[0, 0], [interval, size[1]], [size[0] - interval, 0]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, size, borderMode=cv2.BORDER_CONSTANT, borderValue=borderValue)
    return dst

#添加透视畸变
def add_perspective(img, factor, size, borderValue=(0, 0, 0)):
    shape = size
    t = factor/2
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor)-t, r(factor)-t], [r(factor)-t, shape[0]+r(factor)-t], 
                      [shape[1]+r(factor)-t, r(factor)-t], [shape[1]-r(factor)+t, shape[0]-r(factor)+t]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size, borderMode=cv2.BORDER_CONSTANT, borderValue=borderValue)
    return dst

#添加饱和度光照的噪声
def add_tfactor(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = hsv[:, :, 0] * (0.8 + np.random.random() * 0.2)
    hsv[:, :, 1] = hsv[:, :, 1] * (0.7 + np.random.random() * 0.3)
    hsv[:, :, 2] = hsv[:, :, 2] * (0.5 + np.random.random() * 0.5)

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

#添加自然环境的噪声
def random_envirment(img, data_set):
    index = r(len(data_set))
    env = cv2.imread(data_set[index])
    env = cv2.resize(env, (img.shape[1], img.shape[0]))
    bak = (img == 0)
    bak = bak.astype(np.uint8) * 255
    inv = cv2.bitwise_and(bak, env)
    img = cv2.bitwise_or(inv, img)
    return img

#添加高斯模糊
def add_gauss(img, level):
    return cv2.blur(img, (level * 2 + 1, level * 2 + 1))

#添加随机因子，np.random.random()随机样本取值范围是[0,1)
def r(val):
    return int(np.random.random() * val)

#添加高斯噪声（单个通道）
def AddNoiseSingleChannel(single):
    diff = 255 - single.max()
    noise = np.random.normal(0, 1 + r(1), single.shape)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = diff * noise
    noise = noise.astype(np.uint8)
    dst = single + noise
    return dst

#添加高斯噪声
def add_noise(img, sdev=0.5, avg=10):
    img[:, :, 0] = AddNoiseSingleChannel(img[:, :, 0])
    img[:, :, 1] = AddNoiseSingleChannel(img[:, :, 1])
    img[:, :, 2] = AddNoiseSingleChannel(img[:, :, 2])
    return img

#随机生成font
def randomfontcolor():
    a = np.random.random()
    char = [int(a * 50) for i in range(3)]
    bg = [255 - char[i] for i in range(3)]
    if np.random.random() < 0.5:
        return (tuple(char), tuple(bg))
    else:
        return (tuple(bg), tuple(char))

#生成中文字符
def GenCh(f, val, color=(20, 20, 20), size=(23, 70)):
    img = Image.new("RGB", (45, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 3), val, color, font=f)
    img = img.resize(size)
    A = np.array(img)
    return A

#生成中文字符
def GenCh_s(f, val, color=(20, 20, 20)):
    img = Image.new("RGB", (50, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 5), val, color, font=f)
    img = img.resize((45, 45))
    A = np.array(img)
    return A

#生成英文字符
def GenEn(f, val, color=(20, 20, 20), size=(23, 70)):
    if val == 'W':
        img = Image.new("RGB", (28, 70), (255, 255, 255))
    else:
        img = Image.new("RGB", (23, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 2), val, color, font=f)
    img = img.resize(size)
    A = np.array(img)
    return A

#生成英文字符
def GenEn_s(f, val, color=(20, 20, 20)):
    if val == 'W':
        img = Image.new("RGB", (30, 70), (255, 255, 255))
    else:
        img = Image.new("RGB", (25, 70), (255, 255, 255))
    #img = Image.new("RGB", (25, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 5), val, color, font=f)
    img = img.resize((45, 45))
    A = np.array(img)
    return A

#生成英文字符
def GenEn_s1(f, val, color=(20, 20, 20)):
    if val == 'W':
        img = Image.new("RGB", (28, 65), (255, 255, 255))
    else:
        img = Image.new("RGB", (23, 65), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), val, color, font=f)
    img = img.resize((35, 72))
    A = np.array(img)
    return A

#Add salt and pepper noise to image, prob: Probability of the noise
#添加椒盐噪声
def sp_noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

#展示图片
def show(img):
    show_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(show_img)
    plt.show()


class GenPlate:

    def __init__(self, fontCh, fontEng, env_dir):
        self.fontC = ImageFont.truetype(fontCh, 45, 0)
        self.fontE = ImageFont.truetype(fontEng, 60, 0)
        self.img = np.array(Image.new("RGB", (226, 70), (255, 255, 255)))
        self.img_s = np.array(Image.new("RGB", (226, 105), (255, 255, 255)))
        self.bg1 = cv2.resize(cv2.imread("../data/generate/template/blue.jpg"), (226, 70))
        self.bg2 = cv2.resize(cv2.imread("../data/generate/template/yellow.jpg"), (226, 70))
        self.bg2 = cv2.bitwise_not(self.bg2)
        self.bg3 = cv2.resize(cv2.imread("../data/generate/template/green1.jpg"), (226, 70))
        self.bg3 = cv2.bitwise_not(self.bg3)
        self.bg4 = cv2.resize(cv2.imread("../data/generate/template/green2.jpg"), (226, 70))
        self.bg4 = cv2.bitwise_not(self.bg4)
        self.bg5 = cv2.resize(cv2.imread("../data/generate/template/white.jpg"), (226, 70))
        self.bg5 = cv2.bitwise_not(self.bg5)
        self.bg6 = cv2.resize(cv2.imread("../data/generate/template/black.jpg"), (226, 70))
        self.bg7 = cv2.resize(cv2.imread("../data/generate/template/yellow_s.jpg"), (226, 105))
        self.bg7 = cv2.bitwise_not(self.bg7)
        self.smu = cv2.imread("../data/generate/template/smu2.jpg")
        self.env_path = []
        for parent, parent_folder, filenames in os.walk(env_dir):
            for filename in filenames:
                path = parent + "/" + filename
                self.env_path.append(path)
    #生成图片
    def draw(self, val):
        offset = 2
        self.img[0:70, offset+8:offset+8+23] = GenCh(self.fontC, val[0])
        self.img[0:70, offset+8+23+6:offset+8+23+6+23] = GenEn(self.fontE, val[1])
        for i in range(5):
            base = offset + 8 + 23 + 6 + 23 + 17 + i * 23 + i * 6
            self.img[0:70, base: base + 23] = GenEn(self.fontE, val[i + 2])
        return self.img

    def draw_s(self, val):
        for i in range(5):
            base = 10 + i * 43
            self.img_s[33:105, base: base+35] = GenEn_s1(self.fontE, val[i+2])
        self.img_s[0:45, 55:100] = GenCh_s(self.fontC, val[0])
        self.img_s[0:45, 126:171] = GenEn_s(self.fontE, val[1])
        return self.img_s

    def draw_g(self, val):
        self.img[0:70, 9:30] = GenCh(self.fontC, val[0], size=(21, 70))
        self.img[0:70, 35:55] = GenEn(self.fontE, val[1], size=(20, 70))
        for i in range(6):
            base = 55 + 17 + i * 25
            self.img[0:70, base: base + 20] = GenEn(self.fontE, val[i + 2], size=(20, 70))
        return self.img

    def generate(self, text):
        com = None
        if len(text) == 7:
            fg = self.draw(text)
            fg = cv2.bitwise_not(fg)
            color = np.random.random() * 4

            if color < 1:
                com = cv2.bitwise_or(fg, self.bg1)
            elif color < 2:
                com = cv2.add(fg, self.bg2)
                com = cv2.bitwise_not(com)
            elif color < 3:
                com = cv2.add(fg, self.bg5)
                com = cv2.bitwise_not(com)
            else:
                com = cv2.bitwise_or(fg, self.bg6)

            #borderValue = (r(255), r(255), r(255))
            com = add_affine(com,r(20) - 10, com.shape, 10)
            com = add_perspective(com, 3, (com.shape[1], com.shape[0]))
            #com = random_envirment(com, self.env_path)
            com = add_tfactor(com)
            com = add_gauss(com, 1 + r(1))
            com = add_noise(com)
        return com

    #生成两行字的车牌
    def generate_s(self, text):
        com = None
        if len(text) == 7:
            fg = self.draw_s(text)
            fg = cv2.bitwise_not(fg)
            com = cv2.add(fg, self.bg7)
            com = cv2.bitwise_not(com)

            borderValue = (r(255), r(255), r(255))
            com = add_affine(com,r(20)-10, com.shape, 10, borderValue=borderValue)
            com = add_perspective(com, 3, (com.shape[1], com.shape[0]), borderValue=borderValue)
            #com = random_envirment(com, self.env_path)
            com = add_tfactor(com)
            com = add_gauss(com, r(1))
            com = add_noise(com)
        return com

    #生成绿色车牌
    def generate_g(self, text):
        com = None
        if len(text) == 8:
            fg = self.draw_g(text)
            fg = cv2.bitwise_not(fg)            
            color = np.random.random() * 2
            if color < 1:
                com = cv2.add(fg, self.bg3)
                com = cv2.bitwise_not(com)
            else:
                com = cv2.add(fg, self.bg4)
                com = cv2.bitwise_not(com)

            borderValue = (r(255), r(255), r(255))
            com = add_affine(com, r(20)-10, com.shape, 10, borderValue=borderValue)
            com = add_perspective(com, 3, (com.shape[1], com.shape[0]), borderValue=borderValue)
            #com = random_envirment(com, self.env_path)
            com = add_tfactor(com)
            com = add_gauss(com, 1 + r(1))
            com = add_noise(com)
        return com


    #生成车牌String,存为图片；生成车牌list,存为label。其中pos,val是设定制定位置的值，其他随机，pos=-1时全部随机
    def genPlateString(self, pos, val, num=7):
        plate_str = ""
        #box = [0, 0, 0, 0, 0, 0, 0]
        box = [0 for i in range(num)]
        if (pos != -1):
            box[pos] = 1
        for unit, cpos in zip(box, range(len(box))):
            if unit == 1:
                plate_str += val
            else:
                if cpos == 0:
                    plate_str += CHARS[r(31)]
                elif cpos == 1:
                    plate_str += CHARS[41 + r(24)]
                else:
                    plate_str += CHARS[31 + r(34)]
        return plate_str

    # 将生成的车牌图片写入文件夹，对应的label写入label.txt，special=0，生成7个字的普通车牌，special=1，生成7个字的两行车牌, special=2，生成8个字的绿牌
    def genBatch(self, batchSize, outputPath, size, num=7, special=0):
        if (not os.path.exists(outputPath)):
            os.mkdir(outputPath)
        for i in range(batchSize):
            plateStr= self.genPlateString(-1, -1, num)
            #plate = list(plateStr)
            #print(i, plateStr, plate)
            flag = 0
            if special == 0:
                img = self.generate(plateStr)
            elif special == 1:
                img = self.generate_s(plateStr)
                flag = 1
            elif special == 2:
                img = self.generate_g(plateStr)
            
            img = cv2.resize(img, size)

            count = 0
            if plateStr in lpnumber_dict.keys():
                count = lpnumber_dict[plateStr] + 1
            lpnumber_dict[plateStr] = count
            lp_name = dict2[plateStr[0]]+'_'+''.join(plateStr[1:])+'_'+str(count)+'_'+str(flag)+".jpg"
            print(i+1, list(plateStr), lp_name)
            cv2.imwrite(outputPath+"/"+lp_name, img)

    def generatePlate(self, batch_size, dst_dir, special=0):
        if special == 0:
            self.genBatch(batch_size, dst_dir, (226, 70), num=7, special=0)
        elif special == 1:
            self.genBatch(batch_size, dst_dir, (226, 105), num=7, special=1)
        elif special == 2:
            self.genBatch(batch_size, dst_dir, (226, 70), num=8, special=2)
        pass

if __name__ == '__main__':
    Gen = GenPlate("../data/generate/font/platech.ttf", '../data/generate/font/platechar.ttf', '../data/generate/envirment')
    Gen.generatePlate(9326, '../data/generate/train_s_2line', 1)
