#coding=utf-8
import os
import sys
import cv2
import numpy as np
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize
from PIL import Image, ImageDraw, ImageFont
import random
import datetime
sys.path.append('/home/heatonli/caffe-master-201506026/python/')
import caffe


# Implement Singleton in MyClassify
class Singleton(type):
    def __init__(cls, name, bases, dict):
        super(Singleton, cls).__init__(name, bases, dict)
        cls._instance = None

    def __call__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__call__(*args, **kw)
        return cls._instance


class MyClassify():
    __metaclass__ = Singleton

    def __init__(self):
        root_dir = '/home/heatonli/caffe-master-201506026/python/age_gender/'
        self.imagefile = root_dir + 'example_image.jpg'
        mean_filename = root_dir + 'mean.binaryproto'

        proto_data = open(mean_filename, "rb").read()
        a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
        mean = caffe.io.blobproto_to_array(a)[0]

        age_net_pretrained = root_dir + 'age_net.caffemodel'
        age_net_model_file = root_dir + 'deploy_age.prototxt'

        # Generate age classifier
        self.age_net = caffe.Classifier(age_net_model_file, \
            age_net_pretrained, mean = mean, channel_swap = (2,1,0), \
                raw_scale = 255, image_dims=(256, 256))

        gender_net_pretrained = root_dir + 'gender_net.caffemodel'
        gender_net_model_file = root_dir + 'deploy_gender.prototxt'

        # Generate gender classifier
        self.gender_net = caffe.Classifier(gender_net_model_file, \
            gender_net_pretrained, mean = mean, channel_swap = (2,1,0), \
                raw_scale = 255, image_dims = (256, 256))

        self.age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', \
            '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

        self.gender_list = ['Male', 'Female']


    def Classify(self, imagefile):
        # Detect faces in the input image
        faces, number = self.detectFaces(imagefile)
        print 'face number  ', number
        # While at least one face being detected
        if faces:
            if number == 1:
                gender, age = self.predict(imagefile)
                faces_draw = self.drawFaces(faces, imagefile)
                gender = [gender]
                age = [age]
                print 'list before'
                char = self.char_list(gender, age, True)
                print 'list end'
            else:
                gender, age, char, faces_draw = self.facesPredict(faces, imagefile)

            # Save processed image into new directory
            new_dir = os.path.split(imagefile)[0] + '/' + \
                os.path.split(imagefile)[1].split('.')[0] + \
                     '_' + os.path.split(imagefile)[1].split('.')[1] + "_draw.jpg"

            faces_draw.save(new_dir)
            new_name = os.path.split(new_dir)[1]

            # Remove the original input image
            #os.remove(imagefile)

            return number, gender, age, char, new_name

        else:
            return None, None, None, None, None

    # Predict the age and gender of each face in input image
    def facesPredict(self, faces, imagefile):

        image_dir = os.path.split(imagefile)[0]
        img = Image.open(imagefile)
        img_new = img
        #img_new = img_new.resize((500,500))

        # Draw rectangles around detected faces
        draw_image = ImageDraw.Draw(img_new)

        Gender = []
        Age = []
        Char = []

        # Get the size of upload image
        size = img.size
        width = size[0]
        height = size[1]
        
        tmp = str(datetime.datetime.now()).replace(' ', '_') + 'tmp.png'
        file_name = os.path.join(image_dir, tmp)
        count = 0
        label_dir = './tmpl/source/'

        for (x1, y1, x2, y2) in faces:
            if 1: #(x2 - x1) > (width / 20) and (y2 - y1) > (height / 20):

                # Choose an appropriate area around each detected face
                x = x2 - x1
                y = y2 - y1
                x1n = x1 - (x1 > x and x/2 or x1/2)
                y1n = y1 - (y1 > y and y/2 or y1/2)
                x2n = x2 + ((width - x2) > x and x/2 or (width-x2)/2)
                y2n = y2 + ((height - y2) > y and y*2/3 or (height-y2)*2/3)

                Image.open(imagefile).crop((x1n, y1n, x2n, y2n)).save(file_name)
                gender, age = self.predict(file_name)
                Gender.append(gender)
                Age.append(age)
                os.remove(file_name)

                draw_image.rectangle((x1, y1, x2, y2), outline = (255, 0, 0))
                draw_image.rectangle((x1+1, y1+1, x2-1, y2-1), outline = (255, 0, 0))
                count += 1

                # Paste label over each face
                box = (x1, y1-25, x1+20, y1-5)
                label = Image.open(label_dir + str(count) + '.jpg')
                label = label.resize((20,20))
                img_new.paste(label, box)

        Char = self.char_list(Gender, Age, True)

        return Gender, Age, Char, img_new

    # Return the gender and age of predicted face
    def predict(self, image):
        input_image = self.load_image(image)

        prediction = self.age_net.predict([input_image])
        age = prediction[0].argmax()
        print 'predicted age: ', self.age_list[age]

        prediction = self.gender_net.predict([input_image])
        gender = prediction[0].argmax()
        print 'predicted gender: ', self.gender_list[gender]
        return gender, age

    # Load image and change the format
    def load_image(self, filename, color=True):
        img = skimage.img_as_float(skimage.io.imread(filename)).astype(np.float32)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
            if color:
                img = np.tile(img, (1, 1, 3))
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img

    # Detect the faces in uploaded image
    def detectFaces(self, imagefile):
        img = cv2.imread(imagefile)

        # Face detection model
        face_cascade = cv2.CascadeClassifier\
            ("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml")

        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        result = []
        count = 0
        
        # Get the size of input image
        image_size = Image.open(imagefile).size
        image_width = image_size[0]
        image_height = image_size[1]

        # Find the max scale of detected faces
        img_max = 0
        for (x, y, width, height) in faces:
            if width > img_max:
                img_max = width

        # Only return the faces that are not too small
        # comparing with the largest face and entire image
        for (x, y, width, height) in faces:
            if width > image_width/25 and height > image_height/25 and width > img_max/4:
                result.append((x, y, x + width, y + height))
                count += 1
            if count == 6:
                break
        return result, count

    # Draw a rectangle arount the face in the image
    # in which only face is detected
    def drawFaces(self, faces, imagefile):
        img = Image.open(imagefile)
        draw_image = ImageDraw.Draw(img)
        for (x1, y1, x2, y2) in faces:
            draw_image.rectangle((x1, y1, x2, y2), outline = (255, 0, 0))
            draw_image.rectangle((x1+1, y1+1, x2-1, y2-1), outline = (255, 0, 0))
        return img
 
    def char_list(self, Gender, Age, ran):
        male0 = [
            '一句话描述: 可爱的小宝贝; 喜欢的歌手: 难说...; 喜欢的歌曲: 两只老虎; 喜欢的电影: 少看...; 喜欢的游戏: 应该还没玩过...',
            '一句话描述: 小宝贝好漂亮; 喜欢的歌手: 难说...; 喜欢的歌曲: 拔萝卜; 喜欢的电影: 少看...; 喜欢的游戏: 应该还没玩过...',
            '一句话描述: 今天又哭了吗?; 喜欢的歌手: 难说...; 喜欢的歌曲: 白龙马; 喜欢的电影: 少看...; 喜欢的游戏: 应该还没玩过...'
        ]
        male1 = [
            '一句话描述: 你今天淘气了吗?; 喜欢的歌手: 难说...; 喜欢的歌曲: 小苹果; 喜欢的电影: 喜羊羊与灰太郎, 熊出没; 喜欢的游戏: 捉迷藏, 过家家',
            '一句话描述: 活泼可爱,人小鬼大; 喜欢的歌手: 难说...; 喜欢的歌曲: 小毛驴; 喜欢的电影: 熊出没, 喜羊羊与灰太郎; 喜欢的游戏: 过家家, 捉迷藏',
            '一句话描述: 天真活泼,可爱无邪; 喜欢的歌手: 难说...; 喜欢的歌曲: 三个和尚; 喜欢的电影: 喜羊羊与灰太郎, 熊出没; 喜欢的游戏: 过家家, 捉迷藏'
        ]
        male2 = [
            '一句话描述: 英俊潇洒的小正太; 喜欢的歌手: 邓紫棋, 杨宗纬; 喜欢的歌曲: A.N.I.Y, 王妃; 喜欢的电影: 捉妖记, 超能陆战队; 喜欢的游戏: 英雄联盟',
            '一句话描述: 小正太好漂亮; 喜欢的歌手: exo, 萧敬腾; 喜欢的歌曲: 你的世界, 王妃; 喜欢的电影: 功夫熊猫, 大圣归来; 喜欢的游戏: Dota2',
            '一句话描述: 孺子可教也; 喜欢的歌手: 周杰伦, 陈亦迅; 喜欢的歌曲: 晴天, 爱情转移; 喜欢的电影: 冰雪奇缘, 超能陆战队; 喜欢的游戏: LOL',
			'一句话描述: 好好学习,天天向上; 喜欢的歌手: 邓紫棋, 张杰; 喜欢的歌曲: 泡沫, 勿忘心安; 喜欢的电影: 功夫熊猫, 冰雪奇缘; 喜欢的游戏: 英雄联盟'
        ]
        male3 = [
            '一句话描述: 小鲜肉啊!; 喜欢的歌手: 五月天, 周杰伦; 喜欢的歌曲: 知足, 彩虹; 喜欢的电影: 煎饼侠, 大圣归来; 喜欢的游戏: 英雄联盟, 穿越火线',
            '一句话描述: 恰同学少年,风华正茂; 喜欢的歌手: 周杰伦, 陈亦迅; 喜欢的歌曲: 彩虹, 十年; 喜欢的电影: 武林外传, 捉妖记; 喜欢的游戏: Dota2, LOL',
            '一句话描述: 早晨八九点钟的太阳; 喜欢的歌手: 胡彦斌, 蔡依林; 喜欢的歌曲: 红颜, 日不落; 喜欢的电影: 煎饼侠, 大圣归来; 喜欢的游戏: 英雄联盟, 穿越火线',
			'一句话描述: 书生意气,挥斥方遒; 喜欢的歌手: 梁静茹, 林俊杰; 喜欢的歌曲: 勇气, 一千年以后; 喜欢的电影: 煎饼侠, 大圣归来; 喜欢的游戏: Dota2, 英雄联盟'
        ]
        male4 = [
            '一句话描述: 英姿飒爽,意气风发; 喜欢的歌手: Beyong, 王菲; 喜欢的歌曲: 海阔天空, 我愿意; 喜欢的电影: 大话西游, 爱情公寓; 喜欢的游戏: Dota2, WOW',
            '一句话描述: 事业旗开得胜,马到成功; 喜欢的歌手: 许巍, 梁静茹; 喜欢的歌曲: 蓝莲花, 勇气; 喜欢的电影: 谍影重重, 盗梦空间; 喜欢的游戏: WOW, 传奇',
            '一句话描述: 拼搏在人生最好的年华; 喜欢的歌手: 五月天, 周杰伦; 喜欢的歌曲: 知足, 稻香; 喜欢的电影: 虎胆龙威, 爱情公寓; 喜欢的游戏: 三国杀, 魔兽争霸',
			'一句话描述: 房贷,车贷......加油!; 喜欢的歌手: 陈亦迅, 王力宏; 喜欢的歌曲: 富士山下, 大城小爱; 喜欢的电影: 大圣归来, 终结者; 喜欢的游戏: 红色警戒, CS',
			'一句话描述: 结婚生子,其乐融融; 喜欢的歌手: 张学友, 王菲; 喜欢的歌曲: 大城小事, 红豆; 喜欢的电影: 速度与激情, 失恋33天; 喜欢的游戏: 魔兽世界, 三国杀'
        ]
        male5 = [
            '一句话描述: 大叔气质; 喜欢的歌手: 汪锋, 刘德华; 喜欢的歌曲: 春天里, 忘情水; 喜欢的电影: 肖申克的救赎, 泰坦尼克号; 喜欢的娱乐: 斗地主, 打麻将',
            '一句话描述: 成熟稳重; 喜欢的歌手: 张学友, 韩红; 喜欢的歌曲: 一路上有你, 青藏高原; 喜欢的电影: 阿甘正传, 盗梦空间; 喜欢的娱乐: 打麻将, 斗地主',
            '一句话描述: 事业小有所成; 喜欢的歌手: 汪锋, 刘德华; 喜欢的歌曲: 春天里, 忘情水; 喜欢的电影: 肖申克的救赎, 泰坦尼克号; 喜欢的娱乐: 斗地主, 打麻将',
			'一句话描述: 家庭和睦幸福; 喜欢的歌手: 张学友, 那英; 喜欢的歌曲: 一路上有你, 白天不懂夜的黑; 喜欢的电影: 阿甘正传, 盗梦空间; 喜欢的娱乐: 打麻将, 斗地主'
        ]
        male6 = [
            '一句话描述: 家庭美满,事业有成; 喜欢的歌手: 邓丽君, 蔡琴; 喜欢的歌曲: 我只在乎你, 恰似你的温柔; 喜欢的电影: 泰坦尼克号, 地道战; 喜欢的娱乐: 打麻将',
            '一句话描述: 老骥伏枥,志在千里; 喜欢的歌手: 齐秦, 罗大佑; 喜欢的歌曲: 大约在冬季, 恋曲1990; 喜欢的电影: 阿甘正传, 地道战; 喜欢的娱乐: 打麻将',
            '一句话描述: 年龄不饶人,注意身体; 喜欢的歌手: 邓丽君, 崔健; 喜欢的歌曲: 甜蜜蜜, 一无所有; 喜欢的电影: 地道战, 泰坦尼克号; 喜欢的娱乐: 打麻将'
        ]
        male7 = [
            '一句话描述: 最美不过夕阳红; 喜欢的歌手: 邓丽君, 蔡琴; 喜欢的歌曲: 甜蜜蜜, 你的眼神; 喜欢的电影: 地道战, 地雷战; 喜欢的娱乐: 打麻将',
            '一句话描述: 福如东海,寿比南山; 喜欢的歌手: 蔡琴, 邓丽君; 喜欢的歌曲: 你的眼神, 我只在乎你; 喜欢的电影: 地雷战, 地道战; 喜欢的娱乐: 打麻将',
            '一句话描述: 人老心不老; 喜欢的歌手: 邓丽君, 蔡琴; 喜欢的歌曲: 我只在乎你, 恰似你的温柔; 喜欢的电影: 地道战, 地雷战; 喜欢的娱乐: 打麻将',
        ]
        male = [male0, male1, male2, male3, male4, male5, male6, male7]

        female0 = [
            '一句话描述: 可爱的小宝贝; 喜欢的歌手: 难说...; 喜欢的歌曲: 两只老虎; 喜欢的电影: 少看...; 喜欢的游戏: 应该还没玩过...',
            '一句话描述: 小宝贝好漂亮; 喜欢的歌手: 难说...; 喜欢的歌曲: 拔萝卜; 喜欢的电影: 少看...; 喜欢的游戏: 应该还没玩过...',
            '一句话描述: 今天又哭了吗?; 喜欢的歌手: 难说...; 喜欢的歌曲: 白龙马; 喜欢的电影: 少看...; 喜欢的游戏: 应该还没玩过...'
        ]
        female1 = [
            '一句话描述: 你今天撒娇了吗?; 喜欢的歌手: 难说...; 喜欢的歌曲: 小苹果; 喜欢的电影: 喜羊羊与灰太郎; 喜欢的游戏: 捉迷藏',
            '一句话描述: 聪明伶俐,率真可爱; 喜欢的歌手: 难说...; 喜欢的歌曲: 小毛驴; 喜欢的电影: 熊出没; 喜欢的游戏: 过家家',
            '一句话描述: 伶俐乖巧,古怪精灵; 喜欢的歌手: 难说...; 喜欢的歌曲: 三个和尚; 喜欢的电影: 喜羊羊与灰太郎; 喜欢的游戏: 过家家',
        ]
        female2 = [
            '一句话描述: 小萝莉真可爱; 喜欢的歌手: 邓紫棋, 杨宗纬; 喜欢的歌曲: A.N.I.Y, 王妃; 喜欢的电影: 捉妖记, 超能陆战队',
            '一句话描述: 萌萌哒; 喜欢的歌手: exo, 萧敬腾; 喜欢的歌曲: 你的世界, 王妃; 喜欢的电影: 功夫熊猫, 大圣归来',
			'一句话描述: 好好学习,天天向上; 喜欢的歌手: 周杰伦, 陈亦迅; 喜欢的歌曲: 晴天, 爱情转移; 喜欢的电影: 冰雪奇缘, 超能陆战队'
        ]
        female3 = [
            '一句话描述: 窈窕淑女,如花似玉; 喜欢的歌手: 五月天, 周杰伦; 喜欢的歌曲: 知足, 彩虹; 喜欢的电影: 煎饼侠, 大圣归来',
            '一句话描述: 冰雪聪明,明艳动人; 喜欢的歌手: 周杰伦, 陈亦迅; 喜欢的歌曲: 彩虹, 十年; 喜欢的电影: 武林外传, 捉妖记',
			'一句话描述: 大家闺秀; 喜欢的歌手: 胡彦斌, 蔡依林; 喜欢的歌曲: 红颜, 日不落; 喜欢的电影: 煎饼侠, 大圣归来',
			'一句话描述: 小家碧玉; 喜欢的歌手: 梁静茹, 林俊杰; 喜欢的歌曲: 勇气, 一千年以后; 喜欢的电影: 煎饼侠, 大圣归来'
        ]
        female4 = [
            '一句话描述: 温柔共美丽并存; 喜欢的歌手: Beyong, 王菲; 喜欢的歌曲: 海阔天空, 我愿意; 喜欢的电影: 大话西游, 爱情公寓',
            '一句话描述: 沉鱼落雁,闭月羞花; 喜欢的歌手: 许巍, 梁静茹; 喜欢的歌曲: 蓝莲花, 勇气; 喜欢的电影: 谍影重重, 盗梦空间',
            '一句话描述: 花容月貌; 喜欢的歌手: 五月天, 周杰伦; 喜欢的歌曲: 知足, 稻香; 喜欢的电影: 虎胆龙威, 爱情公寓',
			'一句话描述: 亭亭玉立,明艳动人; 喜欢的歌手: 陈亦迅, 王力宏; 喜欢的歌曲: 富士山下, 大城小爱; 喜欢的电影: 大圣归来, 终结者',
			'一句话描述: 品貌端庄,丽质天成; 喜欢的歌手: 张学友, 王菲; 喜欢的歌曲: 大城小事, 红豆; 喜欢的电影: 速度与激情, 失恋33天'
        ]
        female5 = [
            '一句话描述: 贤妻良母; 喜欢的歌手: 汪锋, 刘德华; 喜欢的歌曲: 春天里, 忘情水; 喜欢的电影: 肖申克的救赎, 泰坦尼克号; 喜欢的娱乐: 斗地主',
            '一句话描述: 温文尔雅,品貌端庄; 喜欢的歌手: 张学友, 韩红; 喜欢的歌曲: 一路上有你, 青藏高原; 喜欢的电影: 阿甘正传, 盗梦空间; 喜欢的娱乐: 打麻将',
            '一句话描述: 成熟知性,韵味十足; 喜欢的歌手: 汪锋, 刘德华; 喜欢的歌曲: 春天里, 忘情水; 喜欢的电影: 肖申克的救赎, 泰坦尼克号; 喜欢的娱乐: 斗地主',
			'一句话描述: 贤良淑德,秀外慧中; 喜欢的歌手: 张学友, 那英; 喜欢的歌曲: 一路上有你, 白天不懂夜的黑; 喜欢的电影: 阿甘正传, 盗梦空间; 喜欢的娱乐: 打麻将'
        ]
        female6 = [
            '一句话描述: 注意保养哦; 喜欢的歌手: 邓丽君, 蔡琴; 喜欢的歌曲: 我只在乎你, 恰似你的温柔; 喜欢的电影: 泰坦尼克号; 喜欢的娱乐: 广场舞',
            '一句话描述: 不要让衰老成为你最大的敌人; 喜欢的歌手: 齐秦, 罗大佑; 喜欢的歌曲: 大约在冬季, 恋曲1990; 喜欢的电影: 阿甘正传; 喜欢的娱乐: 打麻将',
            '一句话描述: 韶华如初; 喜欢的歌手: 邓丽君, 崔健; 喜欢的歌曲: 甜蜜蜜, 一无所有; 喜欢的电影: 地道战; 喜欢的娱乐: 广场舞',
        ]
        female7 = [
            '一句话描述: 最美不过夕阳红; 喜欢的歌手: 邓丽君, 蔡琴; 喜欢的歌曲: 甜蜜蜜, 你的眼神; 喜欢的电影: 地道战; 喜欢的娱乐: 打麻将',
            '一句话描述: 韶华易逝,容颜易老; 喜欢的歌手: 蔡琴, 邓丽君; 喜欢的歌曲: 你的眼神, 我只在乎你; 喜欢的电影: 地雷战; 喜欢的娱乐: 打麻将',
            '一句话描述: 种草养花,享田园之乐; 喜欢的歌手: 邓丽君, 蔡琴; 喜欢的歌曲: 我只在乎你, 恰似你的温柔; 喜欢的电影: 地道战; 喜欢的娱乐: 打麻将',
        ]
        female = [female0, female1, female2, female3, female4, female5, female6, female7]

        Res = []
        i = 0
        if ran == True:
            for age in Age:
                if Gender[i] == 0:
                    c = random.randrange(0, len(male[age]), 1)
                    Res.append(male[age][c])
                else:
                    c = random.randrange(0, len(female[age]), 1)
                    Res.append(female[age][c])
                i += 1
        else:
            dm = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
            df = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
            n = random.randrange(0, 3, 1)
            for age in Age:
                if Gender[i] == 0:
                    c = (dm[age] + n) % len(male[age])
                    Res.append(male[age][c])
                    dm[age] += 1
                else:
                    c = (df[age] + n) % len(female[age])
                    Res.append(female[age][c])
                    df[age] += 1
                i += 1
        return Res






