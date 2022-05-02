import copy
import os
import shutil

import cv2
import numpy as np
import torch.utils.data
from PIL import Image
from torch import Tensor, from_numpy

try:
    from utils import creatProjectFolder, readInTagsFolderAndSort, saveImage, imgXYToChannel12
except:
    from utils.utils import creatProjectFolder, readInTagsFolderAndSort, saveImage, imgXYToChannel12


# 定义一个类 实现图片切换和画圈
class ImageList(torch.utils.data.Dataset):
    def __init__(self, project, category, dataset, json_folder=None, transform=None, random_hflip=True):
        """
        :param project: Required fields, the name of the current project
        :param category: Required fields, the name of the current category
        :param dataset: Required fields, the name of the current dataset
        :param json_folder: default None, the location of the json tag file
        :param transform: default None, Image preprocessing parameters
        :param random_hflip: default None, whether to apply random rollover to the data
        """
        # 继承父类
        super(ImageList, self).__init__()

        # 初始化参数
        self.index = 0  # 图片切换时的开始计数值
        self.transform = transform  # 图片预处理参数
        self.random_hflip = random_hflip  # 数据随机翻转
        self.project = project  # 项目名称
        self.category = category  # 类别名称
        self.dataset = dataset  # 数据集名
        self.filename_list = []

        # 判断文件夹是否存在，如果不存在则创建
        self.img_folder, self.json_folder = creatProjectFolder(project=self.project, dataset=self.dataset,
                                                               category=self.category)
        if json_folder is not None:
            self.json_folder = json_folder

        # self.dict_index_list 排序后的字典索引列表
        # self.json_dict 标签字典
        self.dict_index_list, self.json_dict = readInTagsFolderAndSort(self.json_folder)
        # self.size 当前标签字典内子字典数量
        self.size = len(self.dict_index_list)
        # 如果文件夹不为空 则依次读入图片存储到json_dict的["img_value"]项
        if self.size != 0:
            for sorted_num in self.dict_index_list:
                self.json_dict[sorted_num]["img_value"] = cv2.imread(self.json_dict[sorted_num]["img_path"])
                self.filename_list.append(self.json_dict[sorted_num]["img_name"])

        print("{} files in the project folder".format(self.size))

    # 实现__getitem__方法以支持使用torch.utils.data.DataLoader加载数据集
    # 该函数用以讲该类视为可迭代对象，使之允许使用ImageList[a]调用
    def __getitem__(self, idx):
        json_single_dict = self.json_dict[self.dict_index_list[idx]]
        image_value = Image.fromarray(json_single_dict["img_value"])
        x = json_single_dict["standard_channel"]["1"]
        y = json_single_dict["standard_channel"]["2"]
        if self.transform is not None:
            image_value = self.transform(image_value)
        if self.random_hflip and float(np.random.random(1)) > 0.5:
            image_value = from_numpy(image_value.numpy()[..., ::-1].copy())
            x = -x
        return image_value, 0, Tensor([x, y])
    
    def __len__(self):
        return self.size

    def iter_img_list(self):
        for a in self.dict_index_list:
            yield self.json_dict[a]

    def get_next(self, step=1):
        if (self.index + step) < self.size:
            self.index = self.index + step
        else:
            self.index = (self.index + step) - self.size
        return self.json_dict[self.dict_index_list[self.index]]

    def get_value(self):
        return self.json_dict[self.dict_index_list[self.index]]

    def get_count(self):
        return self.size

    def get_index(self):
        return self.index

    def get_filename_list(self):
        return self.filename_list

    def save_image(self, image_value, channel_value):
        count = self.size
        json_single_dict = saveImage(img_path=self.img_folder,
                                     img_value=image_value,
                                     channel_value=channel_value,
                                     project=self.project,
                                     category=self.category,
                                     dataset=self.dataset,
                                     count=count,
                                     json_path=self.json_folder)
        self.dict_index_list.append(int(count))
        self.filename_list.append(json_single_dict["img_name"])
        json_single_dict["image_value"] = image_value
        self.json_dict[int(count)] = json_single_dict
        self.size += 1
        return json_single_dict

    # 在图片上画圆 cv2.circle(输入图片变量, (中点坐标), 半径 , (B, G, R), 线条厚度)
    @staticmethod
    def draw_circle(image_value, img_x, img_y, radius=8, RGB=(0, 255, 0), thickness=3):
        img_y = 112-(img_y-112)
        draw_image = cv2.circle(copy.copy(image_value), (int(img_x), int(img_y)), radius, RGB, thickness)
        return draw_image

    @staticmethod
    def draw_ruler(image_value, RGB=(255, 255, 255), thickness=1):
        ruler_img = copy.copy(image_value)
        image_weight = image_value.shape[1]
        image_height = image_value.shape[0]

        helf_height = int(image_height / 2)
        helf_weight = int(image_weight / 2)
        # cv2.circle(image, (image_height / 2, image_weight / 2), 5, (255, 255, 255), 2)
        cv2.line(ruler_img, (0, 0), (image_height, image_weight), RGB, thickness)
        cv2.line(ruler_img, (0, image_height), (image_weight, 0), RGB, thickness)
        cv2.line(ruler_img, (0, helf_weight), (image_weight, helf_weight), RGB, thickness)
        cv2.line(ruler_img, (helf_height, 0), (helf_height, image_height), RGB, thickness)
        return ruler_img

    def change_image(self, new_x, new_y):
        # 创建更改备份文件夹
        backup_changes_folder = os.path.join("{}_{}".format(self.project, self.dataset),
                                             "{}_backup".format(self.category)).replace(
            "\\", "/")
        if not os.path.exists(backup_changes_folder):
            os.makedirs(backup_changes_folder)
            print("successful creat {}".format(backup_changes_folder))

        # 取出原字典并拼接原始图片/标签名
        json_single_dict = self.json_dict[self.dict_index_list[self.index]]
        source_basename, a = os.path.splitext(json_single_dict["img_name"])
        source_json_filepath = os.path.join(self.json_folder, "{}.json".format(source_basename))
        source_img_filepath = json_single_dict["img_path"]

        # 更改原字典值
        new_base_name = "{}_{}_{}".format(new_x, new_y, source_basename.split("_")[2])
        json_single_dict["img_XY"]["X"] = new_x
        json_single_dict["img_XY"]["Y"] = new_y
        json_single_dict["img_name"] = "{}.jpg".format(new_base_name)
        json_single_dict["img_path"] = os.path.join(self.img_folder, "{}.jpg".format(new_base_name)).replace("\\", "/")
        new_channel_1, new_channel_2 = imgXYToChannel12(new_x, new_y, json_single_dict["img_WH"]["width"],
                                                        json_single_dict["img_WH"]["height"])
        json_single_dict["standard_channel"]["1"] = new_channel_1
        json_single_dict["standard_channel"]["2"] = new_channel_2

        # 移动旧文件
        shutil.move(source_img_filepath, os.path.join(backup_changes_folder, "{}.jpg".format(source_basename)))
        shutil.move(source_json_filepath, os.path.join(backup_changes_folder, "{}.json".format(source_basename)))

        # 组装相关变量并创建新标签
        new_json_single_dict = saveImage(img_path=self.img_folder,
                                         img_value=json_single_dict["img_value"],
                                         channel_value=[new_channel_1, new_channel_2,
                                                        json_single_dict["standard_channel"]["3"]],
                                         project=self.project,
                                         category=self.category,
                                         dataset=self.dataset,
                                         count=json_single_dict["count"],
                                         json_path=self.json_folder,
                                         img_xy=[new_x, new_y],
                                         basename=new_base_name,
                                         basic_dict={"changed": "True"})

        # 更改self.filename_list
        filename_index = self.filename_list.index("{}.jpg".format(source_basename))
        self.filename_list[filename_index] = json_single_dict["img_name"]

        return new_json_single_dict
