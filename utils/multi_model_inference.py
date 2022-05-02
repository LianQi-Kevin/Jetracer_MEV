import argparse
import copy
import glob
import json
import os
import time

import cv2
# import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image as Image
from torch2trt import TRTModule
from colormap import rgb2hex

try:
    from utils.utils import creatProjectFolder, channel12ToImgSize
    from utils.hix_color import ncolors
except ModuleNotFoundError:
    from utils import creatProjectFolder, channel12ToImgSize
    from hix_color import ncolors


class DecodeMultiInferResult:
    def __init__(self, json_path):
        self.json_path = json_path
        with open(json_path, "r") as json_f:
            self.source_dict = json.loads("".join(json_f.readlines()))
            json_f.close()

        # 索引列表排序
        self.sorted_key_list = list(self.source_dict.keys())
        for a in range(len(self.sorted_key_list)):
            self.sorted_key_list[a] = int(self.sorted_key_list[a])
        self.sorted_key_list = sorted(self.sorted_key_list)
        for a in range(len(self.sorted_key_list)):
            self.sorted_key_list[a] = str(self.sorted_key_list[a])

        # 初始化参数
        self.model_name_list = []
        self.index = 0
        self.size = len(self.sorted_key_list)
        self.img_path_list = []

        # 遍历字典 确定模型数量并读入图片
        for item in self.sorted_key_list:
            single_dict = self.source_dict[item]
            if os.path.isfile(single_dict["img_path"]):
                self.img_path_list.append(single_dict["img_path"])
                single_dict["img_value"] = cv2.imread(single_dict["img_path"], cv2.IMREAD_COLOR)
                for model_name in list(single_dict["model_output"].values()):
                    self.model_name_list.append(model_name["model_path"])
        self.model_name_list = list(set(self.model_name_list))

        self.color_dict = dict(source_value={"RGB": [0, 0, 0], "HEX": rgb2hex(0, 0, 0, normalised=False)})
        color_list = ncolors(len(self.model_name_list))
        for a in range(len(self.model_name_list)):
            self.color_dict[self.model_name_list[a]] = dict(RGB=color_list[a],
                                                            HEX=rgb2hex(color_list[a][0], color_list[a][1],
                                                                        color_list[a][2], normalised=False))

        print("{} models in {}".format(len(self.model_name_list), self.json_path))
        print("successful load {} images".format(len(self.img_path_list)))

    def __len__(self):
        return self.size

    def get_img_next(self, step=1):
        if (self.index + step) < self.size:
            self.index = self.index + step
        else:
            self.index = (self.index + step) - self.size
        single_dict = self.source_dict[self.sorted_key_list[self.index]]
        return self.__dict_draw_circle(single_dict), single_dict

    def get_img_value(self):
        single_dict = self.source_dict[self.sorted_key_list[self.index]]
        return self.__dict_draw_circle(single_dict), single_dict

    def __dict_draw_circle(self, single_dict):
        img_x, img_y = channel12ToImgSize(single_dict["source_value"]["1"], single_dict["source_value"]["2"])
        img = self.draw_circle(image_value=single_dict["img_value"], img_x=img_x, img_y=img_y, radius=6,
                               BGR=tuple(self.__RGB2BGR(self.color_dict["source_value"]["RGB"])), thickness=3)
        for inferred_dict in list(single_dict["model_output"].values()):
            img_x, img_y = channel12ToImgSize(inferred_dict["detected_value"]["1"], inferred_dict["detected_value"]["2"])
            img = self.draw_circle(image_value=img, img_x=img_x, img_y=img_y, radius=6,
                                   BGR=tuple(self.__RGB2BGR(RGB=self.color_dict[inferred_dict["model_path"]]["RGB"])), thickness=3)
        return img

    def get_count(self):
        return self.size

    def get_index(self):
        return self.index

    def get_img_path_list(self):
        return self.img_path_list

    def get_img_name_list(self):
        img_name_list = []
        for img_path in self.img_path_list:
            img_name_list.append(os.path.basename(img_path))
        return img_name_list

    def get_model_color(self, model_name="source_value"):
        return self.color_dict[model_name]["RGB"], self.color_dict[model_name]["HEX"]

    @staticmethod
    def __RGB2BGR(RGB):
        BGR = copy.copy(RGB)
        if type(RGB) is not list:
            BGR = list(BGR)
        BGR.reverse()
        return BGR

    # 在图片上画圆 cv2.circle(输入图片变量, (中点坐标), 半径 , (B, G, R), 线条厚度)
    @staticmethod
    def draw_circle(image_value, img_x, img_y, radius=8, BGR=(0, 255, 0), thickness=3):
        half_height = (image_value.shape[0]) / 2
        img_y = half_height * 2 - img_y
        draw_image = cv2.circle(copy.copy(image_value), (int(img_x), int(img_y)), radius, BGR, thickness)
        return draw_image

    @staticmethod
    def draw_ruler(image_value, RGB=(255, 255, 255), thickness=1):
        ruler_img = copy.copy(image_value)
        image_weight = image_value.shape[1]
        image_height = image_value.shape[0]

        half_height = int(image_height / 2)
        half_weight = int(image_weight / 2)
        # cv2.circle(image, (image_height / 2, image_weight / 2), 5, (255, 255, 255), 2)
        cv2.line(ruler_img, (0, 0), (image_height, image_weight), RGB, thickness)
        cv2.line(ruler_img, (0, image_height), (image_weight, 0), RGB, thickness)
        cv2.line(ruler_img, (0, half_weight), (image_weight, half_weight), RGB, thickness)
        cv2.line(ruler_img, (half_height, 0), (half_height, image_height), RGB, thickness)
        return ruler_img


class MultiModelDataset:
    def __init__(self, project, category, dataset):
        # 初始化参数
        self.project = project  # 项目名称
        self.category = category  # 类别名称
        self.dataset = dataset  # 数据集名
        self.img_folder, self.json_folder = creatProjectFolder(project=self.project, dataset=self.dataset,
                                                               category=self.category)
        self.output_dict = {}
        self.img_value_dict = {}
        self.local_json_path = os.path.join("{}_{}".format(self.project, self.dataset),
                                            "{}_multi-model_inference.json".format(self.category)).replace("\\", "/")

        # 依次读取json文件的内容并存入self.output_dict
        json_path_list = glob.glob(os.path.join(self.json_folder, '*.json'))
        if len(json_path_list) != 0:
            for file_path in json_path_list:
                with open(file_path, "r") as json_file:
                    input_dict = json.loads("".join(json_file.readlines()))
                    if os.path.isfile(input_dict["img_path"]):
                        count = str(input_dict["count"])
                        self.img_value_dict[count] = {}
                        self.img_value_dict[count]["img_value"] = cv2.imread(input_dict["img_path"], cv2.IMREAD_COLOR)
                        self.output_dict[count] = {}
                        self.output_dict[count]["img_path"] = input_dict["img_path"]
                        self.output_dict[count]["source_value"] = {}
                        self.output_dict[count]["source_value"]["1"] = input_dict["standard_channel"]["1"]
                        self.output_dict[count]["source_value"]["2"] = input_dict["standard_channel"]["2"]
                        self.output_dict[count]["model_output"] = {}
        else:
            print("MultiModelDataset; {} is empty, Please check your input.".format(self.json_folder))
            exit()

        self.dict_index_list = sorted(list(self.output_dict.keys()))
        print("There is {} file in the project folder".format(len(self.dict_index_list)))

        if os.path.isfile(self.local_json_path):
            with open(self.local_json_path, "r") as local_json:
                basic_dict = json.loads("".join(local_json.readlines()))
                self.output_dict.update(basic_dict)

    def __getitem__(self, item):
        """
        :param item: the item in self.dict_index_list
        :return: item's count, img_value, source_value
        """
        count = self.dict_index_list[item]
        source_value = self.output_dict[count]["source_value"]
        img_value = self.img_value_dict[count]["img_value"]
        return count, img_value, source_value

    def add_new_model(self, count, model_path, model_type, detected_value):
        """
        添加新的model_inference_value
        :param model_path:
        :param count: the img count
        :param model_type: torch or TRT
        :param detected_value: the model output
        :return: self.output_dict[count]
        """
        model_value = dict(model_type=model_type, category=self.category, model_path=model_path,
                           detected_value={"1": float(detected_value[0]), "2": float(detected_value[1])},
                           creation_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        self.output_dict[count]["model_output"][model_path] = {}
        self.output_dict[count]["model_output"][model_path].update(model_value)
        return self.output_dict[count]

    def refresh_local(self):
        """
        refresh local json file
        根据self.output_dict重写self.local_json_file
        """
        with open(self.local_json_path, "w") as local_json:
            json.dump(self.output_dict, local_json, indent=4, ensure_ascii=False)
            local_json.close()
        # json.dump将字典写入本地时 会将int的key值转换为str 故解析时需要重新转换到int
        print("Refresh local file, path: {}".format(self.local_json_path))


def preprocess(img_value):
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    device = torch.device('cuda')
    image = Image.fromarray(img_value)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


class ModelDetect:
    def __init__(self, model_type, model_path, model_name="RESNET_18", output_dim=2, fp16=True,
                 device=torch.device('cuda')):
        self.model_path = model_path
        self.model_type = model_type
        self.model_name = model_name
        self.output_dim = output_dim
        self.fp16 = fp16

        if not os.path.isfile(self.model_path):
            print("{} not found, Please check the root".format(self.model_path))

        if self.model_type == "torch":
            self.set_model()
            self.model.to(device=device)
        elif self.model_type == "TRT":
            self.model = TRTModule()
        self.model.load_state_dict(torch.load(self.model_path))

        print("Successful load model {}".format(self.model_path))

    def __str__(self):
        return str("model_path: {} \n model_type: {}\n".format(self.model_path, self.model_type))

    def set_model(self):
        if self.model_name == "RESNET_18":
            self.model = torchvision.models.resnet18(pretrained=True)
            self.model.fc = torch.nn.Linear(512, self.output_dim)
        elif self.model_name == "ALEX_NET":
            self.model = torchvision.models.alexnet(pretrained=True)
            self.model.classifier[-1] = torch.nn.Linear(4096, self.output_dim)
        elif self.model_name == "SQUEEZE_NET":
            self.model = torchvision.models.squeezenet1_1(pretrained=True)
            self.model.classifier[1] = torch.nn.Conv2d(512, self.output_dim, kernel_size=1)
            self.model.num_classes = 1
        elif self.model_name == "RESNET_34":
            self.model = torchvision.models.resnet34(pretrained=True)
            self.model.fc = torch.nn.Linear(512, self.output_dim)
        elif self.model_name == "DENSENET121":
            self.model = torchvision.models.densenet121(pretrained=True)
            self.model.classifier = torch.nn.Linear(self.model.num_features, self.output_dim)
        else:
            print("Please check your model name, and check our model support list.")
            exit()

    def inference(self, img_value):
        processed_img = preprocess(img_value)
        if self.model_type == "TRT" and self.fp16 is True:
            processed_img = processed_img.half()
        output = self.model(processed_img).detach().cpu().numpy().flatten()
        return output[0], output[1]


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='multi model inference.')
    parser.add_argument('--project', type=str, default="0421school", help='project')
    parser.add_argument('--dataset', type=str, default="A", help='dataset')
    parser.add_argument('--category', type=str, default="race_train", help='category')
    parser.add_argument('--model_type', type=str, default="torch", help='model_type: torch/TRT')
    parser.add_argument('--model_path', type=str, default="0421school_A_fan-test_30_0421.pth", help='model_path')
    parser.add_argument('--model_name', type=str, default="RESNET_18", help='model_name: RESNET_18')
    parser.add_argument('--output_dim', type=int, default=2, help='output_dim')
    parser.add_argument('--half_precision', type=bool, default=True, help='fp16')
    args = parser.parse_args()

    # 导出解析器的参数
    project = args.project
    dataset = args.dataset
    category = args.category
    model_type = args.model_type
    model_path = args.model_path
    model_name = args.model_name
    output_dim = args.output_dim
    fp16 = args.half_precision

    # 基于参数初始化库
    detection = ModelDetect(model_type=model_type, model_path=model_path, model_name=model_name, output_dim=output_dim,
                            fp16=fp16)
    multi_dataset = MultiModelDataset(project=project, category=category, dataset=dataset)

    # 开始检测
    for count, img_value, source_value in iter(multi_dataset):
        output0, output1 = detection.inference(img_value)
        multi_dataset.add_new_model(count=count, model_type=model_type, model_path=model_path,
                                    detected_value=[output0, output1])
    multi_dataset.refresh_local()
