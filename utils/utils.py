import datetime
import glob
import json
import os
import uuid

import cv2
from PIL import Image

try:
    from SerialObject import SerialObject
except:
    # noinspection PyUnresolvedReferences
    from utils.SerialObject import SerialObject


#  四舍五入
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


# 从-1~+1的接收机返回值转换到图片上的坐标
def channel12ToImgSize(channel_1, channel_2, imgWidth=224, imgHeight=224):
    """
    :param channel_1: Channel-1 standardized value from RC receiver
    :param channel_2: Channel-2 standardized value from RC receiver
    :param imgWidth: The width of the corresponding image
    :param imgHeight: The height of the corresponding image
    :return: x, y
    """
    # init
    channel_1 = float(channel_1)
    channel_2 = float(channel_2)
    half_width = imgWidth / 2
    half_height = imgHeight / 2

    # width
    x = half_width + (half_width * channel_1)

    # height
    y = half_height + (half_height * channel_2)

    return truncate(x), truncate(y)


# 从图片上的坐标反推到-1~+1的标准化接收机值
def imgXYToChannel12(img_x, img_y, imgWidth=224, imgHeight=224):
    # init
    img_x = int(img_x)
    img_y = int(img_y)
    half_width = imgWidth / 2
    half_height = imgHeight / 2

    # width
    if img_x > half_width:
        channel_1 = (img_x - half_width) / half_width
    elif img_x < half_width:
        channel_1 = -(1 - (img_x / half_width))
    else:
        channel_1 = 0.00

    # height
    if img_y > half_width:
        channel_2 = (img_y - half_height) / half_height
    elif img_y < half_height:
        channel_2 = -(1 - (img_y / half_height))
    else:
        channel_2 = 0.00

    # postprocess
    channel_1 = float('%.2f' % channel_1)
    channel_2 = float('%.2f' % channel_2)

    return channel_1, channel_2


# 检测路径是否存在，如果不存在则创建
def creatProjectFolder(project, dataset, category):
    images_path = os.path.join("{}_{}".format(project, dataset), category)
    labels_path = os.path.join("{}_{}".format(project, dataset), "{}_labels".format(category))
    if not os.path.exists(images_path):
        os.makedirs(images_path)
        print("{} already creat".format(images_path))
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)
        print("{} already creat".format(labels_path))
    return images_path, labels_path


# 根据图片及其路径，创建json标签
def fromImgCreatJsonLabel(img_folder, json_path=None, input_dict=None, basic_count=0):
    basic_dict = dict(img_name="", project="road_following", dataset="A", count=0, category="apex", standard_channel={
        "1": "",
        "2": "",
        "3": "",
    }, img_XY={
        "X": "",
        "Y": "",
    }, img_WH={
        "width": "",
        "height": "",
    }, creation_time="", img_path="{project}_{dataset}/{category}/{X}_{Y}_{rand}.jpg")
    if input_dict is not None:
        basic_dict.update(input_dict)

    img_list = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))
    for img_path in img_list:
        img_path = img_path
        basename = os.path.basename(img_path)
        basic_dict["img_name"] = basename
        basic_dict["count"] = int(img_list.index(img_path) + basic_count)
        items = basename.split('_')
        basic_dict["img_XY"]["X"] = int(items[0])
        basic_dict["img_XY"]["Y"] = int(items[1])
        with Image.open(img_path) as PIL_img:
            basic_dict["img_WH"]["width"] = PIL_img.width
            basic_dict["img_WH"]["height"] = PIL_img.height
            channel_1, channel_2 = imgXYToChannel12(items[0], items[1], PIL_img.width, PIL_img.height)
            basic_dict["standard_channel"]["1"] = channel_1
            basic_dict["standard_channel"]["2"] = channel_2
            basic_dict["standard_channel"]["3"] = 0.00
        basic_dict["creation_time"] = datetime.datetime.fromtimestamp(os.stat(img_path).st_ctime).strftime(
            '%Y-%m-%d-%H:%M')
        basic_dict["img_path"] = img_path.replace("\\", "/")
        if json_path is None:
            json_path = os.path.join(os.path.join(os.path.split(img_path)[0], ".."),
                                     "{}_labels".format(category)).replace("\\", "/")
        with open(os.path.join(json_path, "{}.json".format(basename.split(".")[0])), "w") as json_file:
            json.dump(basic_dict, json_file, indent=4, ensure_ascii=False)


# 读入文件夹内所有json文件并按照count值排序
def readInTagsFolderAndSort(json_path, img_path=None):
    """
    :param img_path: Image folder for additional input when calling the function from a non-standard location
    :param json_path: Folder where json tags are stored
    :return: sorted json_list
    """
    # print(os.path.exists(json_path))
    # print(glob.glob(os.path.join(json_path, '*.json')))
    if len(glob.glob(os.path.join(json_path, '*.json'))) != 0:
        json_dict = {}
        for json_file_path in glob.glob(os.path.join(json_path, '*.json')):
            with open(json_file_path.replace("\\", "/"), "r") as json_file:
                input_dict = json.loads("".join(json_file.readlines()))
                if img_path is None:
                    if os.path.isfile(input_dict["img_path"]):
                        json_dict[input_dict["count"]] = input_dict
                else:
                    if os.path.isfile(os.path.join(img_path, input_dict["img_name"])):
                        input_dict["img_path"] = os.path.join(img_path, input_dict["img_name"]).replace("\\", "/")
                        json_dict[input_dict["count"]] = input_dict
        key_list = list(json_dict.keys())
        # return json_dict.keys(), json_dict
        return sorted(key_list), json_dict
    else:
        print("{} is empty".format(json_path))
        return [], {}


# Prominent Arduino map function
def _map(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


# 发送Y并获取通道标准值
def getChannelValue(mySerial, arduino_standardization=True, standardization=True, max_value=2000, min_value=1000, CH1_bias=0, CH2_bias=0, CH3_bias=0):
    if arduino_standardization:
        mySerial.sendData(data="Y")
        return mySerial.getData(split_str="#")
    else:
        mySerial.sendData(data="L")
        if standardization:
            data = mySerial.getData(split_str="#")
            for a in range(len(data)):
                value = int(data[a])
                if a == 0:
                    value -= CH1_bias
                elif a == 1:
                    value -= CH2_bias
                elif a == 2:
                    value -= CH3_bias
                else:
                    return ValueError
                if value > 2000:
                    value = 2000
                elif value < 1000:
                    value = 1000
                elif abs(value - 1500) < 50:
                    value = 1500
                data.append(_map(value, min_value, max_value, -100, 100) / 100)
            return data
        else:
            return mySerial.getData(split_str="#")


def bgr8_to_jpeg(value, quality=75):
    return bytes(cv2.imencode('.jpg', value)[1])


# 存储图片及标签
def saveImage(img_path, img_value, channel_value, project, category, dataset, count=0, json_path=None, basic_dict=None,
              img_xy=None, basename=None):
    """
    :param basic_dict:
    :param basename:
    :param img_xy:
    :param count: picture number
    :param img_path: the folder where the image will save to
    :param img_value: the numpy array of image data
    :param channel_value: a list, [channel_1, channel_2, channel_3], the standardized receiver value
    :param json_path: the folder where the json label will save to. If None, folders will be
    created automatically based on project, category and dataset
    :param project: the project name
    :param category: the category name
    :param dataset: the dataset name
    :return: json_dict contains data about the image

    This function should run in a loop, accumulating the count value successively to ensure that the label file is read back correctly
    """
    # 检查标签存储文件夹
    if json_path is None:
        images_path, json_path = creatProjectFolder(project, category, dataset)
        # print(json_path)

    # 转换channel_value的str到float
    for a in range(len(channel_value)):
        channel_value[a] = float(channel_value[a])

    # 取出图像大小
    # pil_img = Image.fromarray(img_value)
    # img_width = pil_img.width
    # img_height = pil_img.height
    img_width = img_value.shape[1]
    img_height = img_value.shape[0]

    # 从接收机值转换到图片坐标
    if img_xy is None:
        img_x, img_y = channel12ToImgSize(channel_value[0], channel_value[1], img_width, img_height)
    else:
        img_x = img_xy[0]
        img_y = img_xy[1]

    # 创建名称并合成路径
    if basename is None:
        basename = "{}_{}_{}".format(int(img_x), int(img_y), uuid.uuid1())
    img_filename = "{}.jpg".format(basename)
    json_filename = "{}.json".format(basename)
    img_path = os.path.join(img_path, img_filename).replace("\\", "/")
    json_path = os.path.join(json_path, json_filename).replace("\\", "/")

    # 保存图片
    # pil_img.save(img_path)
    cv2.imwrite(img_path, img_value)

    # 创建字典并写json文件
    json_dict = dict(img_name=img_filename,
                     project=project,
                     dataset=dataset,
                     count=count,
                     category=category,
                     standard_channel={
                         "1": channel_value[0],
                         "2": channel_value[1],
                         "3": channel_value[2],
                     },
                     img_XY={
                         "X": img_x,
                         "Y": img_y,
                     },
                     img_WH={
                         "width": img_width,
                         "height": img_height,
                     },
                     creation_time=datetime.datetime.fromtimestamp(os.stat(img_path).st_ctime).strftime(
                         '%Y-%m-%d-%H:%M'),
                     img_path=img_path
                     )

    if basic_dict is not None:
        json_dict.update(basic_dict)

    with open(json_path, "w") as json_file:
        json.dump(json_dict, json_file, indent=4, ensure_ascii=False)

    return json_dict


# 拆分图片路径 转换成project, dataset, category 和 img_name
def img_path_to_msg(img_path):
    path_l = img_path.replace("\\", "/").split("/")
    # ***/<project>_<dataset>/<category>/<img_name>
    # return img_name, project, category, dataset
    return path_l[-1], path_l[-3].split("_")[0], path_l[-2], path_l[-3].split("_")[1]


if __name__ == '__main__':
    # var init
    project = "testProject"
    DATASET = ['A', 'B']
    dataset = DATASET[0]
    category = "apex"

    # serial init
    # mySerial = SerialObject(portNo="COM6", baudRate=9600)
    # time.sleep(2)

    # creat save folder tree
    images_path, json_path = creatProjectFolder(project, dataset, category)

    # Create label files for old datasets
    # fromImgCreatJsonLabel("testProject_A/apex/", "testProject_A/apex_labels/", basic_count=254)

    # 重新读回标签数据并根据count排序
    index_list, json_dict = readInTagsFolderAndSort("testProject_A/apex_labels/")
    # 由于字典创建会乱序 故先获取一个排序后的key值列表，再从列表中依次调用
    # key_list = sorted(json_dict.keys())
    # for key_value in sorted(json_dict.keys()):
    #     print(json_dict[key_value])
    print(index_list)
    print(type(index_list))

    # 根据已有参数存储图片
    # img_value = cv2.imread("testProject_A/apex/1_87_5b41d0d6-f233-11e9-a00b-00044be5ef56.jpg")
    # # channel_value = getChannelValue(mySerial)
    # channel_value = ["-0.99", "-0.22", "1.00"]
    # saveImage(img_path=images_path, img_value=img_value, channel_value=channel_value,
    #           json_path=json_path, category=category, dataset=dataset, project=project)
