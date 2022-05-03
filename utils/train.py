import time

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image as Image

try:
    from ImageList import ImageList
except:
    from utils.ImageList import ImageList


# 设置模型
def set_model(model_name, output_dim):
    # 定义每个类有几个输出值
    #     output_dim = 2 * len(dataset.categories)
    # 定义模型
    if model_name == "RESNET_18":
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, output_dim)
    elif model_name == "ALEXNET":
        model = torchvision.models.alexnet(pretrained=True)
        model.classifier[-1] = torch.nn.Linear(4096, output_dim)
    elif model_name == "SQUEEZENET":
        model = torchvision.models.squeezenet1_1(pretrained=True)
        model.classifier[1] = torch.nn.Conv2d(512, output_dim, kernel_size=1)
        model.num_classes = len(dataset.categories)
    elif model_name == "RESNET_34":
        model = torchvision.models.resnet34(pretrained=True)
        model.fc = torch.nn.Linear(512, output_dim)
    elif model_name == "DENSENET_121":
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = torch.nn.Linear(model.num_features, output_dim)
    else:
        print("Please check your model name, and check our model support list.")
        exit()

    return model


# 保存模型
def save_model(model, model_filename):
    torch.save(model.state_dict(), model_filename)


# 加载模型
def load_model(model, model_filename):
    model.load_state_dict(torch.load(model_filename))
    return model


def preprocess(image):
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    device = torch.device('cuda')
    image = Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def train_eval_model(model, train_eval, BATCH_SIZE, optimizer, epochs, device):
    try:
        # 加载数据集
        train_loader = torch.utils.data.DataLoader(
            image_list,
            batch_size=BATCH_SIZE,
            shuffle=True  # 每次加载数据时打乱数据顺序 推断具有抗过拟合的功能
        )

        time.sleep(1)

        # 判断训练还是验证
        if train_eval == "train":
            model = model.train()
        elif train_eval == "eval":
            model = model.eval()

        for a in range(epochs):
            i = 0
            sum_loss = 0.0
            # error_count = 0.0

            for images, category_idx, xy in iter(train_loader):
                # 将数据送到GPU
                images = images.to(device)
                xy = xy.to(device)

                if train_eval == "train":
                    # 优化器梯度置0
                    # https://flyswiftai.com/li-jieoptimizerzerograd-lossbackward-opt/
                    optimizer.zero_grad()

                # 获取模型输出
                outputs = model(images)

                # 计算loss
                loss = 0.0
                for batch_idx, cat_idx in enumerate(list(category_idx.flatten())):
                    # torch.mean() 返回Tensor内所有参数的平均值
                    loss += torch.mean((outputs[batch_idx][2 * cat_idx:2 * cat_idx + 2] - xy[batch_idx]) ** 2)
                loss /= len(category_idx)

                if train_eval == "train":
                    # 反向传播
                    loss.backward()
                    # 优化器调整参数
                    optimizer.step()

                # 计算进度和loss
                count = len(category_idx.flatten())
                i += count
                sum_loss += float(loss)
                # ------------------------------------------------------------------
                # 进度
                progress = i / len(dataset)
                # 归一化的loss值
                normalized_loss = sum_loss / i

                print("epoch: {}, progress: {}, normalized_loss: {}".format(a + 1, progress, normalized_loss), end="\n")
    except:
        pass
    model = model.eval()
    return model


if __name__ == '__main__':
    # torch自带的图像处理
    # transforms.Compose  用来拼接多个变换方法
    TRANSFORMS = transforms.Compose([
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),  # 改变图片的对比度 亮度 饱和度 色调
        transforms.Resize((224, 224)),  # 调整图片尺寸到[224, 224]
        transforms.ToTensor(),  # 转换到Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # 用平均值和标准差对浮点张量图像进行标准化 list内三个值对图片的三个通道使用不同值
    ])

    project = "../testProject"
    dataset = "C"
    category = "apex"

    image_list = ImageList(project, category, dataset, transform=TRANSFORMS, random_hflip=True)

    # ----------------------------------------------------------------------------
    # 设定GPU版torch
    device = torch.device('cuda')
    # 设置模型   RESNET 18
    model = set_model("RESNET_18", 2)
    # 将模型传到GPU
    model = model.to(device)
    # -----------------------------------------------------------------------------
    # 模型训练相关
    BATCH_SIZE = 8

    # 优化器
    # https://zhuanlan.zhihu.com/p/32338983
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    # 训练轮数
    epochs = 50

    # 训练/验证
    train_eval = "train"
    # train_eval = "eval"

    time_start = time.time()

    # 训练
    model = train_eval_model(model, train_eval, BATCH_SIZE, optimizer, epochs, device)

    # 保存
    save_model(model, "T_road_following_model.pth")

    time_end = time.time()

    print('time cost', time_end - time_start, 's')
