import os

# 定义模型信息的基础字典
model_basic = {
    "model_type": "",
    "model_path": "",
    "model_name": "RESNET_18",
    "output_dim": 2,
    "fp16": False
}


def main(model_list):
    for model_msg in iter(model_list):
        model_data = model_basic.copy()
        model_data.update(model_msg)
        # print(model_data)
        os.system("python3 ./utils/multi_model_inference.py "
                  "--project {} "
                  "--dataset {} "
                  "--category {} "
                  "--model_type {} "
                  "--model_path {} "
                  "--model_name {} "
                  "--output_dim {} "
                  "--half_precision {}"
                  .format(project, dataset, category, model_data['model_type'], model_data["model_path"],
                          model_data["model_name"], model_data["output_dim"], model_data["fp16"]))


if __name__ == '__main__':
    # 定义数据集信息
    project = "0422school"
    dataset = "A"
    category = "choicest_val"

    # 定义多个模型
    # model_0 = {
    #     "model_type": "torch",
    #     "model_path": "0422school_A/choicest_batch8_epoch1_RESNET_18.pth",
    #     "model_name": "RESNET_18",
    #     "output_dim": 2
    # }

    model_0 = {
        "model_type": "trt",
        "model_path": "0422school_A/choicest_batch8_epoch5_RESNET_18_trt.pth",
        "fp16": True,
        "output_dim": 2
    }
    model_1 = {
        "model_type": "trt",
        "model_path": "0422school_A/choicest_batch8_epoch10_RESNET_18_trt.pth",
        "fp16": True,
        "output_dim": 2
    }
    model_2 = {
        "model_type": "trt",
        "model_path": "0422school_A/choicest_batch8_epoch15_RESNET_18_trt.pth",
        "fp16": True,
        "output_dim": 2
    }
    model_3 = {
        "model_type": "trt",
        "model_path": "0422school_A/choicest_batch8_epoch20_RESNET_18_trt.pth",
        "fp16": True,
        "output_dim": 2
    }
    model_4 = {
        "model_type": "trt",
        "model_path": "0422school_A/choicest_batch8_epoch30_RESNET_18_trt.pth",
        "fp16": True,
        "output_dim": 2
    }
    # main([model_0, model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9])
    main([model_0, model_1, model_2, model_3, model_4])
