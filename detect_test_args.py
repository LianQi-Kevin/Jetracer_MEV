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
    project = "0421school"
    dataset = "A"
    category = "race_train"

    # 定义多个模型
    model_0 = {
        "model_type": "torch",
        "model_path": "0421school_A_fan-test_30_0421.pth",
        "model_name": "RESNET_18",
        "output_dim": 2
    }
    model_2 = {
        "model_type": "torch",
        "model_path": "0422_Ep50_race_train.pth",
        "model_name": "RESNET_18",
        "output_dim": 2
    }
    model_1 = {
        "model_type": "TRT",
        "model_path": "0421school_A_fan-test_30_0421_TRT.pth",
        "output_dim": 2,
        "fp16": True
    }
    main([model_0, model_1, model_2])
