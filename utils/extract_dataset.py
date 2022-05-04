import os
import shutil
from utils import readInTagsFolderAndSort
import json


if __name__ == '__main__':
    input_json_path = "../0422school_A/race_train_labels"
    input_img_path = "../0422school_A/race_train/"

    output_1 = {
        "max": 4000,
        "min": 2000,
        "json_path": "../0422school_A/choicest_labels",
        "img_path": "../0422school_A/choicest"
    }

    output_2 = {
        "max": 5100,
        "min": 4500,
        "json_path": "../0422school_A/choicest_val_labels",
        "img_path": "../0422school_A/choicest_val"
    }

    sorted_json_index, json_dict = readInTagsFolderAndSort(json_path=input_json_path,
                                                           img_path=input_img_path)

    for a in range(len(sorted_json_index)):
        single_dict = json_dict[sorted_json_index[a]]
        basename = os.path.basename(os.path.splitext(single_dict["img_path"])[0])
        json_path = os.path.join(input_json_path, "{}.json".format(basename))
        if a in range(output_1["min"], output_1["max"]):
            if not os.path.exists(output_1["img_path"]):
                os.makedirs(output_1["img_path"])
            if not os.path.exists(output_1["json_path"]):
                os.makedirs(output_1["json_path"])
            shutil.copy(single_dict["img_path"], output_1["img_path"])
            shutil.copy(json_path, output_1["json_path"])
        if a in range(output_2["min"], output_2["max"]):
            if not os.path.exists(output_2["img_path"]):
                os.makedirs(output_2["img_path"])
            if not os.path.exists(output_2["json_path"]):
                os.makedirs(output_2["json_path"])
            shutil.copy(single_dict["img_path"], output_2["img_path"])
            shutil.copy(json_path, output_2["json_path"])
        print(a)

