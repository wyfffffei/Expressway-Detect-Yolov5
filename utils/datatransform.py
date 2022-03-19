"""
@ Author: wyfffffei
@ Project: Expressway-Detect-Yolov5
@ File: datatransform.py
@ Time: 2022/3/17 18:39
@ Function: change the annotation format
"""
import json
import numpy as np
from tqdm import tqdm


def panda2YOLO():
    """
    Origin Format(panda):
        {
          "Marathon/marathon_01.jpg": {
            "image size": {
              "height": 15024,
              "width": 26908
            },
            "objects list": [
              {
                "category": "person head",
                "rect": {
                  "tl": {
                    "x": 0.21301714115, <- left
                    "y": 0.73392984635  <- top
                  },
                  "br": {
                    "x": 0.21621386995, <- right
                    "y": 0.742857099225 <- bottom
                  }
                }
              },
              ...

    To YOLO Format:
        class_index x_center y_center width height
    """
    ano_file_path = "D:/projects/datasets/panda/PANDA_Crowd/head_bbox_annotations.json"
    save_path = "D:/projects/datasets/EXP_Detect/static/annotations/"
    names = ['person', 'car', 'fire', 'rubbish']  # class names

    with open(ano_file_path) as data:
        ano = json.load(data)
        for name, item in ano.items():
            save_ano = open(save_path + name.split("/")[1][:-4] + '.txt', 'w')
            obj_list = item["objects list"]

            num_ano = 0
            # 逐行写入 YOLO 格式标记框信息
            for obj in tqdm(obj_list):
                # 格式参考 http://www.panda-dataset.com/Download.html
                if obj["category"] == "person":
                    rect = obj["rects"]["full body"]
                elif obj["category"] == "people" or obj["category"] == "person head":
                    rect = obj["rect"]
                else:
                    continue
                x_center = str((rect['tl']['x'] + rect['br']['x']) / 2)
                y_center = str((rect['tl']['y'] + rect['br']['y']) / 2)
                width = str(np.abs(rect['tl']['x'] - rect['br']['x']))
                height = str(np.abs(rect['tl']['y'] - rect['br']['y']))

                # TODO: 类别判断
                save_ano.writelines('0 ' + x_center + ' ' + y_center + ' ' + width + ' ' + height + "\n")
                num_ano += 1

            print("{} -> length: {}".format(str(name), str(num_ano)))
            save_ano.close()


def show_category():
    """
    function: test the number of category (panda)
    """
    # ano_file_path = "D:/projects/datasets/panda/PANDA_Crowd/head_bbox_annotations.json"
    # ano_file_path = "D:/projects/datasets/panda/Panda-Image/PANDA_IMAGE/image_annos/person_bbox_train.json"
    ano_file_path = "D:/projects/datasets/panda/Panda-Image/PANDA_IMAGE/image_annos/vehicle_bbox_train.json"

    with open(ano_file_path) as data:
        ano = json.load(data)
        category = set()
        for name, item in ano.items():
            obj_list = item["objects list"]
            for obj in obj_list:
                category.add(obj["category"])
    print(category)


def test():
    pass


if __name__ == "__main__":
    show_category()
    # panda2YOLO()

