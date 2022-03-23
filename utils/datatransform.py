"""
@ Author: wyfffffei
@ Project: Expressway-Detect-Yolov5
@ File: datatransform.py
@ Time: 2022/3/17 18:39
@ Function: change the annotation format
"""
import os
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
    # ano_file_path = "D:/projects/datasets/panda/PANDA_Crowd/head_bbox_annotations.json"
    # ano_file_path = "D:/projects/datasets/panda/Panda-Image/PANDA_IMAGE/image_annos/person_bbox_train.json"
    ano_file_path = "D:/projects/datasets/panda/Panda-Image/PANDA_IMAGE/image_annos/vehicle_bbox_train.json"
    save_path = "D:/projects/datasets/EXP_Detect/static/annotations/"

    # names = ['person', 'car', 'fire', 'rubbish']  # class names
    cla_index = ('0', '1', '2', '3')
    classes = {"people": ("person", "people", "person head"),
               "car": ("large car", "midsize car", "small car", "electric car", "vehicles", "motorcycle")}

    pics = set()
    with open(ano_file_path) as data:
        ano = json.load(data)
        for name, item in ano.items():
            pics.add(name)
            obj_list = item["objects list"]
            saved_ano = open(save_path + name.split("/")[1][:-4] + '.txt', 'a+')

            num_ano = 0
            # 逐行写入 YOLO 格式标记框信息
            for obj in tqdm(obj_list):
                # 格式参考 http://www.panda-dataset.com/Download.html
                if obj["category"] in classes["people"]:
                    rect = obj["rects"]["full body"] if obj["category"] == "person" else obj["rect"]
                    cla = cla_index[0]
                elif obj["category"] in classes["car"]:
                    rect = obj["rect"]
                    cla = cla_index[1]
                else:
                    continue
                x_center = str((rect['tl']['x'] + rect['br']['x']) / 2)
                y_center = str((rect['tl']['y'] + rect['br']['y']) / 2)
                width = str(np.abs(rect['tl']['x'] - rect['br']['x']))
                height = str(np.abs(rect['tl']['y'] - rect['br']['y']))

                saved_ano.writelines(cla + ' ' + x_center + ' ' + y_center + ' ' + width + ' ' + height + "\n")
                num_ano += 1

            print("{} -> length: {}".format(str(name), str(num_ano)))
            saved_ano.close()
    return pics


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


def data_migration(pics):
    """
    :function: copy train pictures (windows script)
    :param pics: picture name set
    """
    parent = "D:\\projects\\datasets\\panda\\Panda-Image\\PANDA_IMAGE\\image_train\\"
    copy_path = "D:\\projects\\datasets\\EXP_Detect\\static\\train\\"
    for pic in pics:
        pic = pic.replace('/', '\\')
        os.system(f"copy {parent+pic} {copy_path}")


def extract_pictures_from_video(size):
    """
    :function: extract pictures from each video (windows script)
    :parameter size: (0, 1) => (0, '3s/f', [00:00, 00:03, 00:06, ...]), (1, '30s/f', [00:00, 00:30, 01:00, ...])
    :required: ffmpeg -> https://gist.github.com/steven2358/ba153c642fe2bb1e47485962df07c730
    """
    if size not in (0, 1):
        return

    v_parent = "D:\\Document\\res_DeepLearning\\Video1\\"
    # v_parent = "D:\\Document\\res_DeepLearning\\Video2\\"
    val_save_path = "D:\\projects\\datasets\\EXP_Detect\\static\\val\\"
    test_save_path = "D:\\projects\\datasets\\EXP_Detect\\static\\test\\"

    for file in os.listdir(v_parent):
        size = 0  # 孤儿视频默认都小
        if file[-4:] != '.mp4':
            file = file + '\\' + os.listdir(v_parent+file)[0]  # 嵌套视频默认都大，取第一个视频
            size = 1

        for time in range(10):
            if time % 2 == 0:
                if size == 0:
                    os.system("ffmpeg -i '{}' -ss 00:00:{}.000 -vframes 1 '{}{}_00{}.jpg' -hide_banner".format(
                        v_parent + file.split()[1], time * 3, val_save_path, file.split('\\')[-1][:-4], time
                    ))
                elif size == 1:
                    os.system("ffmpeg -i '{}' -ss 00:0{}:00.000 -vframes 1 '{}{}_00{}.jpg' -hide_banner".format(
                        v_parent+file, int(time/2), val_save_path, file.split('\\')[-1][:-4], time
                    ))
            else:
                if size == 0:
                    os.system("ffmpeg -i '{}' -ss 00:00:{}.000 -vframes 1 '{}{}_00{}.jpg' -hide_banner".format(
                        v_parent+file, time * 3, test_save_path, file.split('\\')[-1][:-4], time
                    ))
                elif size == 1:
                    os.system("ffmpeg -i '{}' -ss 00:0{}:30.000 -vframes 1 '{}{}_00{}.jpg' -hide_banner".format(
                        v_parent + file, time // 2, test_save_path, file.split('\\')[-1][:-4], time
                    ))


def data_split(data):
    # train : val : test  <==>  all_panda + 20% highway_pic : 40% highway_pic : 40% highway_pic
    # TODO: split the train data to train / val (files / annotations)
    pass


def test():
    v_parent = "D:\\Document\\res_DeepLearning\\Video1\\"
    val_save_path = "D:\\projects\\datasets\\EXP_Detect\\static\\val\\"
    test_save_path = "D:\\projects\\datasets\\EXP_Detect\\static\\test\\"
    for file in os.listdir(v_parent):
        size = 0  # 孤儿视频默认都小
        if file[-4:] != '.mp4':
            file = file + '\\' + os.listdir(v_parent + file)[0]  # 嵌套视频默认都大，取第一个视频
            size = 1
            print()

        for time in range(10):
            if time % 2 == 0:
                if size == 0:
                    print("ffmpeg -i '{}' -ss 00:00:{}.000 -vframes 1 '{}{}_00{}.jpg' -hide_banner".format(
                        v_parent + file, time * 3, val_save_path, file.split('\\')[-1][:-4], time
                    ))
                elif size == 1:
                    print("ffmpeg -i '{}' -ss 00:0{}:00.000 -vframes 1 '{}{}_00{}.jpg' -hide_banner".format(
                        v_parent + file, int(time / 2), val_save_path, file.split('\\')[-1][:-4], time
                    ))
            else:
                if size == 0:
                    print("ffmpeg -i '{}' -ss 00:00:{}.000 -vframes 1 '{}{}_00{}.jpg' -hide_banner".format(
                        v_parent + file, time * 3, test_save_path, file.split('\\')[-1][:-4], time
                    ))
                elif size == 1:
                    print("ffmpeg -i '{}' -ss 00:0{}:30.000 -vframes 1 '{}{}_00{}.jpg' -hide_banner".format(
                        v_parent + file, time // 2, test_save_path, file.split('\\')[-1][:-4], time
                    ))


def main():
    test()

    # show_category()
    # train_pics = panda2YOLO()
    # print(train_pics)
    # data_migration(train_pics)
    # extract_pictures_from_video(0)
    # data_split(train_pics)


if __name__ == "__main__":
    main()

