"""
@ Author: wyfffffei
@ Project: Expressway-Detect-Yolov5
@ File: show_data.py
@ Time: 2022/4/11 4:23
@ Function:  
"""
import os

ROOT_DATA_PATH = r"D:\projects\datasets\EXP_Detect"


def showImage():
    global ROOT_DATA_PATH
    image_path_list = [ROOT_DATA_PATH + r"\high_equ\images", ROOT_DATA_PATH + r"\low_equ\images"]
    # 读取图片数量
    # number of high_equ_test += 165
    for equ in image_path_list:
        print("↓  {}  ↓".format(equ))
        parent = os.listdir(equ)
        for item in parent:
            print("{} number: {}".format(item, len(os.listdir(equ + '\\' + item))))
        print()


def showLabel():
    global ROOT_DATA_PATH
    label_path_list = [ROOT_DATA_PATH + r"\high_equ\labels", ROOT_DATA_PATH + r"\low_equ\labels"]

    # 读取标记数量和种类  eg => D:\projects\datasets\EXP_Detect\low_equ\labels\train
    for equ in label_path_list:
        classes_number = {0: 0, 1: 0, 2: 0, 3: 0}
        label_number = 0

        print("↓  {}  ↓".format(equ))
        parent = os.listdir(equ)
        for item in parent:
            labels = os.listdir(equ + '\\' + item)
            for label in labels:
                if label == "classes.txt":
                    continue
                lines = open(equ + '\\' + item + '\\' + label, 'r').readlines()
                for line in lines:
                    classes_number[int(line.strip().split(" ")[0])] += 1
                label_number += len(lines)
                del lines
        print("The numbers of each instances: {}".format(label_number))
        print("The numbers of each class: ", end="\t")
        print(classes_number)
        print()


if __name__ == "__main__":
    showImage()
    showLabel()
