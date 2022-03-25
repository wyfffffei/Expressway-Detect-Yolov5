"""
@ Author: wyfffffei
@ Project: Expressway-Detect-Yolov5
@ File: data_extract.py.py
@ Time: 2022/3/26 0:04
@ Function: extract the dataset from the video
"""
import os
from subprocess import call


def action_extract(file_path, save_path, fps):
    """
    :function: execute the `ffmpeg` command
    :parameter fps: [(0, 0.5), (1, 0.01)]
    :parameter file_path: path <- input.mp4
    :parameter save_path: path -> file_001.jpg
    :required ffmpeg: -> https://gist.github.com/steven2358/ba153c642fe2bb1e47485962df07c730
    :command: ffmpeg -i input.mp4 -vf fps=(0.5 / 0.01) output-%03d.jpg -hide_banner
    """
    call("ffmpeg -i {} -vf fps={} {}\\{}_%03d.jpg -hide_banner".format(
        file_path, fps, save_path, file_path.split("\\")[-1][:-4]
    ), shell=True)


def extract_pictures_from_video():
    """
    :function: extract pictures from each video (windows script)
    :back: 1 pic / 100s -> 50% val : 50% test
    # Video  ← v_parent
    # ├── G2_BK1114
    # └── G2_K1116
    #     └── G2_K1116_1.mp4  ← input.mp4
    #     └── G2_K1116_2.mp4  ← normal file (fps = 0.01)
    #     └── _G2_K1116_3.mp4  ← small file (fps = 0.5)
    """
    v_parent = r"D:\Document\res_DeepLearning\Video"
    val_save_path = r"D:\projects\datasets\EXP_Detect\static\images\val"
    test_save_path = r"D:\projects\datasets\EXP_Detect\static\images\test"

    for dir_item in os.listdir(v_parent):
        dir_path = v_parent + '\\' + dir_item
        for i, file in enumerate(os.listdir(dir_path)):
            fps = 0.5 if file[0] == '_' else 0.01  # 每秒提取的帧数: [(small_file, 0.5), (normal_file, 0.01)]
            if i % 2 == 0:
                action_extract(dir_path + '\\' + file, val_save_path, fps)
            else:
                action_extract(dir_path + '\\' + file, test_save_path, fps)


def test():
    inp = "D:\\Document\\res_DeepLearning\\Video\\G2_BK1114\\G2_BK1114_1.mp4"
    oup = "D:\\projects\\datasets\\EXP_Detect\\static\\images\\val\\G2_BK1114_1_%03d.jpg"
    print("ffmpeg -i {} -vf fps=0.01 {} -hide_banner".format(
        inp, oup
    ))


def main():
    # test()
    extract_pictures_from_video()


if __name__ == "__main__":
    main()

