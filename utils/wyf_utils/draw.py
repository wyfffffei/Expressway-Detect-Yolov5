import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def draw_map():
    data_1 = pd.read_csv("./train_2_v5m/results.csv", usecols=[0,6])
    data_1 = np.array(data_1.loc[:, :])
    x_1 = data_1[:, 0]
    y_1 = data_1[:, 1]

    data_2 = pd.read_csv("./train_3_v5x/results.csv", usecols=[0,6])
    data_2 = np.array(data_2.loc[:, :])
    x_2 = data_2[:, 0]
    y_2 = data_2[:, 1]

    data_3 = pd.read_csv("./train_5_v5x_ADAM/results.csv", usecols=[0,6])
    data_3 = np.array(data_3.loc[:, :])
    x_3 = data_3[:, 0]
    y_3 = data_3[:, 1]

    plt.title("metrics/mAP_0.5")
    plt.plot(x_1, y_1, color='yellow', label="YOLOv5m - SGD")
    plt.plot(x_2, y_2, color="blue", label="YOLOv5x - SGD")
    plt.plot(x_3, y_3, color="red", label="YOLOv5x - ADAM")

    plt.ylim(0, 1)
    plt.legend()
    plt.show()
    plt.close()


def draw_pr_F1():
    data = pd.read_csv("./train_3_v5x/results.csv", usecols=[0,4,5])
    data = np.array(data.loc[:, :])
    x = data[:, 0]
    y_1 = data[:, 1]
    y_2 = data[:, 2]
    y_3 = 2 * ((y_1 * y_2) / (y_1 + y_2))
    print("precision: ", end=str(np.mean(y_1)))
    print()
    print("recall: ", end=str(np.mean(y_2)))
    print()
    print("F1-score: ", end=str(np.mean(y_3)))
    plt.title("PR/F1-Curve")
    plt.plot(x, y_1, color='red', label="metrics/precision")
    plt.plot(x, y_2, color="green", label="metrics/recall")
    plt.plot(x, y_3, color="blue", label="metrics/F1-score")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
    plt.close()


def draw_loss():
    data = pd.read_csv("./train_5_v5x_ADAM/results.csv", usecols=[0,1,2,3,8,9,10])
    data = np.array(data.loc[:, :])
    x = data[:, 0]
    train_box = data[:, 1]
    train_obj = data[:, 2]
    train_cls = data[:, 3]
    val_box = data[:, 4]
    val_obj = data[:, 5]
    val_cls = data[:, 6]
    
    plt.subplot(2, 3 ,1)
    plt.plot(x, train_box, color='blue')
    plt.title("train/box_loss")
    plt.grid()

    plt.subplot(2, 3, 2)
    plt.plot(x, train_obj, color="blue")
    plt.title("train/obj_loss")
    plt.grid()

    plt.subplot(2, 3, 3)
    plt.plot(x, train_cls, color="blue")
    plt.title("train/cls_loss")
    plt.grid()

    plt.subplot(2, 3 ,4)
    plt.plot(x, val_box, color='blue')
    plt.title("val/box_loss")
    plt.grid()

    plt.subplot(2, 3, 5)
    plt.plot(x, val_obj, color="blue")
    plt.title("val/obj_loss")
    plt.grid()

    plt.subplot(2, 3, 6)
    plt.plot(x, val_cls, color="blue", label="error")
    plt.title("val/cls_loss")
    plt.grid()

    plt.ylim(0, 1)
    plt.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":
    # draw_map()
    draw_loss()
