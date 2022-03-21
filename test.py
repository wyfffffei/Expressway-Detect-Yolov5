import torch

# print(torch.cuda.is_available())


save_path = "D:/projects/datasets/EXP_Detect/static/"
file = open(save_path + "test.txt", 'a+')
file.writelines("123" + "\n")
file.close()

file_2 = open(save_path + "test.txt", 'a+')
file_2.writelines("456" + "\n")
file_2.close()
