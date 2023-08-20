import random
# 写txt
import os
import tqdm

imgpath = '../data/Istanbul_Road_Data_Set/images'
txtsavepath = '../data/Istanbul_Road_Data_Set/'

imgtxt_1 = open('../data/Istanbul_Road_Data_Set/train.txt', 'w')
imgtxt_2 = open('../data/Istanbul_Road_Data_Set/val.txt', 'w')
imgtxt_3 = open('../data/Istanbul_Road_Data_Set/test.txt', 'w')

n = 0
image_list = os.listdir(imgpath)
# random.shuffle(image_list)  # 使用shuffle()函数打乱原始列表
for img in tqdm.tqdm(image_list):
    if n < (0.8 * len(image_list)):
        name = img[0:-8] + '\n'
        imgtxt_1.write(name)
        n = n + 1
    elif 0.8 * len(image_list) <= n < 0.9 * len(image_list):
        name = img[0:-8] + '\n'
        imgtxt_2.write(name)
        n = n + 1
    else:
        name = img[0:-8] + '\n'
        imgtxt_3.write(name)
        n = n + 1

imgtxt_1.close()
imgtxt_2.close()
imgtxt_3.close()
