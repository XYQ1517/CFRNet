import os
import shutil
import random

# import tqdm
# path = '../data/deepglobe/train'
# new_path_1 = '../data/deepglobe/gt'
# new_path_2 = '../data/deepglobe/images'
#
# if not os.path.exists(new_path_1):
#     os.mkdir(new_path_1)
# for root, dirs, files in os.walk(path):
#     for i in tqdm.tqdm(range(len(files))):
#         if (files[i][-3:] == 'png'):
#             file_path = root + '/' + files[i]
#             new_file_path = new_path_1 + '/' + files[i]
#             shutil.copy(file_path, new_file_path)
# print("完成")
#
# if not os.path.exists(new_path_2):
#     os.mkdir(new_path_2)
# for root, dirs, files in os.walk(path):
#     for i in tqdm.tqdm(range(len(files))):
#         if (files[i][-3:] == 'jpg'):
#             file_path = root + '/' + files[i]
#             new_file_path = new_path_2 + '/' + files[i]
#             shutil.copy(file_path, new_file_path)
# print("完成")


# 写txt
import os
import tqdm

imgpath = '../data/CHN6/images'
txtsavepath = '../data/CHN6/'

imgtxt_1 = open('../data/CHN6/train.txt', 'w')
imgtxt_2 = open('../data/CHN6/val.txt', 'w')
imgtxt_3 = open('../data/CHN6/test.txt', 'w')

n = 1
image_list = os.listdir(imgpath)
random.shuffle(image_list)  # 使用shuffle()函数打乱原始列表
print(0.8 * len(image_list))
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
