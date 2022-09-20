import os
import random
import shutil
import sys
from glob import glob
from pathlib import Path
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    data = pd.read_csv("GTSRB/Test.csv")
    y_test = data['ClassId'].tolist()
    test_data_paths = data['Path'].tolist()
    path = "GTSRB/"
    new_path = "GTSRB/Test2"
    os.mkdir(new_path)
    for i in range(43):
        os.mkdir(new_path + "/" + str(i))
    count = 0
    for data in test_data_paths:
        img_path = os.path.join(path, data)
        shutil.copy(img_path, new_path + "/" + str(y_test[count]))
        count += 1


    # try:
    #     sampled_image_count = int(sys.argv[1])
    # except:
    #     sampled_image_count = 10000
    # print("Trying to sample " + str(sampled_image_count) + " images")
    # original_image_dir = Path("imagenet_data/val")
    # original_path_labels = Path("imagenet_data/ILSVRC2012_validation_ground_truth.txt")
    # sampled_image_dir = Path("imagenet_data/sampled_val/")
    # sampled_path_labels = Path("imagenet_data/sampled_validation_ground_truth.txt")
    #
    # image_paths = sorted(glob(str(original_image_dir / "*")))
    # n_images = len(image_paths)
    #
    # total_index = list(range(n_images))
    # sampled_index = random.sample(total_index, sampled_image_count)
    # sampled_index.sort()
    # sampled_images = [image_paths[i] for i in sampled_index]
    #
    # processed = 0
    # for image in sampled_images:
    #     shutil.copy(image, sampled_image_dir)
    #     processed = processed + 1
    #     if processed % 100 == 0:
    #         print("Processed " + str(processed) + " images")
    #
    # with open(str(original_path_labels), "r") as f:
    #     y_val = f.read().strip().split("\n")
    #     sampled_y_val = [y_val[i] for i in sampled_index]
    #     textfile = open(sampled_path_labels, "w")
    #     for element in sampled_y_val:
    #         textfile.write(element + "\n")
    #     textfile.close()
