import os
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import scipy.io

path_labels = Path("imagenet_data/ILSVRC2012_validation_ground_truth.txt")
path_synset_words = Path("imagenet_data/synset_words.txt")
path_meta = Path("imagenet_data/meta.mat")

meta = scipy.io.loadmat(str(path_meta))
original_idx_to_synset = {}
synset_to_name = {}

for i in range(1000):
    ilsvrc2012_id = int(meta["synsets"][i, 0][0][0][0])
    synset = meta["synsets"][i, 0][1][0]
    name = meta["synsets"][i, 0][2][0]
    original_idx_to_synset[ilsvrc2012_id] = synset
    synset_to_name[synset] = name

synset_to_keras_idx = {}
keras_idx_to_name = {}
with open(str(path_synset_words), "r") as f:
    for idx, line in enumerate(f):
        parts = line.split(" ")
        synset_to_keras_idx[parts[0]] = idx
        keras_idx_to_name[idx] = " ".join(parts[1:])

convert_original_idx_to_keras_idx = lambda idx: synset_to_keras_idx[original_idx_to_synset[idx]]

with open(str(path_labels), "r") as f:
    y_val = f.read().strip().split("\n")
    y_val = np.array([convert_original_idx_to_keras_idx(int(idx)) for idx in y_val])

# The following part writes keras label and object name mapping
file_object = open('imagenet_data/keras_label_to_name_imagenet.txt', 'a')
file_object.write("{\n")
for key in keras_idx_to_name:
    file_object.write(str(key))
    file_object.write(':"')
    file_object.write(keras_idx_to_name[key].rstrip("\n"))
    file_object.write('"\n')
file_object.write("}\n")

# The following part writes keras mapping of all imagenet validation dataset
file_object = open('imagenet_data/keras_labels_imagenet.txt', 'a')
for y in y_val:
    file_object.write(str(y))
    file_object.write('\n')
