import os
import random
import re
import sys
from glob import glob
from pathlib import Path

import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.applications import vgg16, vgg19, resnet, xception, nasnet, mobilenet, mobilenet_v2, \
    inception_resnet_v2, inception_v3, densenet

from src import tensorfi_plus as tfi_batch
from src.utility import get_fault_injection_configs


def get_model_from_name(model_name):
    if model_name == "ResNet50":
        return resnet.ResNet50()
    elif model_name == "ResNet101":
        return resnet.ResNet101()
    elif model_name == "ResNet152":
        return resnet.ResNet152()
    elif model_name == "VGG16":
        return vgg16.VGG16()
    elif model_name == "VGG19":
        return vgg19.VGG19()
    elif model_name == "Xception":
        return xception.Xception()
    elif model_name == "NASNetMobile":
        return nasnet.NASNetMobile()
    elif model_name == "NASNetLarge":
        return nasnet.NASNetLarge()
    elif model_name == "MobileNet":
        return mobilenet.MobileNet()
    elif model_name == "MobileNetV2":
        return mobilenet_v2.MobileNetV2()
    elif model_name == "InceptionResNetV2":
        return inception_resnet_v2.InceptionResNetV2()
    elif model_name == "InceptionV3":
        return inception_v3.InceptionV3()
    elif model_name == "DenseNet121":
        return densenet.DenseNet121()
    elif model_name == "DenseNet169":
        return densenet.DenseNet169()
    elif model_name == "DenseNet201":
        return densenet.DenseNet201()


def get_preprocessed_input_by_model_name(model_name, x_val):
    if model_name == "ResNet50" or model_name == "ResNet101" or model_name == "ResNet152":
        return resnet.preprocess_input(x_val)
    elif model_name == "VGG16":
        return vgg16.preprocess_input(x_val)
    elif model_name == "VGG19":
        return vgg19.preprocess_input(x_val)
    elif model_name == "Xception":
        return xception.preprocess_input(x_val)
    elif model_name == "NASNetMobile" or model_name == "NASNetLarge":
        return nasnet.preprocess_input(x_val)
    elif model_name == "MobileNet":
        return mobilenet.preprocess_input(x_val)
    elif model_name == "MobileNetV2":
        return mobilenet_v2.preprocess_input(x_val)
    elif model_name == "InceptionResNetV2":
        return inception_resnet_v2.preprocess_input(x_val)
    elif model_name == "InceptionV3":
        return inception_v3.preprocess_input(x_val)
    elif model_name == "DenseNet121" or model_name == "DenseNet169" or model_name == "DenseNet201":
        return densenet.preprocess_input(x_val)


def get_data_path_by_model_name(model_name, path_imagenet_val_dataset):
    if model_name == "ResNet50" or model_name == "ResNet101" or model_name == "ResNet152" or model_name == "VGG16" \
            or model_name == "VGG19" or model_name == "NASNetMobile" or model_name == "MobileNet" \
            or model_name == "MobileNetV2" or model_name == "DenseNet121" or model_name == "DenseNet169" \
            or model_name == "DenseNet201" or model_name == "EfficientNetB0":
        return str(path_imagenet_val_dataset) + "/sampled_new_x_val_224_1.npy"
    elif model_name == "Xception" or model_name == "InceptionResNetV2" or model_name == "InceptionV3":
        return str(path_imagenet_val_dataset) + "/sampled_new_x_val_299_1.npy"
    elif model_name == "NASNetLarge":
        return str(path_imagenet_val_dataset) + "/sampled_new_x_val_331_1.npy"


def get_statistics_without_fault_injection(original_label_list, predicted_label_list):
    correct_indices = []
    wrong_indices = []
    for i in range(len(original_label_list)):
        org_val = original_label_list[i]
        pred_val = predicted_label_list[i]
        if org_val != pred_val:
            wrong_indices.append(i)
        else:
            correct_indices.append(i)
    return correct_indices, wrong_indices


def main():
    model_name = sys.argv[1]
    process_no = int(sys.argv[2])
    file_object = open('imagenet_final_log_' + model_name + '_' + str(process_no) + '.txt', 'a')
    path_imagenet_val_dataset = Path("imagenet_data/")  # path/to/data/
    y_val = np.load(str(path_imagenet_val_dataset / "y_val_sampled.npy"))
    x_val_path = get_data_path_by_model_name(model_name=model_name, path_imagenet_val_dataset=path_imagenet_val_dataset)

    print("Model name: " + model_name)
    file_object.write("Model name: " + model_name)
    file_object.write("\n")
    file_object.flush()
    K.clear_session()

    low = 200 * process_no
    high = 200 * (process_no + 1)
    print("Low: " + str(low) + " , High: " + str(high))
    file_object.write("Low: " + str(low) + " , High: " + str(high))
    file_object.write("\n")
    file_object.flush()

    model = get_model_from_name(model_name)

    predicted_label_list = []
    x_val = np.load(x_val_path).astype('float32')
    x_val = get_preprocessed_input_by_model_name(model_name, x_val)
    data_count, _, _, _ = x_val.shape
    for i in range(data_count):
        if low <= i < high:
            img = x_val[i]
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            predicted_label = model.predict(img).argmax(axis=-1)[0]
            predicted_label_list.append(predicted_label)
            print(str(i) + " : " + str(y_val[i]) + " : " + str(predicted_label))
            file_object.write(str(i) + " : " + str(y_val[i]) + " : " + str(predicted_label))
            file_object.write("\n")
            file_object.flush()
    correct_indices, wrong_indices = get_statistics_without_fault_injection(y_val[low:high], predicted_label_list)

    yaml_file = "confFiles/sample1.yaml"

    model_graph, super_nodes = get_fault_injection_configs(model)
    count = 0
    for i in range(data_count):
        if low <= i < high:
            if count in correct_indices:
                for j in range(5):
                    img = x_val[i]
                    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
                    res = tfi_batch.inject(model=model, x_test=img, confFile=yaml_file,
                                           model_graph=model_graph, super_nodes=super_nodes)
                    faulty_prediction = res.final_label[0]
                    print(str(i) + " : " + str(y_val[i]) + " : " + str(predicted_label_list[count]) + " : " + str(
                        faulty_prediction))
                    file_object.write(str(i) + " : " + str(y_val[i]) + " : " + str(predicted_label_list[count]) + " : " + str(
                        faulty_prediction))
                    file_object.write("\n")
                    file_object.flush()
            else:
                for j in range(30):
                    img = x_val[i]
                    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
                    res = tfi_batch.inject(model=model, x_test=img, confFile=yaml_file,
                                           model_graph=model_graph, super_nodes=super_nodes)
                    faulty_prediction = res.final_label[0]
                    print(str(i) + " : " + str(y_val[i]) + " : " + str(predicted_label_list[count]) + " : " + str(
                        faulty_prediction))
                    file_object.write(str(i) + " : " + str(y_val[i]) + " : " + str(predicted_label_list[count]) + " : " + str(
                        faulty_prediction))
                    file_object.write("\n")
                    file_object.flush()
            count += 1
    file_object.close()


if __name__ == '__main__':
    main()
