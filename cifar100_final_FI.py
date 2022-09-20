import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from src import tensorfi_plus as tfi_batch
from src.utility import get_fault_injection_configs


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
    file_object = open('cifar100_final_log_' + model_name + '_' + str(process_no) + '.txt', 'a')
    inputs = np.load("cifar100_test_inputs.npy")
    labels = np.load("cifar100_test_labels.npy")
    total = 10000

    print("Model name: " + model_name)
    file_object.write("Model name: " + model_name)
    file_object.write("\n")
    file_object.flush()
    K.clear_session()
    model = tf.keras.models.load_model('keras_pretrained/' + model_name + '.h5')

    low = 500 * process_no
    high = 500 * (process_no + 1)
    print("Low: " + str(low) + " , High: " + str(high))
    file_object.write("Low: " + str(low) + " , High: " + str(high))
    file_object.write("\n")
    file_object.flush()

    predicted_label_list = []
    for i in range(total):
        if low <= i < high:
            predicted_label = model.predict(tf.expand_dims(inputs[i], axis=0)).argmax(axis=-1)[0]
            print(str(i) + " : " + str(labels[i]) + " : " + str(predicted_label))
            file_object.write(str(i) + " : " + str(labels[i]) + " : " + str(predicted_label))
            file_object.write("\n")
            file_object.flush()
            predicted_label_list.append(predicted_label)
    correct_indices, wrong_indices = get_statistics_without_fault_injection(labels[low:high], predicted_label_list)

    yaml_file = "confFiles/sample1.yaml"

    model_graph, super_nodes = get_fault_injection_configs(model)

    count = 0
    for i in range(total):
        if low <= i < high:
            if count in correct_indices:
                for j in range(5):
                    res = tfi_batch.inject(model=model, x_test=tf.expand_dims(inputs[i], axis=0),
                                           confFile=yaml_file, model_graph=model_graph, super_nodes=super_nodes)
                    faulty_prediction = res.final_label[0]
                    print(str(i) + " : " + str(labels[i]) + " : " + str(predicted_label_list[count]) + " : " + str(faulty_prediction))
                    file_object.write(str(i) + " : " + str(labels[i]) + " : " + str(predicted_label_list[count]) + " : " + str(faulty_prediction))
                    file_object.write("\n")
                    file_object.flush()
            else:
                for j in range(30):
                    res = tfi_batch.inject(model=model, x_test=tf.expand_dims(inputs[i], axis=0),
                                           confFile=yaml_file, model_graph=model_graph, super_nodes=super_nodes)
                    faulty_prediction = res.final_label[0]
                    print(str(i) + " : " + str(labels[i]) + " : " + str(predicted_label_list[count]) + " : " + str(faulty_prediction))
                    file_object.write(str(i) + " : " + str(labels[i]) + " : " + str(predicted_label_list[count]) + " : " + str(faulty_prediction))
                    file_object.write("\n")
                    file_object.flush()
            count += 1
    file_object.close()


if __name__ == '__main__':
    main()
