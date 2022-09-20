import random


def get_statistics_without_fault_injection(original_label_list, predicted_label_list):
    correct_classification = 0
    misclassification = 0
    correct_indices = []
    incorrect_indices = []
    for i in range(len(original_label_list)):
        org_val = original_label_list[i]
        pred_val = predicted_label_list[i]
        if org_val != pred_val:
            misclassification += 1
            incorrect_indices.append(i)
        else:
            correct_classification += 1
            correct_indices.append(i)
    return correct_indices, incorrect_indices, correct_classification, misclassification


def get_statistics_per_image_with_fault_injection(previous_predicted_label_list, faulty_label_list):
    benign_count = 0
    sdc_count = 0
    for i in range(len(previous_predicted_label_list)):
        prev_pred_val = previous_predicted_label_list[i]
        faulty_val = faulty_label_list[i]
        if prev_pred_val != faulty_val:
            sdc_count += 1
        else:
            benign_count += 1
    return benign_count, sdc_count


def get_string(data_list, index, text, model_count):
    s = text + '\t'
    for i in range(0, model_count):
        s += '{:.6f}'.format(data_list[i][index]) + '\t'
    return s + '\n'


def main():
    dataset_name = 'gtsrb'
    model_names = ['vgg16', 'vgg19', 'resnet34', 'resnet50', 'resnet101']
    file_object = open('gtsrb_processed_result.txt', 'a')
    for model_name in model_names:
        file_object.write("Model name: " + model_name + "\n")
        file1 = open(dataset_name + '_final_log_' + model_name + '.txt', 'r')
        lines = file1.readlines()
        lines.pop(0)
        y_val = []
        predicted_label_list = []
        for i in range(10000):
            line_parts = [x.strip() for x in lines[0].split(':')]
            y_val.append(int(line_parts[1]))
            predicted_label_list.append(int(line_parts[2]))
            lines.pop(0)

        correct_indices, wrong_indexes, correct_classification, misclassification \
            = get_statistics_without_fault_injection(y_val, predicted_label_list)
        file_object.write("Correct : " + str(correct_classification/100.0) + "\n")
        file_object.write("Wrong : " + str(misclassification/100.0) + "\n")

        fi_results = []
        fi_taken = []
        for i in range(10000):
            fi_taken.append(0)
            fi_results.append([])
        while len(lines) > 0:
            line_parts = [x.strip() for x in lines[0].split(':')]
            index = int(line_parts[0])
            fi_results[index].append([int(line_parts[2]), int(line_parts[3])])
            lines.pop(0)

        prev_pred_list = []
        faulty_list = []
        for i in range(3000):
            while True:
                image_index = random.choice(correct_indices)
                if fi_taken[image_index] < 5:
                    break
            fi_data = fi_results[image_index][fi_taken[image_index]]
            faulty_list.append(fi_data[1])
            prev_pred_list.append(fi_data[0])
            fi_taken[image_index] += 1

        benign_count, sdc_count = get_statistics_per_image_with_fault_injection(prev_pred_list, faulty_list)
        file_object.write("Benign while correct initially : " + str(benign_count/30.0) + "\n")
        file_object.write("SDC while correct initially : " + str(sdc_count/30.0) + "\n")

        prev_pred_list = []
        faulty_list = []
        for i in range(3000):
            while True:
                image_index = random.choice(wrong_indexes)
                if fi_taken[image_index] < 50:
                    break
            fi_data = fi_results[image_index][fi_taken[image_index]]
            faulty_list.append(fi_data[1])
            prev_pred_list.append(fi_data[0])
            fi_taken[image_index] += 1

        benign_count, sdc_count = get_statistics_per_image_with_fault_injection(prev_pred_list, faulty_list)
        file_object.write("Benign while wrong initially : " + str(benign_count / 30.0) + "\n")
        file_object.write("SDC while wrong initially : " + str(sdc_count / 30.0) + "\n")


if __name__ == '__main__':
    main()
