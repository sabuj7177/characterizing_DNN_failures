# Characterizing Deep Learning Neural Network Failures between Algorithmic Inaccuracy and Transient Hardware Faults

## Installation

Prerequisites for installing the project,
1. Ubuntu OS (Tested with Ubuntu 18.04)
2. Python 3.6
3. Resolve all the dependency from requirements.txt file


Download GTSRB and cifar100_keras pretrained models from https://drive.google.com/drive/folders/1TsKEYBOSVue4BVNq0GSiT7LPlJevpaWV?usp=sharing


Download imagenet_data, GTSRB and cifar100 folders from https://drive.google.com/drive/folders/12_wNg1EXtSQ7ZYP_vEcrx-Et8wQK6tGq?usp=sharing


## Run the following python scripts to reproduce the results:

cifar100_super_label_generate.py: Sample code to generate super label mapping

sample_gtsrb_data: Sample 10000 images from 60000 test set of GTSRB dataset.
sample_imagenet_data: Sample 10000 images from 60000 test set.
imagenet_sampled_data_prepare.py: prepare sampled CIFAR100 data for inference.
imagenet_super_label_generate.py: generate super label for imagenet dataset
synset_to_keras_mapping: Prepare imagenet label

cifar100_final_FI.py : Store fault injection result for a particular range of Cifar100 images
parallel_processing_cifar100.py : Parallely execute fault injection on a specific model of cifar100 dataset. It uses cifar100_final_FI.py file.
merge_cifar100_result.py : Merge all the partial results generated from above command to a single file.
process_cifar100_result_av_mis_normalized.py : Compute SCM probability on initially fault free data.
process_cifar100_result_fault_free.py : Compute SCM probability of fault injected data.

gtsrb_final_FI.py : Store fault injection result for a particular range of gtsrb images
parallel_processing_gtsrb.py : Parallely execute fault injection on a specific model of gtsrb dataset. It uses gtsrb_final_FI.py file.
merge_gtsrb_result.py : Merge all the partial results generated from above command to a single file.
process_gtsrb_result_normalized.py: Compute SDC probability of GTSRB dataset

imagenet_final_FI.py : Store fault injection result for a particular range of imagenet images
parallel_processing_imagenet.py : Parallely execute fault injection on a specific model of imagenet dataset. It uses imagenet_final_FI.py file.
merge_imagenet_result.py : Merge all the partial results generated from above command to a single file.
process_imagenet_result_av_mis_normalized.py : Compute SCM probability on initially fault free data.
process_imagenet_result_fault_free.py : Compute SCM probability of fault injected data.