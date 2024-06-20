#INSTRUCTIONS HOW TO WORK WITH THE CODE
#
# 1ste create the patches
# Go to the create_patches.sh file
# Change the source, conda, which and cd to the right environments
# Then change the--source and --save_dir in the line with CUDA_VISIBLE_DEVICES
#
#
# 2nd create the features
# Go to the create_features.sh file
# Change the source, conda, which and cd to the right environments
# Then change the --model_name, --data_h5_dir, --data_slide_dir, --csv_path and --feat_dir to the right paths
#
# --model_name can be changed to: resnet18, resnet50_trunc and resnet50_ssl
#
#
# 3th make the model
# Go to the train.sh file
# Change the source, conda, which and cd to the right environments
# Then change --exp_code and --data_root_dir to the right paths
#
# Go to the main.py file
# Then change on line 159 the path behind data_root_dir
# Options can be: Camelyon16_patch256_res18, Camelyon16_patch256_res50 and Camelyon16_patch256_res50_ssl these can be found in the data_feat folder
#
# Go to the model_abmil.py file inside of the models folder
# Look at lines 82 and 83
# If you use ResNet50 then use line 82 and mute line 83
# If you use ResNet18 then use line 83 and mute line 82
#
# Model evaluation
# In the results folder
# Every trained model has an summary.csv file where the results of the 5 folds can be found
#
# Visual results
# Go to the terminal 
# Open python by entering: python
# Then enter: import os
# Then enter: os.system('tensorboard --logdir=/home/bme001/20202047/CLAM/results/camelyon16_abmil_resnet_50_ssl_s1 --port=6007')
# where: camelyon16_abmil_resnet_50_ssl_s1 can also be changed to camelyon16_abmil_resnet_50_s1 and camelyon16_abmil_resnet_18_s1
# To exit do: ctrl c 
# and the: exit()
#
