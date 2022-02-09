input=/datasets/UCSD_Anomaly_Dataset_v1p2/ped1_test.txt
output=./datasets/resnet/mask_background/UCSD_Anomaly_Dataset_v1p2_ped1_test_mask_background.pt
python extractor.py $input $output --gpu

input=/datasets/UCSD_Anomaly_Dataset_v1p2/ped1_train.txt
output=./datasets/resnet/mask_background/UCSD_Anomaly_Dataset_v1p2_ped1_train_mask_background.pt
python extractor.py $input $output --gpu
