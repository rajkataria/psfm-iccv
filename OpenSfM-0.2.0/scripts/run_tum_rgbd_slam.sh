for i in /home/ubuntu/Results/classifier/TUM_RGBD_SLAM/*/;
do
    ./bin/opensfm_create_classifier_datasets $i
done
