for i in /home/ubuntu/Results/completed-classifier-datasets-bruteforce/TUM_RGBD_SLAM/*/;
do
    ./bin/opensfm calculate_features $i > $i/output.log 2&>1
done
