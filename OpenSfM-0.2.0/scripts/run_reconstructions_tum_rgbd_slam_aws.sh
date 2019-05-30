for i in /home/ubuntu/Results/completed-classifier-datasets-bruteforce/TUM_RGBD_SLAM/*/;
do
    bash ./scripts/run_reconstructions.sh $i 
done
