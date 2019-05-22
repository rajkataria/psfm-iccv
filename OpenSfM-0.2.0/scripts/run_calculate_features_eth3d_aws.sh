for i in /home/ubuntu/Results/completed-classifier-datasets-bruteforce/ETH3D/*/;
do
    ./bin/opensfm calculate_features $i
done
