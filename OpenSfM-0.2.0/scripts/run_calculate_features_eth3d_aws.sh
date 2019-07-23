for i in /home/ubuntu/Results/completed-classifier-datasets-bruteforce/ETH3D/*/;
do
    echo "***************************************************************************************************"
    echo $i
    ./bin/opensfm calculate_features $i
done
