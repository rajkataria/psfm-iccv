for i in /home/ubuntu/Results/completed-classifier-datasets-bruteforce/UIUCTag/*/;
do
    echo "***************************************************************************************************"
    echo $i
    ./bin/opensfm calculate_features $i 
done
