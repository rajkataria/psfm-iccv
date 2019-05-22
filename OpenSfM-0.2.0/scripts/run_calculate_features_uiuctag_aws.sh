for i in /home/ubuntu/Results/completed-classifier-datasets-bruteforce/UIUCTag/*/;
do
    ./bin/opensfm calculate_features $i 
done
