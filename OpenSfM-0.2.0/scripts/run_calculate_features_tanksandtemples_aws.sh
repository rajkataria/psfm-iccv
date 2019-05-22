for i in /home/ubuntu/Results/completed-classifier-datasets-bruteforce/TanksAndTemples/*/;
do
    ./bin/opensfm calculate_features $i 
done
