for i in /home/ubuntu/Results/completed-classifier-datasets-bruteforce/TanksAndTemples/*/;
do
    echo "***************************************************************************************************"
    echo $i
    ./bin/opensfm calculate_features $i 
done
