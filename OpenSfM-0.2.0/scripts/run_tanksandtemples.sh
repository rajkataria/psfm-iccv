for i in /home/ubuntu/Results/classifier/TanksAndTemples/*/;
do
    ./bin/opensfm_create_classifier_datasets $i ~/Results/classifier/ETH3D/meadow/
done
