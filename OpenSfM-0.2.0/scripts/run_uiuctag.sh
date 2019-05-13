for i in /home/ubuntu/Results/classifier/UIUCTag/*/;
do
    ./bin/opensfm_create_classifier_datasets $i
done
