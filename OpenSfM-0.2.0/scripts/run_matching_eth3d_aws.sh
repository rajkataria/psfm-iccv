for i in /home/ubuntu/Results/classifier/ETH3D/*/;
do
    ./bin/opensfm_create_classifier_datasets $i > $i/output.log 2&>1
done
