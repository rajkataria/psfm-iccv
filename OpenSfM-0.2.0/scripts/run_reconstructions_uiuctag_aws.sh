for i in /home/ubuntu/Results/completed-classifier-datasets-bruteforce/UIUCTag/*/;
do
    bash ./scripts/run_reconstructions.sh $i 
done
