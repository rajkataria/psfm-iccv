for i in /home/ubuntu/Results/completed-classifier-datasets-bruteforce/TanksAndTemples/*/;
do
    bash ./scripts/run_reconstruction.sh $i 
done
