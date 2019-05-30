for i in /home/ubuntu/Results/completed-classifier-datasets-bruteforce/ETH3D/*/;
do
    bash ./scripts/run_reconstruction.sh $i
done
