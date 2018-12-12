for dataset in ./data/*/;
do 
	echo $dataset;
	bash ./scripts/run_reconstructions.sh $dataset > $dataset/output.log 2>&1;
done
