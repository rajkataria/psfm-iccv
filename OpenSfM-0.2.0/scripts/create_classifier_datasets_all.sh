for dataset in /hdd/Research/psfm-iccv/data/classifier-datasets/*/*/;
do 
	echo $dataset;
	bash ./scripts/create_classifier_datasets.sh $dataset > $dataset/create-classifier-datasets.log 2>&1;
done
