copy_mode="full"
remote_server="ec2-3-17-81-213.us-east-2.compute.amazonaws.com"
local_root="/hdd/Research/psfm-iccv/data/temp-copy"

datasets=($(ssh ubuntu@$remote_server ls -d /home/ubuntu/Results/completed-classifier-datasets-bruteforce/*/*/))
echo "*********************************************************************************************************************************************************************************************"
for ds in "${datasets[@]}"; do
	dataset_name=$(echo $ds | cut -d '/' -f 6)
	sequence_name=$(echo $ds | cut -d '/' -f 7)
	local_folder=$(echo $local_root/$dataset_name/$sequence_name)
	
	if [ "$copy_mode" == "full" ] || [ "$copy_mode" == "minimal" ];then
		echo -e "\tCreating necessary folders under $local_folder"
		mkdir -p $local_folder
		mkdir -p $local_folder/classifier_dataset/
		mkdir -p $local_folder/classifier_features/
		mkdir -p $local_folder/images-blurred/
		mkdir -p $local_folder/images-resized/
		mkdir -p $local_folder/images-resized-processed/
		mkdir -p $local_folder/rmatches_secondary/
	fi

	echo -e "\tCopying dataset: "$ds" to $local_folder - mode: "$copy_mode;
	if [ "$copy_mode" == "full" ];then
		echo "scp -r ubuntu@$remote_server:$ds/classifier_dataset/* $local_folder/classifier_dataset/"
		echo "scp -r ubuntu@$remote_server:$ds/classifier_features/* $local_folder/classifier_features/"
		echo "scp -r ubuntu@$remote_server:$ds/images-blurred/* $local_folder/images-blurred/"
		echo "scp -r ubuntu@$remote_server:$ds/images-resized/* $local_folder/images-resized/"
		echo "scp -r ubuntu@$remote_server:$ds/images-resized-processed/* $local_folder/images-resized-processed/"
		echo "scp -r ubuntu@$remote_server:$ds/rmatches_secondary/* $local_folder/rmatches_secondary/"
	elif [ "$copy_mode" == "minimal" ]; then
		echo "scp -r ubuntu@$remote_server:$ds/classifier_dataset/image_matching_dataset_*.csv $local_folder/classifier_dataset/"
		echo "scp -r ubuntu@$remote_server:$ds/classifier_features/feature_maps $local_folder/classifier_features/"
		echo "scp -r ubuntu@$remote_server:$ds/classifier_features/match_maps $local_folder/classifier_features/"
		echo "scp -r ubuntu@$remote_server:$ds/rmatches_secondary/* $local_folder/rmatches_secondary/"
	elif [ "$copy_mode" == "matching_results" ]; then
		echo "scp -r $local_folder/classifier_features/image_matching_results_* ubuntu@$remote_server:$ds/classifier_features/"
	else
	   	echo -e "\tNeed to specify copy mode as either 'full', 'minimal' or 'matching_results'"
	fi
	echo "*********************************************************************************************************************************************************************************************"
	break;
done