# copy_mode="full"
copy_mode="minimal"
# copy_mode="matching_results"
# copy_mode="reconstructions"

# remote_server="ec2-18-218-17-167.us-east-2.compute.amazonaws.com" # tum_rgbd_slam
remote_server="ec2-3-14-82-141.us-east-2.compute.amazonaws.com" # eth3d and tanksandtemples
# remote_server="ec2-18-222-195-45.us-east-2.compute.amazonaws.com" # uiuctag
local_root="/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce"
# relevant_dataset="TUM_RGBD_SLAM"
# relevant_dataset="ETH3D"
relevant_dataset="TanksAndTemples"
# relevant_dataset="UIUCTag"

# TanksAndTemples_relevant_sequences=('Barn' 'Caterpillar' 'Church' 'Courthouse' 'Ignatius' 'Meetingroom' 'Truck')
TanksAndTemples_relevant_sequences=('Auditorium' 'Ballroom' 'Courtroom' 'Family' 'Francis' 'Horse' 'Lighthouse' 'M60' 'Museum' 'Palace' 'Panther' 'Playground' 'Temple' 'Train')

# TanksAndTemples_relevant_sequences=('Auditorium' 'Courtroom' 'Francis' 'Horse' 'Lighthouse' 'M60' 'Train')
# TanksAndTemples_relevant_sequences=('Ballroom' 'Family' 'Museum' 'Palace' 'Panther' 'Playground' 'Temple')

# TanksAndTemples_relevant_sequences=('Meetingroom')
UIUCTag_relevant_sequences=('ece_floor5_wall' 'ece_floor3_loop_ccw')

datasets=($(ssh ubuntu@$remote_server ls -d /home/ubuntu/Results/completed-classifier-datasets-bruteforce/*/*/))
echo "*********************************************************************************************************************************************************************************************"
for ds in "${datasets[@]}"; do
	dataset_name=$(echo $ds | cut -d '/' -f 6)
	sequence_name=$(echo $ds | cut -d '/' -f 7)
	local_folder=$(echo $local_root/$dataset_name/$sequence_name)
	
	echo -e "\tIterating dataset - $dataset_name : $sequence_name"
	if [ "$dataset_name" != "$relevant_dataset" ] ;then
		continue
	fi

	if [ "$relevant_dataset" == "TanksAndTemples" ] && [[ ! " ${TanksAndTemples_relevant_sequences[@]} " =~ " ${sequence_name} " ]];then
		echo -e "\t\tTanksAndTemples: Skipping dataset - $sequence_name"
		continue
	fi

	if [ "$relevant_dataset" == "UIUCTag" ] && [[ ! " ${UIUCTag_relevant_sequences[@]} " =~ " ${sequence_name} " ]];then
		echo -e "\t\tUIUCTag: Skipping dataset - $sequence_name"
		continue
	fi

	echo "*********************************************************************************************************************************************************************************************"
	echo -e "\t\tProcessing dataset - $dataset_name : $sequence_name"

	if [ "$copy_mode" == "full" ] || [ "$copy_mode" == "minimal" ] || [ "$copy_mode" == "reconstructions" ];then
		echo -e "\t\tCreating necessary folders under $local_folder"
		mkdir -p $local_folder
		mkdir -p $local_folder/features/
		mkdir -p $local_folder/classifier_dataset/
		mkdir -p $local_folder/classifier_features/
		mkdir -p $local_folder/rmatches_secondary/
		mkdir -p $local_folder/all_matches/
		mkdir -p $local_folder/matches/
		mkdir -p $local_folder/results/
		if [ "$copy_mode" == "full" ];then
			mkdir -p $local_folder/exif/
			mkdir -p $local_folder/images-blurred/
			mkdir -p $local_folder/images-resized/
			mkdir -p $local_folder/images-resized-processed/
		fi
	fi

	start=$SECONDS
	if [ "$copy_mode" == "full" ];then
		echo -e "\t\tCopying dataset: "$ds" to $local_folder - mode: full";
		# rsync -aqz ubuntu@$remote_server:$ds/camera_models.json $local_folder/
		# rsync -aqz ubuntu@$remote_server:$ds/reconstruction-*.json $local_folder/
		# rsync -aqz ubuntu@$remote_server:$ds/tracks*.csv $local_folder/
		# rsync -aqz ubuntu@$remote_server:$ds/exif/* $local_folder/exif/
		# rsync -aqz ubuntu@$remote_server:$ds/features/* $local_folder/features/
		rsync -aqz ubuntu@$remote_server:$ds/classifier_dataset/* $local_folder/classifier_dataset/
		# rsync -aqz ubuntu@$remote_server:$ds/classifier_features/* $local_folder/classifier_features/
		# rsync -aqz ubuntu@$remote_server:$ds/images-blurred/* $local_folder/images-blurred/
		# rsync -aqz ubuntu@$remote_server:$ds/images-resized/* $local_folder/images-resized/
		# rsync -aqz ubuntu@$remote_server:$ds/images-resized-processed/* $local_folder/images-resized-processed/
		# rsync -aqz ubuntu@$remote_server:$ds/rmatches_secondary/* $local_folder/rmatches_secondary/
		# rsync -aqz ubuntu@$remote_server:$ds/all_matches/* $local_folder/all_matches/
		# rsync -aqz ubuntu@$remote_server:$ds/matches/* $local_folder/matches/
	elif [ "$copy_mode" == "minimal" ]; then
		echo -e "\t\tCopying dataset: "$ds" to $local_folder - mode: minimal";
		# rsync -aqz ubuntu@$remote_server:$ds/reconstruction-*.json $local_folder/
		# rsync -aqz ubuntu@$remote_server:$ds/tracks*.csv $local_folder/
		# rsync -aqz ubuntu@$remote_server:$ds/classifier_dataset/* $local_folder/classifier_dataset/
		# rsync -aqz ubuntu@$remote_server:$ds/features/* $local_folder/features/
		# rsync -aqz ubuntu@$remote_server:$ds/classifier_features/feature_maps $local_folder/classifier_features/
		rsync -aqz ubuntu@$remote_server:$ds/classifier_features/mds_* $local_folder/classifier_features/
		# rsync -aqz ubuntu@$remote_server:$ds/classifier_features/match_maps $local_folder/classifier_features/
		# rsync -aqz ubuntu@$remote_server:$ds/classifier_features/pe_maps $local_folder/classifier_features/
		# rsync -aqz ubuntu@$remote_server:$ds/rmatches_secondary/* $local_folder/rmatches_secondary/
		# rsync -aqz ubuntu@$remote_server:$ds/all_matches/* $local_folder/all_matches/
		# rsync -aqz ubuntu@$remote_server:$ds/matches/* $local_folder/matches/
	elif [ "$copy_mode" == "reconstructions" ]; then
		echo -e "\t\tCopying dataset: "$ds" to $local_folder - mode: reconstructions";
		# rsync -aqz ubuntu@$remote_server:$ds/tracks*.csv $local_folder/
		# rsync -aqz ubuntu@$remote_server:$ds/reconstruction-*.json $local_folder/
		rsync -aqz ubuntu@$remote_server:$ds/results/*.json $local_folder/results/
	elif [ "$copy_mode" == "matching_results" ]; then
		echo -e "\t\tCopying dataset: "$ds" to $local_folder - mode: matching_results";
		rsync -aqz $local_folder/classifier_features/image_matching_results_* ubuntu@$remote_server:$ds/classifier_features/
		rsync -aqz $local_folder/classifier_features/feature_matching_results ubuntu@$remote_server:$ds/classifier_features/
	else
	   	echo -e "\tNeed to specify copy mode as either 'full', 'minimal' or 'matching_results'"
	   	break;
	fi
	duration=$(( SECONDS - start ))
	echo -e "\t\tTime: $duration secs"
	echo "*********************************************************************************************************************************************************************************************"
	# break;
done