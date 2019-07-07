dataset=$1
c_reconstruction_counter==${2:-0}

mode="reconstruction"

echo "############################################################################################################"
echo "############################################################################################################"
echo "########################################### Starting experiments ###########################################"
echo "############################################################################################################"
echo "############################################################################################################"

# Baseline + original resectioning
declare -A run99=(
	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        	[use_image_matching_classifier]='false'
	[use_weighted_resectioning]='original'										         	[use_weighted_feature_matches]='false'
	[use_image_matching_thresholding]='false'   [use_shortest_path_pruning]='false'       	[image_matching_classifier_threshold]='0.5'
	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'					[use_gt_closest_images_pruning]='false'
	[resectioning_config]='NA'
	)

# Baseline + colmap
declare -A run100=(
	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        	[use_image_matching_classifier]='false'
	[use_weighted_resectioning]='colmap'        											[use_weighted_feature_matches]='false'
	[use_image_matching_thresholding]='false'   [use_shortest_path_pruning]='false'       	[image_matching_classifier_threshold]='0.5'
	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'					[use_gt_closest_images_pruning]='false'
	[resectioning_config]='NA'
	)

# Baseline + weighted colmap resectioning (tracks-weighted-score: im+fm) 
declare -A run101=(
	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        	[use_image_matching_classifier]='false'
	[use_weighted_resectioning]='tracks-weighted-score'         							[use_weighted_feature_matches]='false'
	[use_image_matching_thresholding]='false'   [use_shortest_path_pruning]='false'       	[image_matching_classifier_threshold]='0.5'
	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'					[use_gt_closest_images_pruning]='false'
	[resectioning_config]='im+fm'
	)

# Baseline + weighted colmap resectioning (tracks-weighted-score: im) 
declare -A run102=(
	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        	[use_image_matching_classifier]='false'
	[use_weighted_resectioning]='tracks-weighted-score'         							[use_weighted_feature_matches]='false'
	[use_image_matching_thresholding]='false'   [use_shortest_path_pruning]='false'       	[image_matching_classifier_threshold]='0.5'
	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'					[use_gt_closest_images_pruning]='false'
	[resectioning_config]='im'
	)

# Baseline + weighted colmap resectioning (tracks-weighted-score: fm) 
declare -A run103=(
	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        	[use_image_matching_classifier]='false'
	[use_weighted_resectioning]='tracks-weighted-score'         							[use_weighted_feature_matches]='false'
	[use_image_matching_thresholding]='false'   [use_shortest_path_pruning]='false'       	[image_matching_classifier_threshold]='0.5'
	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'					[use_gt_closest_images_pruning]='false'
	[resectioning_config]='fm'
	)

# Baseline + weighted colmap resectioning (tracks-weighted-score: th) 
declare -A run104=(
	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        	[use_image_matching_classifier]='false'
	[use_weighted_resectioning]='tracks-weighted-score'         							[use_weighted_feature_matches]='false'
	[use_image_matching_thresholding]='false'   [use_shortest_path_pruning]='false'       	[image_matching_classifier_threshold]='0.5'
	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'					[use_gt_closest_images_pruning]='false'
	[resectioning_config]='th'
	)

all_runs=(run99 run100 run101 run102 run103 run104)

for run_name in "${all_runs[@]}"; do
    declare -n run_ref="$run_name"

    c_use_gt_matches="${run_ref[use_gt_matches]}"
    c_use_gt_selective_matches="${run_ref[use_gt_selective_matches]}"
    c_use_image_matching_classifier="${run_ref[use_image_matching_classifier]}"
    c_use_weighted_resectioning="${run_ref[use_weighted_resectioning]}"
    c_use_weighted_feature_matches="${run_ref[use_weighted_feature_matches]}"
    c_use_image_matching_thresholding="${run_ref[use_image_matching_thresholding]}"
    c_use_shortest_path_pruning="${run_ref[use_shortest_path_pruning]}"
    c_image_matching_classifier_threshold="${run_ref[image_matching_classifier_threshold]}"
    c_use_closest_images_pruning="${run_ref[use_closest_images_pruning]}"
    c_closest_images_top_k="${run_ref[closest_images_top_k]}"
    c_use_gt_closest_images_pruning="${run_ref[use_gt_closest_images_pruning]}"
    c_resectioning_config="${run_ref[resectioning_config]}"
    
    sed -i "s/use_gt_matches: .*/use_gt_matches: ${c_use_gt_matches}/g" $dataset/config.yaml
    sed -i "s/use_gt_selective_matches: .*/use_gt_selective_matches: ${c_use_gt_selective_matches}/g" $dataset/config.yaml
    sed -i "s/use_image_matching_classifier: .*/use_image_matching_classifier: ${c_use_image_matching_classifier}/g" $dataset/config.yaml
    sed -i "s/use_weighted_resectioning: .*/use_weighted_resectioning: ${c_use_weighted_resectioning}/g" $dataset/config.yaml
    sed -i "s/use_weighted_feature_matches: .*/use_weighted_feature_matches: ${c_use_weighted_feature_matches}/g" $dataset/config.yaml
    sed -i "s/use_image_matching_thresholding: .*/use_image_matching_thresholding: ${c_use_image_matching_thresholding}/g" $dataset/config.yaml
    sed -i "s/use_shortest_path_pruning: .*/use_shortest_path_pruning: ${c_use_shortest_path_pruning}/g" $dataset/config.yaml
    sed -i "s/image_matching_classifier_threshold: .*/image_matching_classifier_threshold: ${c_image_matching_classifier_threshold}/g" $dataset/config.yaml
    sed -i "s/use_closest_images_pruning: .*/use_closest_images_pruning: ${c_use_closest_images_pruning}/g" $dataset/config.yaml
    sed -i "s/closest_images_top_k: .*/closest_images_top_k: ${c_closest_images_top_k}/g" $dataset/config.yaml
    sed -i "s/use_gt_closest_images_pruning: .*/use_gt_closest_images_pruning: ${c_use_gt_closest_images_pruning}/g" $dataset/config.yaml
    sed -i "s/resectioning_config: .*/resectioning_config: ${c_resectioning_config}/g" $dataset/config.yaml
    sed -i "s/reconstruction_counter: .*/reconstruction_counter: ${c_reconstruction_counter}/g" $dataset/config.yaml

	grep "use_image_matching_classifier:" $dataset/config.yaml
	grep "use_weighted_resectioning:" $dataset/config.yaml
	grep "use_gt_matches:" $dataset/config.yaml
	grep "use_gt_selective_matches:" $dataset/config.yaml
	grep "use_weighted_feature_matches:" $dataset/config.yaml
	grep "use_image_matching_thresholding:" $dataset/config.yaml
	grep "use_shortest_path_pruning:" $dataset/config.yaml
	grep "image_matching_classifier_threshold:" $dataset/config.yaml
	grep "use_closest_images_pruning:" $dataset/config.yaml
	grep "closest_images_top_k:" $dataset/config.yaml
	grep "use_gt_closest_images_pruning:" $dataset/config.yaml
	grep "resectioning_config:" $dataset/config.yaml
	grep "reconstruction_counter:" $dataset/config.yaml

	# if [ "$count" -eq -1 ]
	# then

	# ./bin/opensfm extract_metadata $dataset
	# ./bin/opensfm detect_features $dataset
	# ./bin/opensfm evaluate_vt_rankings $dataset
	# ./bin/opensfm match_features $dataset
	# ./bin/opensfm create_tracks $dataset

	# 	# ./bin/opensfm classify_features $dataset
	# 	# ./bin/opensfm match_fm_classifier_features $dataset
	# 	# ./bin/opensfm calculate_features $dataset
	# 	# ./bin/opensfm classify_images $dataset
	# fi

	if [ "$mode" == "reconstruction" ];then
		# ./bin/opensfm yan $dataset
		./bin/opensfm create_tracks $dataset
		./bin/opensfm create_tracks_classifier $dataset
		./bin/opensfm reconstruct $dataset
	fi

	echo "************************************************************************************************************"
	echo "************************************************************************************************************"
	# (( count++ ));
done

# ./bin/opensfm convert_colmap $dataset
# ./bin/opensfm validate_results $dataset

# for run in "${runs[@]}"
# do
# 	cmd="google-chrome"
# 	url="http://localhost:8888/viewer/reconstruction.html#file="$dataset
# 	echo $cmd" "$url"/"$run >> $dataset/chrome-commands.sh
# done
