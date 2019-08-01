dataset=$1
c_reconstruction_counter=${2:-0}

mode="calculate_features"
# mode="reconstruction"

echo "############################################################################################################"
echo "############################################################################################################"
echo "########################################### Starting experiments ###########################################"
echo "############################################################################################################"
echo "############################################################################################################"

# Baseline + original resectioning
declare -A run99=(
	[use_image_matching_classifier]='false'
	[use_weighted_resectioning]='original'										         	[use_weighted_feature_matches]='false'
	[use_image_matching_thresholding]='false'   [use_distance_threshold]='false'       		[image_matching_classifier_threshold]='0.5'
	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'					[use_gt_closest_images_pruning]='false'
	[resectioning_config]='NA'					[distance_threshold_value]='0.6'
    [mds_k_closest_images_percentage]='0.15'    [mds_k_closest_images_min]='7'              [mds_k_closest_images_max]='12'
	)

# Baseline + colmap
declare -A run100=(
	[use_image_matching_classifier]='false'
	[use_weighted_resectioning]='colmap'        											[use_weighted_feature_matches]='false'
	[use_image_matching_thresholding]='false'   [use_distance_threshold]='false'       		[image_matching_classifier_threshold]='0.5'
	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'					[use_gt_closest_images_pruning]='false'
	[resectioning_config]='NA'					[distance_threshold_value]='0.6'
    [mds_k_closest_images_percentage]='0.15'    [mds_k_closest_images_min]='7'              [mds_k_closest_images_max]='12'
	)

# MDS + colmap + udt (7-12 at 15%)
declare -A run101=(
    [use_image_matching_classifier]='false'
    [use_weighted_resectioning]='colmap'                                                    [use_weighted_feature_matches]='false'
    [use_image_matching_thresholding]='false'   [use_distance_threshold]='true'             [image_matching_classifier_threshold]='0.5'
    [use_closest_images_pruning]='false'        [closest_images_top_k]='H'                  [use_gt_closest_images_pruning]='false'
    [resectioning_config]='NA'                  [distance_threshold_value]='0.6'
    [mds_k_closest_images_percentage]='0.15'    [mds_k_closest_images_min]='7'              [mds_k_closest_images_max]='12'
    )

# MDS + colmap + udt (6-14 at 15%)
declare -A run102=(
    [use_image_matching_classifier]='false'
    [use_weighted_resectioning]='colmap'                                                    [use_weighted_feature_matches]='false'
    [use_image_matching_thresholding]='false'   [use_distance_threshold]='true'             [image_matching_classifier_threshold]='0.5'
    [use_closest_images_pruning]='false'        [closest_images_top_k]='H'                  [use_gt_closest_images_pruning]='false'
    [resectioning_config]='NA'                  [distance_threshold_value]='0.6'
    [mds_k_closest_images_percentage]='0.15'    [mds_k_closest_images_min]='6'              [mds_k_closest_images_max]='14'
    )

# MDS + colmap + udt (6-14 at 20%)
declare -A run103=(
    [use_image_matching_classifier]='false'
    [use_weighted_resectioning]='colmap'                                                    [use_weighted_feature_matches]='false'
    [use_image_matching_thresholding]='false'   [use_distance_threshold]='true'             [image_matching_classifier_threshold]='0.5'
    [use_closest_images_pruning]='false'        [closest_images_top_k]='H'                  [use_gt_closest_images_pruning]='false'
    [resectioning_config]='NA'                  [distance_threshold_value]='0.6'
    [mds_k_closest_images_percentage]='0.20'    [mds_k_closest_images_min]='6'              [mds_k_closest_images_max]='14'
    )

all_runs=(run99 run100 run101 run102 run103)

for j in `seq 0 0`; do
    if [ "$mode" == "calculate_features" ];then
        ./bin/opensfm create_tracks $dataset
        ./bin/opensfm calculate_features $dataset
    fi
done

for run_name in "${all_runs[@]}"; do
    declare -n run_ref="$run_name"
    
    c_use_image_matching_classifier="${run_ref[use_image_matching_classifier]}"
    c_use_weighted_resectioning="${run_ref[use_weighted_resectioning]}"
    c_use_weighted_feature_matches="${run_ref[use_weighted_feature_matches]}"
    c_use_image_matching_thresholding="${run_ref[use_image_matching_thresholding]}"
    c_use_distance_threshold="${run_ref[use_distance_threshold]}"
    c_image_matching_classifier_threshold="${run_ref[image_matching_classifier_threshold]}"
    c_use_closest_images_pruning="${run_ref[use_closest_images_pruning]}"
    c_closest_images_top_k="${run_ref[closest_images_top_k]}"
    c_use_gt_closest_images_pruning="${run_ref[use_gt_closest_images_pruning]}"
    c_resectioning_config="${run_ref[resectioning_config]}"
    c_distance_threshold_value="${run_ref[distance_threshold_value]}"

    c_mds_k_closest_images_percentage="${run_ref[mds_k_closest_images_percentage]}"
    c_mds_k_closest_images_min="${run_ref[mds_k_closest_images_min]}"
    c_mds_k_closest_images_max="${run_ref[mds_k_closest_images_max]}"
    
    
    sed -i "s/use_image_matching_classifier: .*/use_image_matching_classifier: ${c_use_image_matching_classifier}/g" $dataset/config.yaml
    sed -i "s/use_weighted_resectioning: .*/use_weighted_resectioning: ${c_use_weighted_resectioning}/g" $dataset/config.yaml
    sed -i "s/use_weighted_feature_matches: .*/use_weighted_feature_matches: ${c_use_weighted_feature_matches}/g" $dataset/config.yaml
    sed -i "s/use_image_matching_thresholding: .*/use_image_matching_thresholding: ${c_use_image_matching_thresholding}/g" $dataset/config.yaml
    sed -i "s/use_distance_threshold: .*/use_distance_threshold: ${c_use_distance_threshold}/g" $dataset/config.yaml
    sed -i "s/image_matching_classifier_threshold: .*/image_matching_classifier_threshold: ${c_image_matching_classifier_threshold}/g" $dataset/config.yaml
    sed -i "s/use_closest_images_pruning: .*/use_closest_images_pruning: ${c_use_closest_images_pruning}/g" $dataset/config.yaml
    sed -i "s/closest_images_top_k: .*/closest_images_top_k: ${c_closest_images_top_k}/g" $dataset/config.yaml
    sed -i "s/use_gt_closest_images_pruning: .*/use_gt_closest_images_pruning: ${c_use_gt_closest_images_pruning}/g" $dataset/config.yaml
    sed -i "s/resectioning_config: .*/resectioning_config: ${c_resectioning_config}/g" $dataset/config.yaml
    sed -i "s/reconstruction_counter: .*/reconstruction_counter: ${c_reconstruction_counter}/g" $dataset/config.yaml
    sed -i "s/distance_threshold_value: .*/distance_threshold_value: ${c_distance_threshold_value}/g" $dataset/config.yaml
    sed -i "s/mds_k_closest_images_percentage: .*/mds_k_closest_images_percentage: ${c_mds_k_closest_images_percentage}/g" $dataset/config.yaml
    sed -i "s/mds_k_closest_images_min: .*/mds_k_closest_images_min: ${c_mds_k_closest_images_min}/g" $dataset/config.yaml
    sed -i "s/mds_k_closest_images_max: .*/mds_k_closest_images_max: ${c_mds_k_closest_images_max}/g" $dataset/config.yaml

	grep "use_image_matching_classifier:" $dataset/config.yaml
	grep "use_weighted_resectioning:" $dataset/config.yaml
	grep "use_weighted_feature_matches:" $dataset/config.yaml
	grep "use_image_matching_thresholding:" $dataset/config.yaml
	grep "use_distance_threshold:" $dataset/config.yaml
	grep "image_matching_classifier_threshold:" $dataset/config.yaml
	grep "use_closest_images_pruning:" $dataset/config.yaml
	grep "closest_images_top_k:" $dataset/config.yaml
	grep "use_gt_closest_images_pruning:" $dataset/config.yaml
	grep "resectioning_config:" $dataset/config.yaml
	grep "reconstruction_counter:" $dataset/config.yaml
	grep "distance_threshold_value:" $dataset/config.yaml
    grep "mds_k_closest_images_percentage:" $dataset/config.yaml
    grep "mds_k_closest_images_min:" $dataset/config.yaml
    grep "mds_k_closest_images_max:" $dataset/config.yaml

	if [ "$mode" == "reconstruction" ];then
		./bin/opensfm create_tracks_classifier $dataset
		./bin/opensfm reconstruct $dataset
	fi

	echo "************************************************************************************************************"
	echo "************************************************************************************************************"
done

# ./bin/opensfm convert_colmap $dataset
./bin/opensfm validate_results $dataset
