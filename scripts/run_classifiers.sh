# # Baseline: all important features
# declare -A run1=(
# 	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
# 	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
# 	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='yes'
# 	[convnet_use_matches_map]='yes'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='yes'
# 	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
# 	[convnet_resnet_model]='18'										[convnet_loss]='ce'																			
# 	[train_on_val]='no'												[convnet_use_rmatches_secondary_motion_map]='yes'
# 	)

# # Baseline: removed nrmm
# declare -A run2=(
# 	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
# 	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
# 	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='no'
# 	[convnet_use_matches_map]='yes'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='yes'
# 	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
# 	[convnet_resnet_model]='18'										[convnet_loss]='ce'																			
# 	[train_on_val]='no'												[convnet_use_rmatches_secondary_motion_map]='yes'
# 	)

# # Baseline: removed rmm
# declare -A run3=(
# 	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
# 	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
# 	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='yes'
# 	[convnet_use_matches_map]='yes'									[convnet_use_rmatches_map]='no'										[convnet_use_photometric_error_maps]='yes'
# 	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
# 	[convnet_resnet_model]='18'										[convnet_loss]='ce'																			
# 	[train_on_val]='no'												[convnet_use_rmatches_secondary_motion_map]='yes'
# 	)

# # Baseline: removed mm
# declare -A run4=(
# 	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
# 	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
# 	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='yes'
# 	[convnet_use_matches_map]='no'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='yes'
# 	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
# 	[convnet_resnet_model]='18'										[convnet_loss]='ce'																			
# 	[train_on_val]='no'												[convnet_use_rmatches_secondary_motion_map]='yes'
# 	)

# # Baseline: removed pemms
# declare -A run5=(
# 	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
# 	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
# 	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='yes'
# 	[convnet_use_matches_map]='yes'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='no'
# 	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
# 	[convnet_resnet_model]='18'										[convnet_loss]='ce'																			
# 	[train_on_val]='no'												[convnet_use_rmatches_secondary_motion_map]='yes'
# 	)

# # Baseline: removed rmsmm
# declare -A run6=(
# 	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
# 	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
# 	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='yes'
# 	[convnet_use_matches_map]='yes'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='yes'
# 	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
# 	[convnet_resnet_model]='18'										[convnet_loss]='ce'																			
# 	[train_on_val]='no'												[convnet_use_rmatches_secondary_motion_map]='no'
# 	)

# # Baseline: all important features + resnet34
# declare -A run7=(
# 	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
# 	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
# 	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='yes'
# 	[convnet_use_matches_map]='yes'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='yes'
# 	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
# 	[convnet_resnet_model]='34'										[convnet_loss]='ce'																			
# 	[train_on_val]='no'												[convnet_use_rmatches_secondary_motion_map]='yes'
# 	)

# # Baseline: removed nrmm + resnet34
# declare -A run8=(
# 	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
# 	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
# 	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='no'
# 	[convnet_use_matches_map]='yes'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='yes'
# 	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
# 	[convnet_resnet_model]='34'										[convnet_loss]='ce'																			
# 	[train_on_val]='no'												[convnet_use_rmatches_secondary_motion_map]='yes'
# 	)

# # Baseline: removed rmm + resnet34
# declare -A run9=(
# 	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
# 	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
# 	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='yes'
# 	[convnet_use_matches_map]='yes'									[convnet_use_rmatches_map]='no'										[convnet_use_photometric_error_maps]='yes'
# 	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
# 	[convnet_resnet_model]='34'										[convnet_loss]='ce'																			
# 	[train_on_val]='no'												[convnet_use_rmatches_secondary_motion_map]='yes'
# 	)

# # Baseline: removed mm + resnet34
# declare -A run10=(
# 	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
# 	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
# 	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='yes'
# 	[convnet_use_matches_map]='no'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='yes'
# 	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
# 	[convnet_resnet_model]='34'										[convnet_loss]='ce'																			
# 	[train_on_val]='no'												[convnet_use_rmatches_secondary_motion_map]='yes'
# 	)

# # Baseline: removed pemms + resnet34
# declare -A run11=(
# 	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
# 	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
# 	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='yes'
# 	[convnet_use_matches_map]='yes'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='no'
# 	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
# 	[convnet_resnet_model]='34'										[convnet_loss]='ce'																			
# 	[train_on_val]='no'												[convnet_use_rmatches_secondary_motion_map]='yes'
# 	)

# # Baseline: removed rmsmm + resnet34
# declare -A run12=(
# 	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
# 	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
# 	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='yes'
# 	[convnet_use_matches_map]='yes'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='yes'
# 	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
# 	[convnet_resnet_model]='34'										[convnet_loss]='ce'																			
# 	[train_on_val]='no'												[convnet_use_rmatches_secondary_motion_map]='no'
# 	)

# # Baseline: all important features + train on val (used for final run)
# declare -A run13=(
# 	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
# 	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
# 	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='yes'
# 	[convnet_use_matches_map]='yes'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='yes'
# 	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
# 	[convnet_resnet_model]='18'										[convnet_loss]='ce'																			
# 	[train_on_val]='yes'											[convnet_use_rmatches_secondary_motion_map]='yes'
# 	)

# # Baseline: all important features + train on val (used for final run) + resnet34
# declare -A run14=(
# 	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
# 	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
# 	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='yes'
# 	[convnet_use_matches_map]='yes'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='yes'
# 	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
# 	[convnet_resnet_model]='34'										[convnet_loss]='ce'																			
# 	[train_on_val]='yes'											[convnet_use_rmatches_secondary_motion_map]='yes'
# 	)

# all_runs=(run1 run2 run3 run4 run5 run6 run13)
# all_runs=(run1)
# all_runs=(run1 run6)

# Baseline: all important features
declare -A run1=(
	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='5000'  							[classifier]='CONVNET'
	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='yes'
	[convnet_use_matches_map]='yes'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='yes'
	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
	[convnet_resnet_model]='18'										[convnet_loss]='ce'																			
	[train_on_val]='no'												[convnet_use_rmatches_secondary_motion_map]='yes'
	)

# Run: RMM
declare -A run2=(
	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='5000'    						[classifier]='CONVNET'
	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='no'								[convnet_use_non_rmatches_map]='no'
	[convnet_use_matches_map]='no'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='no'
	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
	[convnet_resnet_model]='18'										[convnet_loss]='ce'																			
	[train_on_val]='no'												[convnet_use_rmatches_secondary_motion_map]='no'
	)

# Run: RMM + NRMM
declare -A run3=(
	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='5000'  							[classifier]='CONVNET'
	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='no'								[convnet_use_non_rmatches_map]='yes'
	[convnet_use_matches_map]='no'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='no'
	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
	[convnet_resnet_model]='18'										[convnet_loss]='ce'																			
	[train_on_val]='no'												[convnet_use_rmatches_secondary_motion_map]='no'
	)

# Run: RMM + PEMs
declare -A run4=(
	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='5000'  							[classifier]='CONVNET'
	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='no'								[convnet_use_non_rmatches_map]='no'
	[convnet_use_matches_map]='no'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='yes'
	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
	[convnet_resnet_model]='18'										[convnet_loss]='ce'																			
	[train_on_val]='no'												[convnet_use_rmatches_secondary_motion_map]='no'
	)

# Run: PEMs
declare -A run5=(
	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='5000'  							[classifier]='CONVNET'
	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='no'								[convnet_use_non_rmatches_map]='no'
	[convnet_use_matches_map]='no'									[convnet_use_rmatches_map]='no'										[convnet_use_photometric_error_maps]='yes'
	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
	[convnet_resnet_model]='18'										[convnet_loss]='ce'																			
	[train_on_val]='no'												[convnet_use_rmatches_secondary_motion_map]='no'
	)

# Run: RMM + NRMM + PEMs
declare -A run6=(
	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='5000'  							[classifier]='CONVNET'
	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='no'								[convnet_use_non_rmatches_map]='yes'
	[convnet_use_matches_map]='no'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='yes'
	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
	[convnet_resnet_model]='18'										[convnet_loss]='ce'																			
	[train_on_val]='no'												[convnet_use_rmatches_secondary_motion_map]='no'
	)

# Run: RMM + NRMM + PEMs  + Triplet Loss
declare -A run7=(
	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='5000'  							[classifier]='CONVNET'
	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='no'								[convnet_use_non_rmatches_map]='yes'
	[convnet_use_matches_map]='no'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='yes'
	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
	[convnet_resnet_model]='18'										[convnet_loss]='t'																			
	[train_on_val]='no'												[convnet_use_rmatches_secondary_motion_map]='no'
	)

# all_runs=(run2 run3 run4 run5 run6 run1)

all_runs=(run5 run1)
# all_runs=(run7)

for run_name in "${all_runs[@]}"; do
	echo "**********************************************************************************";
	echo "**********************************************************************************";
	declare -n run_ref="$run_name"

    c_n_estimators="${run_ref[n_estimators]}"
    c_max_depth="${run_ref[max_depth]}"
    c_lr="${run_ref[lr]}"

    c_image_match_classifier_min_match="${run_ref[image_match_classifier_min_match]}"
    c_image_match_classifier_max_match="${run_ref[image_match_classifier_max_match]}"
    c_classifier="${run_ref[classifier]}"
    c_use_all_training_data="${run_ref[use_all_training_data]}"

    c_convnet_use_matches_map="${run_ref[convnet_use_matches_map]}"
    c_convnet_use_rmatches_map="${run_ref[convnet_use_rmatches_map]}"
    c_convnet_use_photometric_error_maps="${run_ref[convnet_use_photometric_error_maps]}"

    c_convnet_use_feature_match_map="${run_ref[convnet_use_feature_match_map]}"
    c_convnet_use_non_rmatches_map="${run_ref[convnet_use_non_rmatches_map]}"
    c_convnet_use_rmatches_secondary_motion_map="${run_ref[convnet_use_rmatches_secondary_motion_map]}"
    c_convnet_use_track_map="${run_ref[convnet_use_track_map]}"
    c_convnet_use_images="${run_ref[convnet_use_images]}"
    c_convnet_features="${run_ref[convnet_features]}"
    c_convnet_resnet_model="${run_ref[convnet_resnet_model]}"
    c_convnet_loss="${run_ref[convnet_loss]}"
    c_train_on_val="${run_ref[train_on_val]}"

	if [ "$c_convnet_resnet_model" == "18" ];then
		if [ "$c_convnet_loss" == "ce" ];then
	   		batchsize=96
		else
			batchsize=48
		fi
	elif [ "$c_convnet_resnet_model" == "34" ]; then
		if [ "$c_convnet_use_images" == "yes" ]; then
			batchsize=64
		else
   			batchsize=64
   		fi
	elif [ "$c_convnet_resnet_model" == "50" ]; then
	   	batchsize=24
	elif [ "$c_convnet_resnet_model" == "101" ]; then
	   	batchsize=8
	elif [ "$c_convnet_resnet_model" == "152" ]; then
	   	batchsize=8
	else
	   	echo "Unknown parameter"
	fi

	if [ "$c_convnet_loss" == "ce" ];then
	   	triplet_sampling_strategy="n"
	else
		triplet_sampling_strategy="r"
	fi

	echo "Running '$c_classifier' classifier with the options \
		use_all_training_data=$c_use_all_training_data \
		image_match_classifier_min_match=$c_image_match_classifier_min_match \
		image_match_classifier_max_match=$c_image_match_classifier_max_match \
		convnet_lr=$c_lr \
		convnet_features=$c_convnet_features \
		convnet_resnet_model=$c_convnet_resnet_model \
		convnet_loss=$c_convnet_loss \
		convnet_use_images=$c_convnet_use_images \
		convnet_triplet_sampling_strategy=$triplet_sampling_strategy \
		convnet_batch_size=$batchsize \
		convnet_use_feature_match_map=$c_convnet_use_feature_match_map \
		convnet_use_track_map=$c_convnet_use_track_map \
		convnet_use_non_rmatches_map=$c_convnet_use_non_rmatches_map \
		convnet_use_rmatches_secondary_motion_map=$c_convnet_use_rmatches_secondary_motion_map \
		convnet_use_rmatches_map=$c_convnet_use_rmatches_map \
		convnet_use_matches_map=$c_convnet_use_matches_map \
		convnet_use_photometric_error_maps=$c_convnet_use_photometric_error_maps \
		train_on_val=$c_train_on_val";

	python2.7 -u matching_classifiers.py \
		--opensfm_path ./OpenSfM-0.2.0/ \
		--max_depth $c_max_depth \
		--convnet_lr $c_lr \
		--n_estimators $c_n_estimators \
		--image_match_classifier_min_match $c_image_match_classifier_min_match \
		--image_match_classifier_max_match $c_image_match_classifier_max_match \
		--classifier $c_classifier \
		--use_all_training_data $c_use_all_training_data \
		--train_on_val $c_train_on_val \
		--convnet_features $c_convnet_features \
		--convnet_resnet_model $c_convnet_resnet_model \
		--convnet_loss $c_convnet_loss \
		--convnet_use_images $c_convnet_use_images \
		--convnet_triplet_sampling_strategy $triplet_sampling_strategy \
		--convnet_batch_size $batchsize \
		--convnet_use_feature_match_map $c_convnet_use_feature_match_map \
		--convnet_use_track_map $c_convnet_use_track_map \
		--convnet_use_non_rmatches_map $c_convnet_use_non_rmatches_map \
		--convnet_use_rmatches_secondary_motion_map $c_convnet_use_rmatches_secondary_motion_map \
		--convnet_use_matches_map $c_convnet_use_matches_map \
		--convnet_use_rmatches_map $c_convnet_use_rmatches_map \
		--convnet_use_photometric_error_maps $c_convnet_use_photometric_error_maps \
		2>&1 > output-$c_classifier-lr-$c_lr-use-all-training-data-$c_use_all_training_data-min-match-$c_image_match_classifier_min_match-max-match-$c_image_match_classifier_max_match-features-$c_convnet_features-model-$c_convnet_resnet_model-loss-$c_convnet_loss-ss-$triplet_sampling_strategy-use_images-$c_convnet_use_images-use_fmm-$c_convnet_use_feature_match_map-use_tm-$c_convnet_use_track_map-use_nrmm-$c_convnet_use_non_rmatches_map-use_rmsmm-$c_convnet_use_rmatches_secondary_motion_map-train_on_val-$c_train_on_val-use_rmm-$c_convnet_use_rmatches_map-use_mm-$c_convnet_use_matches_map-use_pems-$c_convnet_use_photometric_error_maps.out
	
    echo "************************************************************************************************************"
	echo "************************************************************************************************************"
done
