# declare -a n_estimators=("50")
# declare -a max_depth=("6")
# declare -a image_match_classifier_max_match=("50")
# declare -a image_match_classifier_min_match=("15")
# # declare -a image_match_classifier_max_match=("10000")
# # declare -a image_match_classifier_min_match=("50")
# declare -a classifier=("CONVNET")
# declare -a train_on_val=("no")
# # declare -a train_on_val=("yes")

# declare -a use_all_training_data=("yes")
# # declare -a use_all_training_data=("no")

# declare -a convnet_use_feature_match_map=("yes")
# declare -a convnet_use_non_rmatches_map=("yes")
# declare -a convnet_use_track_map=("no")

# declare -a convnet_use_images=("no")

# declare -a convnet_features=("RM")
# # declare -a convnet_features=("RM+TE")
# # declare -a convnet_features=("RM+NBVS")
# # declare -a convnet_features=("RM+TE+NBVS")

# declare -a convnet_resnet_model=("18")
# # declare -a convnet_resnet_model=("50")
# # declare -a convnet_resnet_model=("34")
# # declare -a convnet_resnet_model=("18")
# # declare -a convnet_resnet_model=("18" "34" "50" "101" "152")



# declare -a convnet_loss=("ce")
# # declare -a convnet_loss=("t")


# Baseline: all important features
declare -A run1=(
	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='yes'
	[convnet_use_matches_map]='yes'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='yes'
	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
	[convnet_resnet_model]='18'										[convnet_loss]='ce'																			
	[train_on_val]='no'
	)

# Baseline: removed nrmm
declare -A run2=(
	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='no'
	[convnet_use_matches_map]='yes'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='yes'
	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
	[convnet_resnet_model]='18'										[convnet_loss]='ce'																			
	[train_on_val]='no'
	)

# Baseline: removed rmm
declare -A run3=(
	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='yes'
	[convnet_use_matches_map]='yes'									[convnet_use_rmatches_map]='no'									[convnet_use_photometric_error_maps]='yes'
	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
	[convnet_resnet_model]='18'										[convnet_loss]='ce'																			
	[train_on_val]='no'
	)

# Baseline: removed mm
declare -A run4=(
	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='yes'
	[convnet_use_matches_map]='no'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='yes'
	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
	[convnet_resnet_model]='18'										[convnet_loss]='ce'																			
	[train_on_val]='no'
	)

# Baseline: removed pemms
declare -A run5=(
	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='yes'
	[convnet_use_matches_map]='yes'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='no'
	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
	[convnet_resnet_model]='18'										[convnet_loss]='ce'																			
	[train_on_val]='no'
	)

# Baseline: all important features + train on val (used for final run)
declare -A run6=(
	[n_estimators]='50'                    							[max_depth]='6'    													[lr]='0.01'
	[image_match_classifier_min_match]='15'                    		[image_match_classifier_max_match]='50'    							[classifier]='CONVNET'
	[use_all_training_data]='yes'									[convnet_use_feature_match_map]='yes'								[convnet_use_non_rmatches_map]='yes'
	[convnet_use_matches_map]='yes'									[convnet_use_rmatches_map]='yes'									[convnet_use_photometric_error_maps]='yes'
	[convnet_use_track_map]='no'									[convnet_use_images]='no'											[convnet_features]='RM'
	[convnet_resnet_model]='18'										[convnet_loss]='ce'																			
	[train_on_val]='yes'
	)

# all_runs=(run1 run2 run3 run4 run5 run6)
all_runs=(run1)

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
    c_convnet_use_track_map="${run_ref[convnet_use_track_map]}"
    c_convnet_use_images="${run_ref[convnet_use_images]}"
    c_convnet_features="${run_ref[convnet_features]}"
    c_convnet_resnet_model="${run_ref[convnet_resnet_model]}"
    c_convnet_loss="${run_ref[convnet_loss]}"
    c_train_on_val="${run_ref[train_on_val]}"

	if [ "$c_convnet_resnet_model" == "18" ];then
		batchsize=96
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
		--convnet_use_matches_map $c_convnet_use_matches_map \
		--convnet_use_rmatches_map $c_convnet_use_rmatches_map \
		--convnet_use_photometric_error_maps $c_convnet_use_photometric_error_maps \
		2>&1 > output-$c_classifier-lr-$c_lr-use-all-training-data-$c_use_all_training_data-min-match-$c_image_match_classifier_min_match-max-match-$c_image_match_classifier_max_match-features-$c_convnet_features-model-$c_convnet_resnet_model-loss-$c_convnet_loss-ss-$triplet_sampling_strategy-use_images-$c_convnet_use_images-use_fmm-$c_convnet_use_feature_match_map-use_tm-$c_convnet_use_track_map-use_nrmm-$c_convnet_use_non_rmatches_map-train_on_val-$c_train_on_val-use_rmm-$c_convnet_use_rmatches_map-use_mm-$c_convnet_use_matches_map-use_pems-$c_convnet_use_photometric_error_maps.out
	
    echo "************************************************************************************************************"
	echo "************************************************************************************************************"
done
# for n in "${n_estimators[@]}"
# 	do
# 	for d in "${max_depth[@]}"
# 		do
# 		for x in "${image_match_classifier_max_match[@]}"
# 			do
# 			for m in "${image_match_classifier_min_match[@]}"
# 				do
# 				for c in "${classifier[@]}"
# 					do
# 					for v in "${train_on_val[@]}"
# 						do
# 						for a in "${use_all_training_data[@]}"
# 							do
# 							for f in "${convnet_features[@]}"
# 								do
# 								for o in "${convnet_resnet_model[@]}"
# 									do
# 									for l in "${convnet_loss[@]}"
# 										do
# 										for i in "${convnet_use_images[@]}"
# 											do
# 											for fmm in "${convnet_use_feature_match_map[@]}"
# 												do
# 												for tm in "${convnet_use_track_map[@]}"
# 													do
# 													for nrmm in "${convnet_use_non_rmatches_map[@]}"
# 														do
# 														echo "**********************************************************************************";

# 														if [ "$o" == "18" ];then
# 																batchsize=128 # works for training, but not testing (224), ce
# 																# batchsize=256  # works for training and testing (112), ce
# 														   	# batchsize=72
# 														elif [ "$o" == "34" ]; then
# 															if [ "$i" == "yes" ]; then
# 																batchsize=64
# 															else
# 														   		# batchsize=256
# 														   		batchsize=64
# 														   	fi
# 														elif [ "$o" == "50" ]; then
# 															# Do not change
# 														   	batchsize=24
# 														elif [ "$o" == "101" ]; then
# 														   	batchsize=8
# 														elif [ "$o" == "152" ]; then
# 														   	batchsize=8
# 														else
# 														   	echo "Unknown parameter"
# 														fi

# 														if [ "$l" == "ce" ];then
# 														   	triplet_sampling_strategy="n"
# 														else
# 																triplet_sampling_strategy="r"
# 														fi
# 														# echo $l
# 														# echo $batchsize
# 														# echo $triplet_sampling_strategy
# 														# exit 1
# 														nohup python2.7 -u matching_classifiers.py \
# 															--opensfm_path ./OpenSfM-0.2.0/ \
# 															--max_depth $d \
# 															--n_estimators $n \
# 															--image_match_classifier_min_match $m \
# 															--image_match_classifier_max_match $x \
# 															--classifier $c \
# 															--use_all_training_data $a \
# 															--train_on_val $v \
# 															--convnet_features $f \
# 															--convnet_resnet_model $o \
# 															--convnet_loss $l \
# 															--convnet_use_images $i \
# 															--convnet_triplet_sampling_strategy $triplet_sampling_strategy \
# 															--convnet_batch_size $batchsize \
# 															--convnet_use_feature_match_map $fmm \
# 															--convnet_use_track_map $tm \
# 															--convnet_use_non_rmatches_map $nrmm
# 															2>&1 > output-$c-use-all-training-data-$a-min-match-$m-max-match-$x-features-$f-model-$o-loss-$l-ss-$triplet_sampling_strategy-use_images-$i-use_fmm-$fmm-use_tm-$tm-use_nrmm-$nrmm-train_on_val-$v-filtered.out &
# 														echo "Running '$c' classifier with the options \
# 															use_all_training_data=$a \
# 															image_match_classifier_min_match=$m \
# 															image_match_classifier_max_match=$x \
# 															convnet_features=$f \
# 															convnet_resnet_model=$o \
# 															convnet_loss=$l \
# 															convnet_use_images=$i \
# 															convnet_triplet_sampling_strategy=$triplet_sampling_strategy \
# 															convnet_batch_size=$batchsize \
# 															convnet_use_feature_match_map=$fmm \
# 															convnet_use_track_map=$tm \
# 															convnet_use_non_rmatches_map=$nrmm \
# 															train_on_val=$v";
# 													done;
# 												done;
# 											done;
# 										done;
# 									done;
# 								done;
# 							done;
# 						done;
# 					done;
# 				done;
# 			done;
# 		done;
# 	done;
# done
# echo "**********************************************************************************";
