declare -a n_estimators=("50")
declare -a max_depth=("6")
declare -a image_match_classifier_max_match=("50")
declare -a image_match_classifier_min_match=("15")
# declare -a image_match_classifier_max_match=("10000")
# declare -a image_match_classifier_min_match=("50")
declare -a classifier=("CONVNET")
# declare -a use_small_weights=("yes")
declare -a use_small_weights=("no")

declare -a use_all_training_data=("yes")
# declare -a use_all_training_data=("no")

declare -a convnet_use_images=("no")

declare -a convnet_features=("RM")
# declare -a convnet_features=("RM+TE")
# declare -a convnet_features=("RM+NBVS")
# declare -a convnet_features=("RM+TE+NBVS")

# declare -a convnet_resnet_model=("18")
# declare -a convnet_resnet_model=("50")
declare -a convnet_resnet_model=("34")
# declare -a convnet_resnet_model=("18" "34" "50" "101" "152")



declare -a convnet_loss=("ce")
# declare -a convnet_loss=("t")
# declare -a convnet_loss=("cet")


for n in "${n_estimators[@]}"
	do
	for d in "${max_depth[@]}"
		do
		for x in "${image_match_classifier_max_match[@]}"
			do
			for m in "${image_match_classifier_min_match[@]}"
				do
				for c in "${classifier[@]}"
					do
					for s in "${use_small_weights[@]}"
						do
						for a in "${use_all_training_data[@]}"
							do
							for f in "${convnet_features[@]}"
								do
								for o in "${convnet_resnet_model[@]}"
									do
									for l in "${convnet_loss[@]}"
										do
										for i in "${convnet_use_images[@]}"
											do
											echo "**********************************************************************************";

											if [ "$o" == "18" ];then
													# batchsize=128 # works for training, but not testing (224), ce
													batchsize=256  # works for training and testing (112), ce
											   	# batchsize=72
											elif [ "$o" == "34" ]; then
												if [ "$i" == "yes" ]; then
													batchsize=64
												else
											   		# batchsize=256
											   		batchsize=64
											   	fi
											elif [ "$o" == "50" ]; then
												# Do not change
											   	batchsize=24
											elif [ "$o" == "101" ]; then
											   	batchsize=8
											elif [ "$o" == "152" ]; then
											   	batchsize=8
											else
											   	echo "Unknown parameter"
											fi

											if [ "$l" == "ce" ];then
											   	triplet_sampling_strategy="n"
											else
												triplet_sampling_strategy="u"
											fi
											# echo $l
											# echo $batchsize
											# echo $triplet_sampling_strategy
											# exit 1
											nohup python2.7 -u matching_classifiers.py \
												--opensfm_path ./OpenSfM-0.2.0/ \
												--max_depth $d \
												--n_estimators $n \
												--image_match_classifier_min_match $m \
												--image_match_classifier_max_match $x \
												--classifier $c \
												--use_small_weights $s \
												--use_all_training_data $a \
												--convnet_features $f \
												--convnet_resnet_model $o \
												--convnet_loss $l \
												--convnet_use_images $i \
												--convnet_triplet_sampling_strategy $triplet_sampling_strategy \
												--convnet_batch_size $batchsize \
												2>&1 > output-$c-use-all-training-data-$a-use-small-weights-$s-min-match-$m-max-match-$x-features-$f-model-$o-loss-$l-ss-$triplet_sampling_strategy-use_images-$i.out &
											echo "Running '$c' classifier with the options \
												use_all_training_data=$a \
												image_match_classifier_min_match=$m \
												image_match_classifier_max_match=$x \
												use_small_weights=$s \
												convnet_features=$f \
												convnet_resnet_model=$o \
												convnet_loss=$l \
												convnet_use_images=$i \
												convnet_triplet_sampling_strategy=$triplet_sampling_strategy \
												convnet_batch_size=$batchsize";
										done;
									done;
								done;
							done;
						done;
					done;
				done;
			done;
		done;
	done;
done
echo "**********************************************************************************";
