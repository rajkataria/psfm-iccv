declare -a n_estimators=("50")
declare -a max_depth=("6")
declare -a image_match_classifier_max_match=("50")
declare -a image_match_classifier_min_match=("15")
declare -a classifier=("NN" "GCN")
declare -a use_small_weights=("yes" "no")
declare -a use_all_training_data=("yes" "no")

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
							echo "**********************************************************************************";
							nohup python2.7 -u matching_classifiers.py \
								--opensfm_path ./OpenSfM-0.2.0/ \
								--max_depth $d \
								--n_estimators $n \
								--image_match_classifier_min_match $m \
								--image_match_classifier_max_match $x \
								--classifier $c \
								--use_small_weights $s \
								--use_all_training_data $a \
								2>&1 > output-$c-use-all-training-data-$a-use-small-weights-$s-min-match-$m-max-match-$x.out &
							echo "Running '$c' classifier with the options use_all_training_data=$a image_match_classifier_min_match=$m image_match_classifier_max_match=$x use_small_weights=$s";
						done;
					done;
				done;
			done;
		done;
	done;
done
echo "**********************************************************************************";
