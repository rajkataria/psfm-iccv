declare -a n_estimators=("50")
declare -a max_depth=("6")
declare -a image_match_classifier_max_match=("50")
declare -a image_match_classifier_min_match=("15")
declare -a classifier=("NN" "GCN")
declare -a use_small_weights=("yes" "no")
declare -a use_all_training_data=("yes" "no")

# declare inputs_folder="scoring-inputs-12-09-2018-11-05-pm"

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
							# nohup python -u calculate_scoring_matrix.py \
							# 	-u ./$inputs_folder/Users.txt \
							# 	-r ./$inputs_folder/reviewers.csv \
							# 	-t ./$inputs_folder/ReviewerTpmsScores_CVPR2019.csv \
							# 	-p ./$inputs_folder/Papers.csv \
							# 	-q ./$inputs_folder/quotas.csv \
							# 	-s ./$inputs_folder/ReviewerSuggestions.txt \
							# 	-c ./$inputs_folder/ReviewerConflicts.txt \
							# 	-n 3 \
							# 	-g $g \
							# 	-w_t $t \
							# 	-w_a $a \
							# 	-w_s $s \
							# 	-w_e $e \
							# 	-o $c \
							# 	--cached_folder ./output-w_t-$t-w_a-$a-w_s-$s-w_e-$e-g-$g-n-3-config-$c/ > output-w_t-$t-w_a-$a-w_s-$s-w_e-$e-g-$g-n-3-config-$c.log 2>&1 &
							echo "Running '$c' classifier with the options use_all_training_data=$a image_match_classifier_min_match=$m image_match_classifier_max_match=$x use_small_weights=$s";
						done;
					done;
				done;
			done;
		done;
	done;
done
echo "**********************************************************************************";
