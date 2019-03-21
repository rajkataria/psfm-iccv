dataset=$1
# runs=("reconstruction_gt.json")

echo "############################################################################################################"
echo "############################################################################################################"
echo "########################################### Starting experiments ###########################################"
echo "############################################################################################################"
echo "############################################################################################################"
# Baseline
sed -i 's/use_gt_matches: true/use_gt_matches: false/g' $dataset/config.yaml
sed -i 's/use_gt_selective_matches: true/use_gt_selective_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier: true/use_image_matching_classifier: false/g' $dataset/config.yaml
sed -i 's/use_weighted_resectioning: true/use_weighted_resectioning: false/g' $dataset/config.yaml
sed -i 's/use_colmap_resectioning: true/use_colmap_resectioning: false/g' $dataset/config.yaml
sed -i 's/use_weighted_feature_matches: true/use_weighted_feature_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_thresholding: true/use_image_matching_thresholding: false/g' $dataset/config.yaml
sed -i 's/use_shortest_path_pruning: true/use_shortest_path_pruning: false/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "use_weighted_resectioning" $dataset/config.yaml
grep "use_colmap_resectioning" $dataset/config.yaml
grep "use_gt_matches" $dataset/config.yaml
grep "use_gt_selective_matches" $dataset/config.yaml
grep "use_weighted_feature_matches" $dataset/config.yaml
grep "use_image_matching_thresholding" $dataset/config.yaml
grep "use_shortest_path_pruning" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"

# ./bin/opensfm extract_metadata $dataset
# ./bin/opensfm detect_features $dataset
# ./bin/opensfm evaluate_vt_rankings $dataset
# ./bin/opensfm match_features $dataset
# ./bin/opensfm classify_features $dataset
# ./bin/opensfm match_fm_classifier_features $dataset
# ./bin/opensfm calculate_features $dataset
# ./bin/opensfm classify_images $dataset
./bin/opensfm create_tracks $dataset
./bin/opensfm create_tracks_classifier $dataset
./bin/opensfm reconstruct $dataset
# runs+=('reconstruction-imc-False-wr-False-colmapr-False-gm-False-wfm-False-imt-False-spp-False.json')

# Using ground-truth image matches - classifier/thresholding won't matter (since the weight is either a 1 or a 0)
sed -i 's/use_gt_matches: false/use_gt_matches: true/g' $dataset/config.yaml
sed -i 's/use_gt_selective_matches: true/use_gt_selective_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier: true/use_image_matching_classifier: false/g' $dataset/config.yaml
sed -i 's/use_weighted_resectioning: true/use_weighted_resectioning: false/g' $dataset/config.yaml
sed -i 's/use_colmap_resectioning: true/use_colmap_resectioning: false/g' $dataset/config.yaml
sed -i 's/use_weighted_feature_matches: true/use_weighted_feature_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_thresholding: true/use_image_matching_thresholding: false/g' $dataset/config.yaml
sed -i 's/use_shortest_path_pruning: true/use_shortest_path_pruning: false/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "use_weighted_resectioning" $dataset/config.yaml
grep "use_colmap_resectioning" $dataset/config.yaml
grep "use_gt_matches" $dataset/config.yaml
grep "use_gt_selective_matches" $dataset/config.yaml
grep "use_weighted_feature_matches" $dataset/config.yaml
grep "use_image_matching_thresholding" $dataset/config.yaml
grep "use_shortest_path_pruning" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"

./bin/opensfm reconstruct $dataset

# Using ground-truth selective image matches - classifier/thresholding won't matter (since the weight is either a 1 or a 0)
sed -i 's/use_gt_matches: false/use_gt_matches: true/g' $dataset/config.yaml
sed -i 's/use_gt_selective_matches: false/use_gt_selective_matches: true/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier: true/use_image_matching_classifier: false/g' $dataset/config.yaml
sed -i 's/use_weighted_resectioning: true/use_weighted_resectioning: false/g' $dataset/config.yaml
sed -i 's/use_colmap_resectioning: true/use_colmap_resectioning: false/g' $dataset/config.yaml
sed -i 's/use_weighted_feature_matches: true/use_weighted_feature_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_thresholding: true/use_image_matching_thresholding: false/g' $dataset/config.yaml
sed -i 's/use_shortest_path_pruning: true/use_shortest_path_pruning: false/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "use_weighted_resectioning" $dataset/config.yaml
grep "use_colmap_resectioning" $dataset/config.yaml
grep "use_gt_matches" $dataset/config.yaml
grep "use_gt_selective_matches" $dataset/config.yaml
grep "use_weighted_feature_matches" $dataset/config.yaml
grep "use_image_matching_thresholding" $dataset/config.yaml
grep "use_shortest_path_pruning" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"

./bin/opensfm reconstruct $dataset

# Using ground-truth image matches (with pruning) - classifier/thresholding won't matter (since the weight is either a 1 or a 0)
sed -i 's/use_gt_matches: false/use_gt_matches: true/g' $dataset/config.yaml
sed -i 's/use_gt_selective_matches: true/use_gt_selective_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier: true/use_image_matching_classifier: false/g' $dataset/config.yaml
sed -i 's/use_weighted_resectioning: true/use_weighted_resectioning: false/g' $dataset/config.yaml
sed -i 's/use_colmap_resectioning: true/use_colmap_resectioning: false/g' $dataset/config.yaml
sed -i 's/use_weighted_feature_matches: true/use_weighted_feature_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_thresholding: true/use_image_matching_thresholding: false/g' $dataset/config.yaml
sed -i 's/use_shortest_path_pruning: false/use_shortest_path_pruning: true/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "use_weighted_resectioning" $dataset/config.yaml
grep "use_colmap_resectioning" $dataset/config.yaml
grep "use_gt_matches" $dataset/config.yaml
grep "use_gt_selective_matches" $dataset/config.yaml
grep "use_weighted_feature_matches" $dataset/config.yaml
grep "use_image_matching_thresholding" $dataset/config.yaml
grep "use_shortest_path_pruning" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"

./bin/opensfm reconstruct $dataset

# Use robust matches (with pruning)
sed -i 's/use_gt_matches: true/use_gt_matches: false/g' $dataset/config.yaml
sed -i 's/use_gt_selective_matches: true/use_gt_selective_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier: true/use_image_matching_classifier: false/g' $dataset/config.yaml
sed -i 's/use_weighted_resectioning: true/use_weighted_resectioning: false/g' $dataset/config.yaml
sed -i 's/use_colmap_resectioning: true/use_colmap_resectioning: false/g' $dataset/config.yaml
sed -i 's/use_weighted_feature_matches: true/use_weighted_feature_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_thresholding: true/use_image_matching_thresholding: false/g' $dataset/config.yaml
sed -i 's/use_shortest_path_pruning: false/use_shortest_path_pruning: true/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "use_weighted_resectioning" $dataset/config.yaml
grep "use_colmap_resectioning" $dataset/config.yaml
grep "use_gt_matches" $dataset/config.yaml
grep "use_gt_selective_matches" $dataset/config.yaml
grep "use_weighted_feature_matches" $dataset/config.yaml
grep "use_image_matching_thresholding" $dataset/config.yaml
grep "use_shortest_path_pruning" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"

./bin/opensfm reconstruct $dataset

# Baseline + colmap resectioning
sed -i 's/use_gt_matches: true/use_gt_matches: false/g' $dataset/config.yaml
sed -i 's/use_gt_selective_matches: true/use_gt_selective_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier: true/use_image_matching_classifier: false/g' $dataset/config.yaml
sed -i 's/use_weighted_resectioning: true/use_weighted_resectioning: false/g' $dataset/config.yaml
sed -i 's/use_colmap_resectioning: false/use_colmap_resectioning: true/g' $dataset/config.yaml
sed -i 's/use_weighted_feature_matches: true/use_weighted_feature_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_thresholding: true/use_image_matching_thresholding: false/g' $dataset/config.yaml
sed -i 's/use_shortest_path_pruning: true/use_shortest_path_pruning: false/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "use_weighted_resectioning" $dataset/config.yaml
grep "use_colmap_resectioning" $dataset/config.yaml
grep "use_gt_matches" $dataset/config.yaml
grep "use_gt_selective_matches" $dataset/config.yaml
grep "use_weighted_feature_matches" $dataset/config.yaml
grep "use_image_matching_thresholding" $dataset/config.yaml
grep "use_shortest_path_pruning" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"

./bin/opensfm reconstruct $dataset

# Using ground-truth image matches + colmap resectioning - classifier/thresholding won't matter (since the weight is either a 1 or a 0)
sed -i 's/use_gt_matches: false/use_gt_matches: true/g' $dataset/config.yaml
sed -i 's/use_gt_selective_matches: true/use_gt_selective_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier: true/use_image_matching_classifier: false/g' $dataset/config.yaml
sed -i 's/use_weighted_resectioning: true/use_weighted_resectioning: false/g' $dataset/config.yaml
sed -i 's/use_colmap_resectioning: false/use_colmap_resectioning: true/g' $dataset/config.yaml
sed -i 's/use_weighted_feature_matches: true/use_weighted_feature_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_thresholding: true/use_image_matching_thresholding: false/g' $dataset/config.yaml
sed -i 's/use_shortest_path_pruning: true/use_shortest_path_pruning: false/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "use_weighted_resectioning" $dataset/config.yaml
grep "use_colmap_resectioning" $dataset/config.yaml
grep "use_gt_matches" $dataset/config.yaml
grep "use_gt_selective_matches" $dataset/config.yaml
grep "use_weighted_feature_matches" $dataset/config.yaml
grep "use_image_matching_thresholding" $dataset/config.yaml
grep "use_shortest_path_pruning" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"

./bin/opensfm reconstruct $dataset

# Using ground-truth selective image matches + colmap resectioning - classifier/thresholding won't matter (since the weight is either a 1 or a 0)
sed -i 's/use_gt_matches: false/use_gt_matches: true/g' $dataset/config.yaml
sed -i 's/use_gt_selective_matches: false/use_gt_selective_matches: true/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier: true/use_image_matching_classifier: false/g' $dataset/config.yaml
sed -i 's/use_weighted_resectioning: true/use_weighted_resectioning: false/g' $dataset/config.yaml
sed -i 's/use_colmap_resectioning: false/use_colmap_resectioning: true/g' $dataset/config.yaml
sed -i 's/use_weighted_feature_matches: true/use_weighted_feature_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_thresholding: true/use_image_matching_thresholding: false/g' $dataset/config.yaml
sed -i 's/use_shortest_path_pruning: true/use_shortest_path_pruning: false/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "use_weighted_resectioning" $dataset/config.yaml
grep "use_colmap_resectioning" $dataset/config.yaml
grep "use_gt_matches" $dataset/config.yaml
grep "use_gt_selective_matches" $dataset/config.yaml
grep "use_weighted_feature_matches" $dataset/config.yaml
grep "use_image_matching_thresholding" $dataset/config.yaml
grep "use_shortest_path_pruning" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"

./bin/opensfm reconstruct $dataset

# Using ground-truth image matches (with pruning) + colmap resectioning - classifier/thresholding won't matter (since the weight is either a 1 or a 0)
sed -i 's/use_gt_matches: false/use_gt_matches: true/g' $dataset/config.yaml
sed -i 's/use_gt_selective_matches: true/use_gt_selective_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier: true/use_image_matching_classifier: false/g' $dataset/config.yaml
sed -i 's/use_weighted_resectioning: true/use_weighted_resectioning: false/g' $dataset/config.yaml
sed -i 's/use_colmap_resectioning: false/use_colmap_resectioning: true/g' $dataset/config.yaml
sed -i 's/use_weighted_feature_matches: true/use_weighted_feature_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_thresholding: true/use_image_matching_thresholding: false/g' $dataset/config.yaml
sed -i 's/use_shortest_path_pruning: false/use_shortest_path_pruning: true/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "use_weighted_resectioning" $dataset/config.yaml
grep "use_colmap_resectioning" $dataset/config.yaml
grep "use_gt_matches" $dataset/config.yaml
grep "use_gt_selective_matches" $dataset/config.yaml
grep "use_weighted_feature_matches" $dataset/config.yaml
grep "use_image_matching_thresholding" $dataset/config.yaml
grep "use_shortest_path_pruning" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"

./bin/opensfm reconstruct $dataset

# Use robust matches (with pruning) + colmap resectioning
sed -i 's/use_gt_matches: true/use_gt_matches: false/g' $dataset/config.yaml
sed -i 's/use_gt_selective_matches: true/use_gt_selective_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier: true/use_image_matching_classifier: false/g' $dataset/config.yaml
sed -i 's/use_weighted_resectioning: true/use_weighted_resectioning: false/g' $dataset/config.yaml
sed -i 's/use_colmap_resectioning: false/use_colmap_resectioning: true/g' $dataset/config.yaml
sed -i 's/use_weighted_feature_matches: true/use_weighted_feature_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_thresholding: true/use_image_matching_thresholding: false/g' $dataset/config.yaml
sed -i 's/use_shortest_path_pruning: false/use_shortest_path_pruning: true/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "use_weighted_resectioning" $dataset/config.yaml
grep "use_colmap_resectioning" $dataset/config.yaml
grep "use_gt_matches" $dataset/config.yaml
grep "use_gt_selective_matches" $dataset/config.yaml
grep "use_weighted_feature_matches" $dataset/config.yaml
grep "use_image_matching_thresholding" $dataset/config.yaml
grep "use_shortest_path_pruning" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"

./bin/opensfm reconstruct $dataset

./bin/opensfm convert_colmap $dataset
./bin/opensfm validate_results $dataset

# for run in "${runs[@]}"
# do
# 	cmd="google-chrome"
# 	url="http://localhost:8888/viewer/reconstruction.html#file="$dataset
# 	echo $cmd" "$url"/"$run >> $dataset/chrome-commands.sh
# done