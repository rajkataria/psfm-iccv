dataset=$1

echo "############################################################################################################"
echo "############################################################################################################"
echo "########################################### Starting experiments ###########################################"
echo "############################################################################################################"
echo "############################################################################################################"
# Baseline
sed -i 's/use_gt_matches: true/use_gt_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier: true/use_image_matching_classifier: false/g' $dataset/config.yaml
sed -i 's/use_weighted_resectioning: true/use_weighted_resectioning: false/g' $dataset/config.yaml
sed -i 's/use_weighted_feature_matches: true/use_weighted_feature_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_thresholding: true/use_image_matching_thresholding: false/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "use_weighted_resectioning" $dataset/config.yaml
grep "use_gt_matches" $dataset/config.yaml
grep "use_weighted_feature_matches" $dataset/config.yaml
grep "use_image_matching_thresholding" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"

./bin/opensfm extract_metadata $dataset
./bin/opensfm detect_features $dataset
./bin/opensfm evaluate_vt_rankings $dataset
./bin/opensfm match_features $dataset
./bin/opensfm classify_features $dataset
./bin/opensfm match_fm_classifier_features $dataset
./bin/opensfm calculate_features $dataset
./bin/opensfm classify_images $dataset
./bin/opensfm create_tracks $dataset
./bin/opensfm create_tracks_classifier $dataset
./bin/opensfm reconstruct $dataset

# Use image matching classifier - threshold image classifier weights
sed -i 's/use_gt_matches: true/use_gt_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier: false/use_image_matching_classifier: true/g' $dataset/config.yaml
sed -i 's/use_weighted_resectioning: true/use_weighted_resectioning: false/g' $dataset/config.yaml
sed -i 's/use_weighted_feature_matches: true/use_weighted_feature_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_thresholding: true/use_image_matching_thresholding: true/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "use_weighted_resectioning" $dataset/config.yaml
grep "use_gt_matches" $dataset/config.yaml
grep "use_weighted_feature_matches" $dataset/config.yaml
grep "use_image_matching_thresholding" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"

./bin/opensfm reconstruct $dataset

# Use image matching classifier and weighted resectioning (don't threshold matches)
sed -i 's/use_gt_matches: true/use_gt_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier: false/use_image_matching_classifier: true/g' $dataset/config.yaml
sed -i 's/use_weighted_resectioning: false/use_weighted_resectioning: true/g' $dataset/config.yaml
sed -i 's/use_weighted_feature_matches: true/use_weighted_feature_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_thresholding: true/use_image_matching_thresholding: false/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "use_weighted_resectioning" $dataset/config.yaml
grep "use_gt_matches" $dataset/config.yaml
grep "use_weighted_feature_matches" $dataset/config.yaml
grep "use_image_matching_thresholding" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"

./bin/opensfm reconstruct $dataset

# Using ground-truth image matches - thresholding won't matter (since the weight is either a 1 or a 0)
sed -i 's/use_gt_matches: false/use_gt_matches: true/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier: false/use_image_matching_classifier: true/g' $dataset/config.yaml
sed -i 's/use_weighted_resectioning: true/use_weighted_resectioning: false/g' $dataset/config.yaml
sed -i 's/use_weighted_feature_matches: true/use_weighted_feature_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_thresholding: true/use_image_matching_thresholding: false/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "use_weighted_resectioning" $dataset/config.yaml
grep "use_gt_matches" $dataset/config.yaml
grep "use_weighted_feature_matches" $dataset/config.yaml
grep "use_image_matching_thresholding" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"

./bin/opensfm reconstruct $dataset

# Using ground-truth matches and weighted resectioning - thresholding won't matter (since the weight is either a 1 or a 0)
sed -i 's/use_gt_matches: false/use_gt_matches: true/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier: false/use_image_matching_classifier: true/g' $dataset/config.yaml
sed -i 's/use_weighted_resectioning: false/use_weighted_resectioning: true/g' $dataset/config.yaml
sed -i 's/use_weighted_feature_matches: true/use_weighted_feature_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_thresholding: true/use_image_matching_thresholding: false/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "use_weighted_resectioning" $dataset/config.yaml
grep "use_gt_matches" $dataset/config.yaml
grep "use_weighted_feature_matches" $dataset/config.yaml
grep "use_image_matching_thresholding" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"

./bin/opensfm reconstruct $dataset

# Use image matching classifier and weighted RANSAC using feature matching weights - threshold image classifier weights
sed -i 's/use_gt_matches: true/use_gt_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier: false/use_image_matching_classifier: true/g' $dataset/config.yaml
sed -i 's/use_weighted_resectioning: true/use_weighted_resectioning: false/g' $dataset/config.yaml
sed -i 's/use_weighted_feature_matches: false/use_weighted_feature_matches: true/g' $dataset/config.yaml
sed -i 's/use_image_matching_thresholding: true/use_image_matching_thresholding: true/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "use_weighted_resectioning" $dataset/config.yaml
grep "use_gt_matches" $dataset/config.yaml
grep "use_weighted_feature_matches" $dataset/config.yaml
grep "use_image_matching_thresholding" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"

./bin/opensfm reconstruct $dataset

# Use image matching classifier and weighted RANSAC using feature matching weights and weighted resectioning (don't threshold matches)
sed -i 's/use_gt_matches: true/use_gt_matches: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier: false/use_image_matching_classifier: true/g' $dataset/config.yaml
sed -i 's/use_weighted_resectioning: false/use_weighted_resectioning: true/g' $dataset/config.yaml
sed -i 's/use_weighted_feature_matches: false/use_weighted_feature_matches: true/g' $dataset/config.yaml
sed -i 's/use_image_matching_thresholding: true/use_image_matching_thresholding: false/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "use_weighted_resectioning" $dataset/config.yaml
grep "use_gt_matches" $dataset/config.yaml
grep "use_weighted_feature_matches" $dataset/config.yaml
grep "use_image_matching_thresholding" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"

./bin/opensfm reconstruct $dataset

./bin/opensfm convert_colmap $dataset
./bin/opensfm validate_results $dataset