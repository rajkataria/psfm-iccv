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
# ./bin/opensfm match_fm_classifier_features $dataset
./bin/opensfm calculate_features $dataset
./bin/opensfm classify_images $dataset
# ./bin/opensfm formulate_graphs $dataset