dataset=$1

# Baseline
sed -i 's/use_image_matching_classifier: true/use_image_matching_classifier: false/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier:true/use_image_matching_classifier: false/g' $dataset/config.yaml
sed -i 's/weighted_resectioning: true/weighted_resectioning: false/g' $dataset/config.yaml
sed -i 's/weighted_resectioning:true/weighted_resectioning: false/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "weighted_resectioning" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"

./bin/opensfm extract_metadata $dataset
./bin/opensfm detect_features $dataset
./bin/opensfm evaluate_vt_rankings $dataset
./bin/opensfm match_features $dataset
./bin/opensfm calculate_features $dataset
./bin/opensfm classify_images $dataset
./bin/opensfm create_tracks $dataset
./bin/opensfm create_tracks_classifier $dataset
./bin/opensfm reconstruct $dataset

# Use image matching classifier
sed -i 's/use_image_matching_classifier: false/use_image_matching_classifier: true/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier:false/use_image_matching_classifier: true/g' $dataset/config.yaml
sed -i 's/weighted_resectioning: true/weighted_resectioning: false/g' $dataset/config.yaml
sed -i 's/weighted_resectioning:true/weighted_resectioning: false/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "weighted_resectioning" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"

./bin/opensfm reconstruct $dataset

# Use image matching classifier and weighted resectioning
sed -i 's/use_image_matching_classifier: false/use_image_matching_classifier: true/g' $dataset/config.yaml
sed -i 's/use_image_matching_classifier:false/use_image_matching_classifier: true/g' $dataset/config.yaml
sed -i 's/weighted_resectioning: false/weighted_resectioning: true/g' $dataset/config.yaml
sed -i 's/weighted_resectioning:false/weighted_resectioning: true/g' $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
echo "Classifier configurations:"
grep "use_image_matching_classifier" $dataset/config.yaml
grep "weighted_resectioning" $dataset/config.yaml
echo "************************************************************************************************************"
echo "************************************************************************************************************"
./bin/opensfm reconstruct $dataset

./bin/opensfm convert_colmap $dataset
./bin/opensfm validate_results $dataset

