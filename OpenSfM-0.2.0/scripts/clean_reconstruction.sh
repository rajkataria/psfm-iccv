dataset=$1

rm -R $dataset/all_matches
rm -R $dataset/weighted_matches
rm -R $dataset/camera_models.json
rm -R $dataset/classifier_dataset
rm -R $dataset/classifier_features
rm -R $dataset/config.yaml
rm -R $dataset/exif
rm -R $dataset/features
rm -R $dataset/feature_matching_results
rm -R $dataset/matches
rm -R $dataset/*.log
rm -R $dataset/*.png
rm -R $dataset/pairwise_results
rm -R $dataset/profile.log
rm -R $dataset/reconstruction_colmap.json
rm -R $dataset/reconstruction.json
rm -R $dataset/reconstruction-*.json
rm -R $dataset/reference_lla.json
rm -R $dataset/reports
rm -R $dataset/results
rm -R $dataset/sift
rm -R $dataset/tracks*.csv
rm -R $dataset/vocab_out
rm -R $dataset/vt_image_list.txt
rm -R $dataset/vt_sift_list.txt
