dataset=$1

rm -R $dataset/all_matches
rm -R $dataset/camera_models.json
rm -R $dataset/classifier_dataset
rm -R $dataset/classifier_features
rm -R $dataset/config.yaml
#rm -R $dataset/dslr_calibration_undistorted
rm -R $dataset/exif
rm -R $dataset/features
#rm -R $dataset/images
rm -R $dataset/matches
rm -R $dataset/output.log
rm -R $dataset/pairwise_results
rm -R $dataset/profile.log
rm -R $dataset/reconstruction_colmap.json
#rm -R $dataset/reconstruction_gt.json
rm -R $dataset/reconstruction.json
rm -R $dataset/reconstruction-*.json
rm -R $dataset/reference_lla.json
rm -R $dataset/reports
rm -R $dataset/results
rm -R $dataset/sift
#rm -R $dataset/sparse_converted
rm -R $dataset/tracks*.csv
rm -R $dataset/vocab_out
rm -R $dataset/vt_image_list.txt
rm -R $dataset/vt_sift_list.txt
