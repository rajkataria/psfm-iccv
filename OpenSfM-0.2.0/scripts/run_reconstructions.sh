dataset=$1
mode="reconstruction"

echo "############################################################################################################"
echo "############################################################################################################"
echo "########################################### Starting experiments ###########################################"
echo "############################################################################################################"
echo "############################################################################################################"

# # Baseline + colmap
# declare -A run1=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='false'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='false'   [use_shortest_path_pruning]='false'       [image_matching_classifier_threshold]='0.5'
# 	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.5)
# declare -A run2=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'       [image_matching_classifier_threshold]='0.5'
# 	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.4)
# declare -A run3=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'       [image_matching_classifier_threshold]='0.4'
# 	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.3)
# declare -A run4=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'       [image_matching_classifier_threshold]='0.3'
# 	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.2)
# declare -A run5=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'       [image_matching_classifier_threshold]='0.2'
# 	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.5) + SPP
# declare -A run6=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='true'        [image_matching_classifier_threshold]='0.5'
# 	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.4) + SPP
# declare -A run7=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='true'        [image_matching_classifier_threshold]='0.4'
# 	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.3) + SPP
# declare -A run8=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='true'        [image_matching_classifier_threshold]='0.3'
# 	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.2) + SPP
# declare -A run9=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='true'        [image_matching_classifier_threshold]='0.2'
# 	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.4) + CIP (10)
# declare -A run10=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.4'
# 	[use_closest_images_pruning]='true'			[closest_images_top_k]='H'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.3) + CIP (10)
# declare -A run11=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.3'
# 	[use_closest_images_pruning]='true'			[closest_images_top_k]='H'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.2) + CIP (10)
# declare -A run12=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.2'
# 	[use_closest_images_pruning]='true'			[closest_images_top_k]='H'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.4) + CIP (20)
# declare -A run13=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.4'
# 	[use_closest_images_pruning]='true'			[closest_images_top_k]='G'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.3) + CIP (20)
# declare -A run14=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.3'
# 	[use_closest_images_pruning]='true'			[closest_images_top_k]='G'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.2) + CIP (20)
# declare -A run15=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.2'
# 	[use_closest_images_pruning]='true'			[closest_images_top_k]='G'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.3) + ground-truth CIP(10)
# declare -A run16=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.3'
# 	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'										[use_gt_closest_images_pruning]='true'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.3) + ground-truth CIP(20)
# declare -A run17=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.3'
# 	[use_closest_images_pruning]='false'		[closest_images_top_k]='G'										[use_gt_closest_images_pruning]='true'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.3) + CIP (A)
# declare -A run18=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.3'
# 	[use_closest_images_pruning]='true'			[closest_images_top_k]='A'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.3) + CIP (B)
# declare -A run19=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.3'
# 	[use_closest_images_pruning]='true'			[closest_images_top_k]='B'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.3) + CIP (C)
# declare -A run20=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.3'
# 	[use_closest_images_pruning]='true'			[closest_images_top_k]='C'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.3) + CIP (D)
# declare -A run21=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.3'
# 	[use_closest_images_pruning]='true'			[closest_images_top_k]='D'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.3) + CIP (E)
# declare -A run22=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.3'
# 	[use_closest_images_pruning]='true'			[closest_images_top_k]='E'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.3) + CIP (F)
# declare -A run23=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.3'
# 	[use_closest_images_pruning]='true'			[closest_images_top_k]='F'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # colmap + CIP (A)
# declare -A run24=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='false'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='false'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.3'
# 	[use_closest_images_pruning]='true'			[closest_images_top_k]='A'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # colmap + CIP (B)
# declare -A run25=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='false'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='false'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.3'
# 	[use_closest_images_pruning]='true'			[closest_images_top_k]='B'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # colmap + CIP (C)
# declare -A run26=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='false'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='false'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.3'
# 	[use_closest_images_pruning]='true'			[closest_images_top_k]='C'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # colmap + CIP (D)
# declare -A run27=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='false'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='false'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.3'
# 	[use_closest_images_pruning]='true'			[closest_images_top_k]='D'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # colmap + CIP (E)
# declare -A run28=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='false'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='false'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.3'
# 	[use_closest_images_pruning]='true'			[closest_images_top_k]='E'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.3) + ground-truth CIP(A)
# declare -A run29=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.3'
# 	[use_closest_images_pruning]='false'		[closest_images_top_k]='A'										[use_gt_closest_images_pruning]='true'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.3) + ground-truth CIP(B)
# declare -A run30=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.3'
# 	[use_closest_images_pruning]='false'		[closest_images_top_k]='B'										[use_gt_closest_images_pruning]='true'
# 	[use_yan_disambiguation]='false'
# 	)

# # Image matching classifier + colmap + thresholding (0.15) + ground-truth CIP(A)
# declare -A run31=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.15'
# 	[use_closest_images_pruning]='false'		[closest_images_top_k]='A'										[use_gt_closest_images_pruning]='true'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.15) + ground-truth CIP(B)
# declare -A run32=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.15'
# 	[use_closest_images_pruning]='false'		[closest_images_top_k]='B'										[use_gt_closest_images_pruning]='true'
# 	[use_yan_disambiguation]='false'
# 	)

# # Image matching classifier + colmap + thresholding (0.15) + CIP(A)
# declare -A run33=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.15'
# 	[use_closest_images_pruning]='true'		[closest_images_top_k]='A'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)
# # Image matching classifier + colmap + thresholding (0.15) + CIP(B)
# declare -A run34=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='true'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='true'    [use_shortest_path_pruning]='false'        [image_matching_classifier_threshold]='0.15'
# 	[use_closest_images_pruning]='true'		[closest_images_top_k]='B'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='false'
# 	)

# # Baseline + colmap + yan
# declare -A run35=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='false'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='false'   [use_shortest_path_pruning]='false'       [image_matching_classifier_threshold]='0.5'
# 	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='true'
# 	)

# # Baseline + yan
# declare -A run36=(
# 	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        [use_image_matching_classifier]='false'
# 	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='false'          [use_weighted_feature_matches]='false'
# 	[use_image_matching_thresholding]='false'   [use_shortest_path_pruning]='false'       [image_matching_classifier_threshold]='0.5'
# 	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'										[use_gt_closest_images_pruning]='false'
# 	[use_yan_disambiguation]='true'
# 	)

# all_runs=(run1 run2 run3 run4 run5 run6 run7 run8 run9)
# all_runs=(run10 run11 run12)
# all_runs=(run13 run14 run15)
# all_runs=(run17)
# all_runs=(run18 run19 run20 run21 run22 run23)
# all_runs=(run24 run25 run26 run27 run28)
# all_runs=(run29 run30 run31 run32 run33 run34)
# all_runs=(run35)

# all_runs=(run2 run3 run4 run5)
# all_runs=(run18 run19 run20 run21 run22 run23 run29 run30 run31 run32 run33 run34)

# all_runs=(run1 run35)
# all_runs=(run35)
# all_runs=(run36)
# all_runs=(run1)


# Baseline + colmap
declare -A run100=(
	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        	[use_image_matching_classifier]='false'
	[use_weighted_resectioning]='false'         [use_colmap_resectioning]='true'          	[use_weighted_feature_matches]='false'
	[use_image_matching_thresholding]='false'   [use_shortest_path_pruning]='false'       	[image_matching_classifier_threshold]='0.5'
	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'					[use_gt_closest_images_pruning]='false'
	[use_yan_disambiguation]='false'
	)

# Baseline + weighted colmap resectioning (sum)
declare -A run101=(
	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        	[use_image_matching_classifier]='false'
	[use_weighted_resectioning]='sum'         	[use_colmap_resectioning]='false'          	[use_weighted_feature_matches]='false'
	[use_image_matching_thresholding]='false'   [use_shortest_path_pruning]='false'       	[image_matching_classifier_threshold]='0.5'
	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'					[use_gt_closest_images_pruning]='false'
	[use_yan_disambiguation]='false'
	)

# Image matching classifier + weighted colmap resectioning (sum)
declare -A run102=(
	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        	[use_image_matching_classifier]='true'
	[use_weighted_resectioning]='sum'         	[use_colmap_resectioning]='false'          	[use_weighted_feature_matches]='false'
	[use_image_matching_thresholding]='false'   [use_shortest_path_pruning]='false'       	[image_matching_classifier_threshold]='0.5'
	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'					[use_gt_closest_images_pruning]='false'
	[use_yan_disambiguation]='false'
	)

# Baseline + weighted colmap resectioning (max)
declare -A run103=(
	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        	[use_image_matching_classifier]='false'
	[use_weighted_resectioning]='max'         	[use_colmap_resectioning]='false'          	[use_weighted_feature_matches]='false'
	[use_image_matching_thresholding]='false'   [use_shortest_path_pruning]='false'       	[image_matching_classifier_threshold]='0.5'
	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'					[use_gt_closest_images_pruning]='false'
	[use_yan_disambiguation]='false'
	)

# Image matching classifier + weighted colmap resectioning (max)
declare -A run104=(
	[use_gt_matches]='false'                    [use_gt_selective_matches]='false'        	[use_image_matching_classifier]='true'
	[use_weighted_resectioning]='max'         	[use_colmap_resectioning]='false'          	[use_weighted_feature_matches]='false'
	[use_image_matching_thresholding]='false'   [use_shortest_path_pruning]='false'       	[image_matching_classifier_threshold]='0.5'
	[use_closest_images_pruning]='false'		[closest_images_top_k]='H'					[use_gt_closest_images_pruning]='false'
	[use_yan_disambiguation]='false'
	)

# all_runs=(run100 run101 run102 run103 run104)
all_runs=(run101 run102 run103 run104)

count=0
for run_name in "${all_runs[@]}"; do
    declare -n run_ref="$run_name"

    c_use_gt_matches="${run_ref[use_gt_matches]}"
    c_use_gt_selective_matches="${run_ref[use_gt_selective_matches]}"
    c_use_image_matching_classifier="${run_ref[use_image_matching_classifier]}"
    c_use_weighted_resectioning="${run_ref[use_weighted_resectioning]}"
    c_use_colmap_resectioning="${run_ref[use_colmap_resectioning]}"
    c_use_weighted_feature_matches="${run_ref[use_weighted_feature_matches]}"
    c_use_image_matching_thresholding="${run_ref[use_image_matching_thresholding]}"
    c_use_shortest_path_pruning="${run_ref[use_shortest_path_pruning]}"
    c_image_matching_classifier_threshold="${run_ref[image_matching_classifier_threshold]}"
    c_use_closest_images_pruning="${run_ref[use_closest_images_pruning]}"
    c_closest_images_top_k="${run_ref[closest_images_top_k]}"
    c_use_gt_closest_images_pruning="${run_ref[use_gt_closest_images_pruning]}"
    c_use_yan_disambiguation="${run_ref[use_yan_disambiguation]}"
    
    sed -i "s/use_gt_matches: .*/use_gt_matches: ${c_use_gt_matches}/g" $dataset/config.yaml
    sed -i "s/use_gt_selective_matches: .*/use_gt_selective_matches: ${c_use_gt_selective_matches}/g" $dataset/config.yaml
    sed -i "s/use_image_matching_classifier: .*/use_image_matching_classifier: ${c_use_image_matching_classifier}/g" $dataset/config.yaml
    sed -i "s/use_weighted_resectioning: .*/use_weighted_resectioning: ${c_use_weighted_resectioning}/g" $dataset/config.yaml
    sed -i "s/use_colmap_resectioning: .*/use_colmap_resectioning: ${c_use_colmap_resectioning}/g" $dataset/config.yaml
    sed -i "s/use_weighted_feature_matches: .*/use_weighted_feature_matches: ${c_use_weighted_feature_matches}/g" $dataset/config.yaml
    sed -i "s/use_image_matching_thresholding: .*/use_image_matching_thresholding: ${c_use_image_matching_thresholding}/g" $dataset/config.yaml
    sed -i "s/use_shortest_path_pruning: .*/use_shortest_path_pruning: ${c_use_shortest_path_pruning}/g" $dataset/config.yaml
    sed -i "s/image_matching_classifier_threshold: .*/image_matching_classifier_threshold: ${c_image_matching_classifier_threshold}/g" $dataset/config.yaml
    sed -i "s/use_closest_images_pruning: .*/use_closest_images_pruning: ${c_use_closest_images_pruning}/g" $dataset/config.yaml
    sed -i "s/closest_images_top_k: .*/closest_images_top_k: ${c_closest_images_top_k}/g" $dataset/config.yaml
    sed -i "s/use_gt_closest_images_pruning: .*/use_gt_closest_images_pruning: ${c_use_gt_closest_images_pruning}/g" $dataset/config.yaml
    sed -i "s/use_yan_disambiguation: .*/use_yan_disambiguation: ${c_use_yan_disambiguation}/g" $dataset/config.yaml

	echo "Classifier configuration: ${count}"
	grep "use_image_matching_classifier:" $dataset/config.yaml
	grep "use_weighted_resectioning:" $dataset/config.yaml
	grep "use_colmap_resectioning:" $dataset/config.yaml
	grep "use_gt_matches:" $dataset/config.yaml
	grep "use_gt_selective_matches:" $dataset/config.yaml
	grep "use_weighted_feature_matches:" $dataset/config.yaml
	grep "use_image_matching_thresholding:" $dataset/config.yaml
	grep "use_shortest_path_pruning:" $dataset/config.yaml
	grep "image_matching_classifier_threshold:" $dataset/config.yaml
	grep "use_closest_images_pruning:" $dataset/config.yaml
	grep "closest_images_top_k:" $dataset/config.yaml
	grep "use_gt_closest_images_pruning:" $dataset/config.yaml
	grep "use_yan_disambiguation:" $dataset/config.yaml

	# if [ "$count" -eq -1 ]
	# then

	# ./bin/opensfm extract_metadata $dataset
	# ./bin/opensfm detect_features $dataset
	# ./bin/opensfm evaluate_vt_rankings $dataset
	# ./bin/opensfm match_features $dataset
	# ./bin/opensfm create_tracks $dataset

	# 	# ./bin/opensfm classify_features $dataset
	# 	# ./bin/opensfm match_fm_classifier_features $dataset
	# 	# ./bin/opensfm calculate_features $dataset
	# 	# ./bin/opensfm classify_images $dataset
	# fi

	if [ "$mode" == "reconstruction" ];then
		# ./bin/opensfm yan $dataset
		./bin/opensfm create_tracks $dataset
		./bin/opensfm create_tracks_classifier $dataset
		./bin/opensfm reconstruct $dataset
	fi

	echo "************************************************************************************************************"
	echo "************************************************************************************************************"
	# (( count++ ));
done

# ./bin/opensfm convert_colmap $dataset
# ./bin/opensfm validate_results $dataset

# for run in "${runs[@]}"
# do
# 	cmd="google-chrome"
# 	url="http://localhost:8888/viewer/reconstruction.html#file="$dataset
# 	echo $cmd" "$url"/"$run >> $dataset/chrome-commands.sh
# done
