declare -a yan_datasets=("Yan/books" "Yan/cereal" "Yan/cup" "Yan/desk" "Yan/oats" "Yan/street")
# declare -a eth3d_datasets=("ETH3D/boulders" "ETH3D/courtyard" "ETH3D/exhibition_hall" "ETH3D/lecture_room" "ETH3D/living_room" "ETH3D/pipes" "ETH3D/terrains")
declare -a eth3d_datasets=("ETH3D/exhibition_hall")
declare -a uiuctag_datasets=("UIUCTag/ece_floor3_loop_ccw" "UIUCTag/ece_floor3_loop_cw" "UIUCTag/ece_floor5" "UIUCTag/ece_floor5_wall" "UIUCTag/yeh_night_all" "UIUCTag/yeh_night_backward" "UIUCTag/yeh_night_forward")
declare -a duplicate_structures_datasets=("DuplicateStructures/alexander_nevsky_cathedral" "DuplicateStructures/arc_de_triomphe" "DuplicateStructures/brandenburg_gate" "DuplicateStructures/cereal" 
  "DuplicateStructures/church_on_spilled_blood" "DuplicateStructures/indoor" "DuplicateStructures/radcliffe_camera" "DuplicateStructures/street")
declare -a tanksandtemples_datasets=("TanksAndTemples/Ballroom" "TanksAndTemples/Barn" "TanksAndTemples/Meetingroom" "TanksAndTemples/Museum")

# datasets=( "${yan_datasets[@]}" "${eth3d_datasets[@]}" "${uiuctag_datasets[@]}" "${duplicate_structures_datasets[@]}" "${tanksandtemples_datasets[@]}" )
datasets=( "${yan_datasets[@]}" )

for i in "${datasets[@]}"; do
  for j in `seq 1 1`; do
    dset=(${i//// });
    root=${dset[0]};
    sequence=${dset[1]};
    unameOut="$(uname -s)"
    case "${unameOut}" in
        Linux*)     machine=Linux;;
        Darwin*)    machine=Mac;;
        CYGWIN*)    machine=Cygwin;;
        MINGW*)     machine=MinGw;;
        *)          machine="UNKNOWN:${unameOut}"
    esac
    
    echo $machine
    if [ "$machine" == "Linux" ]; then
        command='google-chrome --incognito '
    else
        command='open -na "Google Chrome" --args --incognito '
    fi


    # eval $command"http://localhost:8888/viewer/reconstruction.html#file=/data/classifier-datasets-bruteforce/$root/$sequence/reconstruction_gt.json"
    # eval $command"http://localhost:8888/viewer/reconstruction.html#file=/data/classifier-datasets-bruteforce/$root/$sequence/reconstruction-imc-False-fm-False-wr-original-resc-NA-udt-False-dt-0.6-mkcip-0.15-mkcimin-7-mkcimax-12-recc-0.json"
    # eval $command"http://localhost:8888/viewer/reconstruction.html#file=/data/classifier-datasets-bruteforce/$root/$sequence/reconstruction-imc-False-fm-False-wr-colmap-resc-NA-udt-False-dt-0.6-mkcip-0.15-mkcimin-7-mkcimax-12-recc-0.json"
    # eval $command"http://localhost:8888/viewer/reconstruction.html#file=/data/classifier-datasets-bruteforce/$root/$sequence/reconstruction-imc-False-fm-False-wr-colmap-resc-NA-udt-True-dt-0.6-mkcip-0.15-mkcimin-6-mkcimax-14-recc-0.json"
    # eval $command"http://localhost:8888/viewer/reconstruction.html#file=/data/classifier-datasets-bruteforce/$root/$sequence/reconstruction-imc-False-fm-False-wr-colmap-resc-NA-udt-True-dt-0.6-mkcip-0.2-mkcimin-6-mkcimax-14-recc-0.json"
    # eval $command"http://localhost:8888/viewer/reconstruction.html#file=/data/classifier-datasets-bruteforce/$root/$sequence/reconstruction-imc-False-fm-False-wr-colmap-resc-NA-udt-True-dt-0.6-mkcip-0.15-mkcimin-7-mkcimax-12-recc-0.json"

    # eval $command"http://localhost:8888/viewer/reconstruction.html#file=/data/classifier-datasets-bruteforce/$root/$sequence/reconstruction-imc-False-fm-False-wr-colmap-resc-NA-mdstc-mst-adaptive-distance-mdstv-5-mkcip-0.15-mkcimin-6-mkcimax-14-ust-True-recc-0.json"
    eval $command"http://localhost:8888/viewer/reconstruction.html#file=/data/classifier-datasets-bruteforce/$root/$sequence/reconstruction-imc-False-fm-False-wr-colmap-resc-NA-mdstc-mst-adaptive-distance-mdstv-5-mkcip-0.15-mkcimin-6-mkcimax-14-ust-False-recc-0.json"

    # eval $command"http://localhost:8888/viewer/reconstruction.html#file=/data/classifier-datasets-bruteforce/$root/$sequence/reconstruction-imc-False-fm-False-wr-colmap-resc-NA-mdstc-closest-images-mdstv-5-mkcip-0.15-mkcimin-6-mkcimax-14-ust-True-recc-0.json"
    # eval $command"http://localhost:8888/viewer/reconstruction.html#file=/data/classifier-datasets-bruteforce/$root/$sequence/reconstruction-imc-False-fm-False-wr-colmap-resc-NA-mdstc-closest-images-mdstv-5-mkcip-0.15-mkcimin-6-mkcimax-14-ust-False-recc-0.json"
    
  done;
done




# open -a "Google Chrome" http://localhost:8888/viewer/reconstruction.html#file=/data/variance-exps/UIUCTag-1/ece_floor3_loop_ccw/reconstruction_gt.json
# open -a "Google Chrome" http://localhost:8888/viewer/reconstruction.html#file=/data/variance-exps/UIUCTag-1/ece_floor3_loop_ccw/reconstruction-imc-False-wr-False-colmapr-False-gm-False-wfm-False-imt-False-spp-False.json
# open -a "Google Chrome" http://localhost:8888/viewer/reconstruction.html#file=/data/variance-exps/UIUCTag-1/ece_floor3_loop_ccw/reconstruction-imc-False-wr-False-colmapr-True-gm-False-wfm-False-imt-False-spp-False.json
# open -a "Google Chrome" http://localhost:8888/viewer/reconstruction.html#file=/data/variance-exps/UIUCTag-1/ece_floor3_loop_ccw/reconstruction-imc-False-wr-False-colmapr-True-gm-True-wfm-False-imt-False-spp-False.json
# open -a "Google Chrome" http://localhost:8888/viewer/reconstruction.html#file=/data/variance-exps/UIUCTag-1/ece_floor3_loop_ccw/reconstruction-imc-False-wr-False-colmapr-True-gm-True-wfm-False-imt-False-spp-True.json
# open -a "Google Chrome" http://localhost:8888/viewer/reconstruction.html#file=/data/variance-exps/UIUCTag-1/ece_floor3_loop_ccw/reconstruction-imc-False-wr-False-colmapr-True-gm-False-wfm-False-imt-False-spp-True.json
