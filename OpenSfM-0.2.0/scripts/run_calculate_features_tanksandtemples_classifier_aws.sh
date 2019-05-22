datasets=('Barn' 'Caterpillar' 'Church' 'Courthouse' 'Ignatius' 'Meetingroom' 'Truck')

for i in ${datasets[@]};
do
    ./bin/opensfm calculate_features /home/ubuntu/Results/completed-classifier-datasets-bruteforce/TanksAndTemples/$i 
done
