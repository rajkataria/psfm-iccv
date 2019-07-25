# datasets=('Barn' 'Caterpillar' 'Church' 'Courthouse' 'Ignatius' 'Meetingroom' 'Truck')
datasets=('Barn' 'Caterpillar' 'Ignatius' 'Meetingroom' 'Truck')

for i in ${datasets[@]};
do
	echo "*************************************************************************************************"
	echo $i
    ./bin/opensfm calculate_features /home/ubuntu/Results/completed-classifier-datasets-bruteforce/TanksAndTemples/$i 
done
