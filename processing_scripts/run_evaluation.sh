# to run the evaluation copy this file to the folder containing the test sequences
# each sequence requires a folder odometry_resutls containing the visual odometry results

FourSeasonsTools=~/workspace/4seasons_tools
GpsConfigFile=$FourSeasonsTools/configs/4seasons/gps_config.cfg
GpsVarThreshold=0.01 # this theshold is not really relevant, since for evaluation only poses with accuracy smaller 5cm are used

for d in *; do
    if [ -d $d ]; then
        echo "Evaluate Sequence $d"
	# define accurate poses
        if [ ! -f "$d/keyframe_accuracy.txt" ]; then
            python3 $FourSeasonsTools/tools/pose_accuracy.py \
                $d $d/septentrio.nmea \
                --gps_config $GpsConfigFile \
                --output $d \
                --gps_var_threshold $GpsVarThreshold \
                --only_accurate_kf \
                --save_figs
        fi

        if [ -d $d/odometry_resutls ]; then
            # run evaluation
            if [ ! -d $d/odometry_resutls/pose_error ]; then
                mkdir $d/odometry_resutls/pose_error
            fi
            cd $d/odometry_resutls
            for file in *; do
                if [ -f $file ]; then
                    echo "evaluate method: $file"
                    python3 $FourSeasonsTools/evaluation/odometry/odometry_benchmark.py \
                        $file \
                        ../
                    mv error.txt pose_error/$file
                fi
            done
            cd ../..

            # plot results
            echo "plot results: $d"
            if [ ! -d $d/odometry_resutls/results_plot ]; then
                mkdir $d/odometry_resutls/results_plot
            fi
            if [ -d $d/odometry_resutls/pose_error ]; then
	        python3 $FourSeasonsTools/evaluation/odometry/odometry_eval_plot.py $d/odometry_resutls/pose_error
            fi
            mv odometry_results.pdf $d/odometry_resutls/results_plot/odometry_results.pdf
        fi
    fi
done
