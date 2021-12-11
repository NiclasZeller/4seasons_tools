# to run the this script copy it to the folder containing the sequences to be processed
# and adapt the following paths respectively

FOUR_SEASONS_TOOLS_DIR=~/workspace/4seasons_tools
BASALT_DIR=~/workspace/basalt
BASALT_CONFIG_FILE=$FOUR_SEASONS_TOOLS_DIR/configs/basalt/arti_config.json
BASALT_CALIB_FILE=$FOUR_SEASONS_TOOLS_DIR/configs/basalt/arti_calib.json

for d in *; do
    if [ -d $d ]; then
        if [ ! -d $d/odometry_resutls ]; then
            mkdir $d/odometry_resutls
        fi
        if [ ! -f $d/odometry_resutls/basalt_stereo.txt ]; then
            $BASALT_DIR/build/basalt_vio \
                --dataset-path $d/distorted_images \
                --cam-calib $BASALT_CALIB_FILE \
                --dataset-type arti \
                --config-path $BASALT_CONFIG_FILE \
                --save-trajectory arti \
                --show-gui 0 \
                --use-imu 0 \
                --use-double 1
	
            mv trajectory_left_cam_rect.txt $d/odometry_resutls/basalt_stereo.txt
        fi
    fi
done
