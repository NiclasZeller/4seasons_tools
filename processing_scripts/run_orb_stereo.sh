# to run the this script copy it to the folder containing the sequences to be processed
# and adapt the following paths respectively

FOUR_SEASONS_TOOLS_DIR=~/workspace/4seasons_tools
ORBSLAM_DIR=~/workspace/ORB_SLAM2
BOW_VOCABULARY=$ORBSLAM_DIR/Vocabulary/ORBvoc.txt
ORBSLAM_CALIB_FILE=$FOUR_SEASONS_TOOLS_DIR/configs/orb_slam/orb_slam_calib.yaml

for d in *; do
    if [ -d $d ]; then
        echo "$d"
	if [ ! -f $d/odometry_resutls/orb_slam_stereo.txt ]; then
	    $ORBSLAM_DIR/Examples/Artisense/stereo_artisense \
                $BOW_VOCABULARY \
                $ORBSLAM_CALIB_FILE \
                $d 
	    if [ ! -d $d/odometry_resutls ]; then
                mkdir $d/odometry_resutls
            fi
            mv CameraTrajectory.txt $d/odometry_resutls/orb_slam_stereo.txt
        fi
    fi
done
