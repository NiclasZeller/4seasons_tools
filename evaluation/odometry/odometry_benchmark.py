from libartipy.dataset import Dataset, Constants, CameraType
from libartipy.geometry import CoordinateSystem
import argparse
import numpy as np
import math
import os

from scipy.spatial.transform import Rotation as R

max_pos_deviation = 0.05
max_relative_seg_len = 1.5
relative_inc = 0.25
segment_lengths = np.array([100, 200, 400, 600, 800, 1000])


def tsToMilliSeconds(ts):
    if ts > 1000000000000:
        ts = ts / 1000000
    else:
        ts = ts * 1000
    return np.uint64(ts + 0.5)


def readPoses(filename, min_ts=0):
    poses = list()

    with open(filename, 'r') as file:
        for line in file.readlines():
            items = line.split()
            assert (len(items) == 8)
            ts = float(items[0])
            ts = tsToMilliSeconds(ts)

            if ts < min_ts:
                continue

            x = float(items[1])
            y = float(items[2])
            z = float(items[3])
            qx = float(items[4])
            qy = float(items[5])
            qz = float(items[6])
            qw = float(items[7])

            poses.append((ts, x, y, z, qx, qy, qz, qw))

    def sortTs(val):
        return val[0]

    poses.sort(key=sortTs)
    return poses


def poseToMatrix(pose):
    m = np.identity(4)
    if np.linalg.norm(np.array(pose[4:8])) == 0.0:
        m = -1 * m
    else:
        m[0:3, 3] = np.array(pose[1:4])
        m[0:3, 0:3] = (R.from_quat(np.array(pose[4:8]))).as_matrix()

    return m


def evalOdometry(eval_poses_file, dataset_dir):
    # read reference poses
    constants = Constants()
    constants.GPS_POSES_FILE_NAME = 'GNSSPoses.txt'
    data_gt = Dataset(dataset_dir, constants=constants)
    data_gt.parse_keyframes()
    data_gt.set_keyframe_poses_to_gps_poses()
    poses_gt_enu = data_gt.get_keyframe_poses_in_coordinate_system(
        coordinate_system=CoordinateSystem.ENU)
    timestamps = data_gt.get_all_kf_timestamps()

    # read accurate poses
    keyframe_accuracy_file = os.path.join(dataset_dir, 'keyframe_accuracy.txt')
    if not os.path.exists(keyframe_accuracy_file):
        exit(0)

    accurate_keyframe = list()
    with open(keyframe_accuracy_file, 'r') as file:
        for line in file.readlines():
            if line[0] == '\n' or line[0] == '#':
                continue

            items = line.split(sep=' ')
            assert (len(items) == 4)
            ts_ns = np.uint64(items[0])
            accuracy = np.float(items[3])

            # ingore all pose with accuracy above max_pos_deviation
            if accuracy > max_pos_deviation:
                continue

            accurate_keyframe.append(ts_ns)

    accurate_keyframe = np.array(accurate_keyframe)
    print('{} accurate keyframes'.format(len(accurate_keyframe)))

    i_accurate = 0
    trajectory_len = 0
    last_pos = None
    gt_poses = list()
    for ts in timestamps:
        if i_accurate >= len(accurate_keyframe):
            break
        ts_accurate = accurate_keyframe[i_accurate]

        if last_pos is not None:
            trajectory_len += np.linalg.norm(last_pos - poses_gt_enu[ts].translation)

        last_pos = poses_gt_enu[ts].translation

        if ts_accurate == ts:
            gt_poses.append((np.uint64(ts_accurate / 1e6 + 0.5), trajectory_len, poses_gt_enu[ts]))

        if ts_accurate <= ts:
            i_accurate += 1

    print('{} ground truth poses'.format(len(gt_poses)))

    # read eval poses
    eval_poses = readPoses(eval_poses_file)
    print('{} evaluation poses'.format(len(eval_poses)))

    eval_poses_aligned = list()
    j_last = 0
    for i in range(len(gt_poses)):
        ts_gt = gt_poses[i][0]

        for j in range(j_last, len(eval_poses)):
            ts_eval = eval_poses[j][0]
            j_last = j
            if ts_eval == ts_gt:
                eval_poses_aligned.append(eval_poses[j])
                break

            if ts_eval > ts_gt:
                break

        if len(eval_poses_aligned) < i + 1:
            print("add dummy pose")
            # add a dummy pose since it is missing in the evaluation trajectory
            eval_poses_aligned.append((ts_gt, -1, -1, -1, 0, 0, 0, 0))

    assert (len(gt_poses) == len(eval_poses_aligned))

    print('{} aligned evaluation poses'.format(len(eval_poses_aligned)))

    rel_pose_pairs = list()
    for segment in segment_lengths:
        inc = segment * relative_inc

        i_first = 0
        traj_first = gt_poses[i_first][1]
        traj_next = traj_first
        i_next = i_first

        for i, gt_pose_i in enumerate(gt_poses):
            while (traj_next is not None) and ((traj_next - traj_first) <= inc):
                i_next += 1
                if len(gt_poses) > i_next:
                    traj_next = gt_poses[i_next][1]
                else:
                    traj_next = None
                    break

            traj_second = gt_pose_i[1]
            i_second = i

            if (traj_second - traj_first) >= segment:
                if (traj_second - traj_first) <= max_relative_seg_len * segment:

                    pose_gt_first = np.float64(gt_poses[i_first][2].transformation_matrix)
                    pose_eval_first = poseToMatrix(eval_poses_aligned[i_first])

                    pose_gt_second = np.float64(gt_poses[i_second][2].transformation_matrix)
                    pose_eval_second = poseToMatrix(eval_poses_aligned[i_second])

                    if pose_eval_first[3, 3] == -1 or pose_eval_second[3, 3] == -1:
                        rel_pose_pairs.append((0, np.identity(4), np.identity(4)))
                    else:
                        rel_pose_gt = np.matmul(np.linalg.inv(pose_gt_second), pose_gt_first)
                        rel_pose_eval = np.matmul(np.linalg.inv(pose_eval_second), pose_eval_first)
                        rel_pose_pairs.append((traj_second - traj_first, rel_pose_gt, rel_pose_eval, 0))

                        print("found one pair: seg: {}, dist: {}, inc: {}".format(segment, traj_second - traj_first,
                                                                                  traj_next - traj_first))

                traj_first = traj_next
                i_first = i_next

            if traj_first is None:
                break

    print("{} relative pose pairs defined".format(len(rel_pose_pairs)))

    error_file = open("error.txt", "w+")

    log_scales = list()
    for i in range(len(rel_pose_pairs)):
        if rel_pose_pairs[i][0] == 0:
            continue

        s = np.linalg.norm(rel_pose_pairs[i][1][0:3, 3]) / np.linalg.norm(rel_pose_pairs[i][2][0:3, 3])
        log_s = math.log(s)
        if log_s == log_s:
            log_scales.append(log_s)

    log_scales = np.array(log_scales)
    if len(log_scales) > 0:
        scale = math.exp(np.median(log_scales))
    else:
        scale = 1

    print("absolute scale: {}".format(scale))
    print("calculated from {} relative pose pairs".format(len(log_scales)))

    errors = list()
    for i in range(len(rel_pose_pairs)):
        d = rel_pose_pairs[i][0]

        if d == 0:
            errors.append((0, 100, 100, 100))
        else:
            rel_pose_gt = rel_pose_pairs[i][1]
            rel_pose_eval = rel_pose_pairs[i][2]

            rel_pose_eval[0:3, 3] = rel_pose_eval[0:3, 3] * scale

            delta = np.matmul(rel_pose_eval, np.linalg.inv(rel_pose_gt))
            r = R.from_matrix(delta[0:3, 0:3])
            w = r.as_quat()[3]

            e_trans = np.linalg.norm(delta[0:3, 3]) / d * 100.0
            e_rot = 2 * math.acos(w) / d * 180.0 / math.pi
            e_scale = np.linalg.norm(rel_pose_eval[0:3, 3]) / np.linalg.norm(rel_pose_gt[0:3, 3])
            e_scale = max(e_scale, 1.0 / e_scale)

            errors.append((d, e_trans, e_rot, e_scale))

        error_file.write("%f %f %f %f\n" % errors[-1])

    error_file.close()

    errors = np.array(errors)
    e_trans_vec = errors[:, 1]
    e_rot_vec = errors[:, 2]
    e_scale_vec = errors[:, 3]

    print("mean relative translation error: {} percent of traveled distance".format(np.mean(e_trans_vec)))
    print("mean relative rotation error: {} degree per meter traveled distance".format(np.mean(e_rot_vec)))
    print("mean relative scale error: {}".format(np.mean(e_scale_vec)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Odometry Benchmark')
    parser.add_argument(
        'eval_poses', type=str, help='File containing evaluation poses')
    parser.add_argument(
        'dataset_dir', type=str,
        help='Directory containing the reference dataset')
    args = parser.parse_args()

    eval_poses_file = args.eval_poses
    dataset_dir = args.dataset_dir
    evalOdometry(eval_poses_file, dataset_dir)
