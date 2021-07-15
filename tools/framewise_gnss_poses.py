import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from libartipy.dataset import Dataset, Constants
from libartipy.geometry import CoordinateSystem, Pose, Quaternion


def readTimeStamps(filename):
    timestamps = list()
    with open(filename, 'r') as file:
        for line in file.readlines():
            items = line.split()
            assert (len(items) == 3)
            ts = np.uint64(items[0])
            timestamps.append(ts)

    timestamps = np.array(timestamps)
    timestamps.sort()

    return timestamps


def findNext(ts, times):
    ts = np.int64(ts)
    idx = -1
    for i in range(len(times)):
        if times[i] >= ts:
            delta = np.uint64(abs(times[i] - ts))
            idx = i
            if i > 0:
                delta_pre = np.uint64(abs(times[i - 1] - ts))
                if delta_pre < delta:
                    delta = delta_pre
                    idx = i - 1
            break

    if idx < 0:
        idx = len(times) - 1
        delta = np.uint64(abs(times[idx] - ts))

    return idx, delta


def readPoses(filename, timestamps_ns):
    poses = dict()
    timestamps = list()

    idx_ts = 0
    with open(filename, 'r') as file:
        for line in file.readlines():
            items = line.split()
            assert (len(items) == 8)
            ts = float(items[0])

            delta_idx, delta_t = findNext(np.uint64(ts * 1e9), timestamps_ns[idx_ts:])

            idx_ts += delta_idx
            if delta_t > 2e6:
                print(delta_t)
                print("timestamp difference larger than 2ms")

            x = float(items[1])
            y = float(items[2])
            z = float(items[3])
            qx = float(items[4])
            qy = float(items[5])
            qz = float(items[6])
            qw = float(items[7])

            timestamps.append(timestamps_ns[idx_ts])
            pose = Pose(Quaternion(np.array([qw, qx, qy, qz])), np.array([x, y, z]))
            poses[timestamps_ns[idx_ts]] = pose

    timestamps = np.array(timestamps)
    timestamps.sort()

    return timestamps, poses


def framewiseGnssPoses(dataset_dir, frame, plot):
    constants = Constants()
    dataset = Dataset(dataset_dir, constants=constants)
    dataset.parse_keyframes()
    dataset.set_keyframe_poses_to_gps_poses()

    if frame == 'enu':
        kf_poses = dataset.get_keyframe_poses_in_coordinate_system(
            coordinate_system=CoordinateSystem.ENU)
    elif frame == 'ecef':
        kf_poses = dataset.get_keyframe_poses_in_coordinate_system(
            coordinate_system=CoordinateSystem.ECEF)

    kf_timestamps = dataset.get_all_kf_timestamps()

    rt_timestamps = readTimeStamps(os.path.join(dataset_dir, 'times.txt'))
    rt_timestamps, rt_poses = readPoses(os.path.join(dataset_dir, 'result.txt'), rt_timestamps)
    print('{} poses read'.format(len(rt_timestamps)))

    rt_poses_out = list()
    idx_kf = -1
    ts_post = -1
    for ts in rt_timestamps:
        if ts >= ts_post and len(kf_timestamps) > idx_kf + 1:
            idx_kf += 1
            ts_pre = kf_timestamps[idx_kf]
            if len(kf_timestamps) > idx_kf + 1:
                ts_post = kf_timestamps[idx_kf + 1]
            else:
                ts_post = -1

        pose_pre = kf_poses[ts_pre]
        scale_pre = dataset.get_gps_pose_with_timestamp(ts_pre).get_translation_scale()
        if ts == ts_pre:
            # no interpolation required
            pose_out = pose_pre
        else:
            # do interpolation only based on previous pose
            pose_rel = rt_poses[ts_pre].inverse() * rt_poses[ts]
            pose_rel.translation *= scale_pre
            pose_pr = pose_pre * pose_rel

            # interpolation between previous and next pose seems to result in instabilities in some case
            # might be worth to check why this is the case
            if ts < ts_pre or ts_post < 0:
                pose_out = pose_pr
            else:
                # do interpolation based on next pose
                pose_post = kf_poses[ts_post]
                pose_rel = rt_poses[ts_post].inverse() * rt_poses[ts]
                pose_rel.translation *= scale_pre
                pose_pt = pose_post * pose_rel

                delta_t = float(ts - ts_pre) / float(ts_post - ts_pre)
                mean_tangent = pose_pt.inverse().log() * delta_t + pose_pr.inverse().log() * (1 - delta_t)
                pose_out = Pose.exp(mean_tangent).inverse()

        rt_poses_out.append((ts, pose_out.translation[0], pose_out.translation[1], pose_out.translation[2],
                             pose_out.rotation_quaternion.x, pose_out.rotation_quaternion.y,
                             pose_out.rotation_quaternion.z, pose_out.rotation_quaternion.w))

        print('interpolate pose {}'.format(ts))

    with open(os.path.join(dataset_dir, 'GNSSresult_' + frame + '.txt'), 'w') as file:
        for rt_pose in rt_poses_out:
            file.write('{0[0]:d} {0[1]:1.10f} {0[2]:1.10f} {0[3]:1.10f} {0[4]:1.10f} {0[5]:1.10f} {0[6]:1.10f} {0['
                       '7]:1.10f}\n'.format(rt_pose))

    if plot:
        kf_pos = np.array([kf_poses[ts].translation for ts in kf_timestamps])
        rt_poses_out = np.array(rt_poses_out)
        plt.figure(dpi=120)
        ax = plt.axes(projection='3d')
        ax.plot(rt_poses_out[:, 1], rt_poses_out[:, 2], rt_poses_out[:, 3])
        plt.plot(kf_pos[:, 0], kf_pos[:, 1], kf_pos[:, 2], 'x')

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculated frame-wise ')
    parser.add_argument(
        'dataset_dir', type=str, help='Path to 4Seasons dataset sequence')
    parser.add_argument('--plot', action='store_true',
                        help='Plot trajectory')
    parser.add_argument('--frame', type=str, help='Output coordinate frame (ecef or enu)',
                        default='enu', choices=['ecef', 'enu'])
    args = parser.parse_args()

    framewiseGnssPoses(args.dataset_dir, args.frame, args.plot)
