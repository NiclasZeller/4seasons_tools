import argparse
import numpy as np
import pynmea2
import os.path
import configparser
from datetime import datetime
import calendar
import matplotlib.pyplot as plt
from scipy import stats
from libartipy.dataset import Dataset, Constants
from libartipy.geometry import CoordinateSystem


def readGnssMeasurements(nmea_file):
    def invalidatePackages():
        nonlocal NMEA_GST_received
        nonlocal NMEA_GGA_received
        nonlocal NMEA_RMC_received
        NMEA_GST_received = False
        NMEA_GGA_received = False
        NMEA_RMC_received = False

    def checkUTCTime(time):
        nonlocal timestamp
        if not (time == timestamp):
            invalidatePackages()
            timestamp = time

    def readGST(msg):
        nonlocal NMEA_GST_received
        nonlocal lat_std, lon_std, alt_std

        checkUTCTime(msg.timestamp)

        if (msg.std_dev_latitude is not None) and (msg.std_dev_longitude is not None) and (
                msg.std_dev_altitude is not None):
            lat_std = msg.std_dev_latitude
            lon_std = msg.std_dev_longitude
            alt_std = msg.std_dev_altitude
            NMEA_GST_received = True

    def readGGA(msg):
        nonlocal NMEA_GGA_received
        nonlocal lat, lon, alt

        checkUTCTime(msg.timestamp)

        if None not in [msg.latitude, msg.longitude, msg.altitude, msg.geo_sep]:
            lat = msg.latitude
            lon = msg.longitude
            alt = msg.altitude + float(msg.geo_sep)
            NMEA_GGA_received = True

    def readRMC(msg):
        nonlocal NMEA_RMC_received
        nonlocal datestamp

        checkUTCTime(msg.timestamp)
        datestamp = msg.datestamp

        NMEA_RMC_received = True

    print("read GNSS-data from NMEA-file")

    NMEA_GST_received = False
    NMEA_GGA_received = False
    NMEA_RMC_received = False

    gnss_points = list()

    timestamp = -1
    datestamp = -1
    lat = -1
    lon = -1
    alt = -1

    lat_std = -1
    lon_std = -1
    alt_std = -1

    file = open(nmea_file, "r")
    # ignore first line, since it allways seems to be incomplete.
    file.readline()
    streamreader = pynmea2.NMEAStreamReader(file)
    process = True
    while process:
        process = False
        for msg in streamreader.next():
            process = True

            if msg.sentence_type == 'GST':
                readGST(msg)
            else:
                if msg.sentence_type == 'GGA':
                    readGGA(msg)
                else:
                    if msg.sentence_type == 'RMC':
                        readRMC(msg)

            if NMEA_GST_received and NMEA_GGA_received and NMEA_RMC_received:
                dd = datetime(datestamp.year, datestamp.month, datestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second)
                timestamp_ns = np.uint64(calendar.timegm(dd.timetuple())) * np.uint64(1e9) + np.uint64(
                    timestamp.microsecond * 1e3)

                gnss_points.append(
                    (timestamp_ns, lat, lon, alt, lat_std, lon_std, alt_std))
                invalidatePackages()

    return np.array(gnss_points, dtype=np.float128)


def filterGnssPoints(gnss_points, var_threshold):
    gnss_points_filtered = list()
    for i in range(len(gnss_points[:, 0])):
        if ((gnss_points[i, 4] * gnss_points[i, 4]
             + gnss_points[i, 5] * gnss_points[i, 5]) <= var_threshold):
            gnss_points_filtered.append((gnss_points[i, :]))

    return np.array(gnss_points_filtered)


def alignData(gt_points, eval_points):
    gt_aligned = list()
    eval_aligned = list()
    ts_low = 0
    ts_high = 0
    ii_low = 0
    ii_high = 0
    for i in range(len(eval_points[:, 0])):
        ts = eval_points[i, 0]

        for ii in range(ii_low, len(gt_points[:, 0])):
            ts_gt = gt_points[ii, 0]

            if ts_gt <= ts:
                ts_low = ts_gt
                ii_low = ii
            else:
                if ts_gt > ts:
                    ts_high = ts_gt
                    ii_high = ii
                    break

        if (abs(float(ts_low) - float(ts)) < 1e9) and (abs(float(ts_high) - float(ts)) < 1e9):
            x_low = gt_points[ii_low, 1]
            y_low = gt_points[ii_low, 2]
            z_low = gt_points[ii_low, 3]

            x_high = gt_points[ii_high, 1]
            y_high = gt_points[ii_high, 2]
            z_high = gt_points[ii_high, 3]

            # interpolate position
            x = x_low + (x_high - x_low) / (ts_high - ts_low) * (ts - ts_low)
            y = y_low + (y_high - y_low) / (ts_high - ts_low) * (ts - ts_low)
            z = z_low + (z_high - z_low) / (ts_high - ts_low) * (ts - ts_low)

            std_lat = max(gt_points[ii_low, 4], gt_points[ii_high, 4])
            std_lon = max(gt_points[ii_low, 5], gt_points[ii_high, 5])
            std_alt = max(gt_points[ii_low, 6], gt_points[ii_high, 6])
            gt_aligned.append((ts, x, y, z, std_lat, std_lon, std_alt))
            eval_aligned.append((eval_points[i, :]))

        ts_low = 0
        ts_high = 0

    return np.array(gt_aligned), np.array(eval_aligned)


def calcHorizontalDistance(gt_points, eval_points):
    error_vec = np.empty(0)
    for i in range(len(gt_points[:, 0])):
        if gt_points[i, 0] != eval_points[i, 0]:
            continue
        err = np.linalg.norm(gt_points[i, 1:3] - eval_points[i, 1:3], 2)
        error_vec = np.append(error_vec, err)
    return error_vec


def readDsoPoses(dir_results, gnss_calib=np.zeros(3)):
    # read keyframe data
    constants = Constants()
    constants.GPS_POSES_FILE_NAME = 'GNSSPoses.txt'
    data = Dataset(dir_results, constants=constants)

    data.parse_keyframes()
    data.set_keyframe_poses_to_gps_poses()
    poses_enu = data.get_keyframe_poses_in_coordinate_system(
        coordinate_system=CoordinateSystem.ENU)

    transformation_frame = data.get_transformation_frame()

    scales_metric_pseudometric = data.get_rotation_scales_from_frames()

    T_visual_pseudometric = transformation_frame.transformations[(
        CoordinateSystem.SLAM_WORLD, CoordinateSystem.WORLD)].trans_mat.inverse().transformation_matrix

    point_enu = list()
    for i, (k, v) in enumerate(poses_enu.items()):
        pose = np.matmul(v.transformation_matrix, T_visual_pseudometric)
        translation_gnss = pose[:3, 3] + \
            np.dot(pose[:3, :3], gnss_calib) / scales_metric_pseudometric[i]

        point_enu.append(
            (k, translation_gnss[0], translation_gnss[1], translation_gnss[2]))

    return np.array(point_enu, dtype=np.float128), transformation_frame


def readGnssCalibration(gps_config):
    if os.path.isfile(gps_config):
        # add dummy section, since sections are missing in gps_config.cfg
        dummy_section = 'top'
        with open(gps_config) as config_file:
            config_string = '[' + dummy_section + ']\n' + config_file.read()
        config = configparser.ConfigParser()
        config.read_string(config_string)
        gps_calib_str = config.get(dummy_section, "t_mms_gps")
        gps_calib_str = gps_calib_str.replace('(', '')
        gps_calib_str = gps_calib_str.replace(')', '')
        gps_calib = np.array(gps_calib_str.split(','), dtype=np.float64)
        return np.array([gps_calib[1], -gps_calib[2], -gps_calib[0]], dtype=np.float64)
    else:
        return np.zeros(3, dtype=np.float64)


def evalMapAccuracy(input_file, nmea_file, gps_config, var_threshold, only_accurate_kf, output_dir, plot_results,
                    save_figs):
    # read gnss calibration
    gnss_calib = readGnssCalibration(gps_config)
    print(gnss_calib)
    print("GNSS-Antenna calibration is set to: tx={0:f}, ty={1:f}, tz={2:f} (in camera coordinates)".format(
        gnss_calib[0], gnss_calib[1], gnss_calib[2]))

    # read gnss measurements
    gnss_points_wgs = readGnssMeasurements(nmea_file)
    print("total number of gnss points: {0:d}".format(
        len(gnss_points_wgs[:, 0])))

    # use only accurate measurements
    gnss_points_wgs = filterGnssPoints(gnss_points_wgs, var_threshold)
    print("total number of gnss points after filtering: {0:d}".format(
        len(gnss_points_wgs[:, 0])))

    # read DSO poses in ENU frame
    eval_points_enu, transformation_frame = readDsoPoses(
        input_file, gnss_calib)
    n_kf = len(eval_points_enu[:, 0])
    print("total number of eval points: {0:d}".format(n_kf))

    # convert gnss points to ENU frame
    gnss_points_ecef = gnss_points_wgs
    gnss_points_ecef[:, 1:4] = transformation_frame.transform_from_WGS84_to_ECEF(
        gnss_points_wgs[:, 1:4])
    gnss_points_enu = gnss_points_wgs

    T_ecef_enu = transformation_frame.transformations[(
        CoordinateSystem.ENU, CoordinateSystem.ECEF)].trans_mat
    T_enu_ecef = T_ecef_enu.inverse()
    for i in range(len(gnss_points_ecef[:, 0])):
        pt_ecef = gnss_points_ecef[i, 1:4]
        pt_enu = T_enu_ecef.rotation_matrix.dot(
            pt_ecef) + T_enu_ecef.translation
        gnss_points_enu[i, 1:4] = pt_enu

    # align timestamps
    eval_points_enu_full_trajectory = eval_points_enu
    gnss_points_enu, eval_points_enu = alignData(
        gnss_points_enu, eval_points_enu)
    print("total number of aligned points: {0:d}".format(
        len(gnss_points_enu[:, 0])))

    # calculate horizontal distance
    error_vec = calcHorizontalDistance(gnss_points_enu, eval_points_enu)

    # store keyframe accuracies to file
    std_threshold = np.sqrt(var_threshold)
    n_accurate = 0
    for i in range(len(error_vec)):
        if error_vec[i] <= std_threshold:
            n_accurate = n_accurate + 1

    if (len(output_dir) > 0) and (output_dir[-1] != '/'):
        output_dir = output_dir + '/'

    f = open(output_dir + 'keyframe_accuracy.txt', 'w+')
    f.write('#Keyframe with accuracy better than {:1.4f} m: {:d} of {:d}\n\n'.format(np.sqrt(var_threshold), n_accurate,
                                                                                     n_kf))
    f.write('# timestamp_ns std_gnss_north_m std_gnss_east_m horizontal_diff_m\n')
    good_eval_points_enu = []

    for i in range(len(error_vec)):
        if not only_accurate_kf or (error_vec[i] <= std_threshold):
            f.write(
                '{:d} {:1.4f} {:1.4f} {:1.4f}\n'.format(gnss_points_enu[i, 0].astype(np.uint64), gnss_points_enu[i, 4],
                                                        gnss_points_enu[i, 5], error_vec[i]))
            good_eval_points_enu.append(eval_points_enu[i])

    good_eval_points_enu = np.array(good_eval_points_enu).squeeze()

    f.close()

    print('*** Map Accuracy ***')
    rmse = np.sqrt(np.sum(np.multiply(error_vec, error_vec)) / len(error_vec))
    mae = np.max(error_vec)
    print("Horizontal RMSE: {:f} m".format(rmse))
    print("Horizontal MAE: {:f} m".format(mae))

    if plot_results or save_figs:
        plt.figure(dpi=120)
        plt.plot(gnss_points_enu[:, 0], error_vec, '.')
        plt.title('Map Accuray')
        plt.xlabel('time [ns]')
        plt.ylabel('horizontal distance to GNSS [m]')
        plt.show(block=False)
        if save_figs:
            plt.savefig(output_dir + 'map_accuracy_keyframe_error.png')

        low_lim = 0
        up_lim = 0.5
        n = 1000
        cum_error = stats.cumfreq(
            error_vec, numbins=n + 1, defaultreallimits=(low_lim - (up_lim - low_lim) / n, up_lim))
        # plot with n+1 bins is generated,
        # since we also want to know the number of measurements below the lower threshold
        x = np.linspace(cum_error.lowerlimit + cum_error.binsize, cum_error.binsize * cum_error.cumcount.size,
                        cum_error.cumcount.size)
        cum_error = cum_error.cumcount

        plt.figure(dpi=120)
        plt.plot(x, cum_error / n_kf * 100.0, label='Map Error')
        plt.plot(x, np.ones(len(x)) * 100.0, 'g')
        plt.plot(x, np.ones(len(x)) / n_kf * len(error_vec) * 100.0, 'r',
                 label='Accurate GNSS (std < {:.3f} m)'.format(std_threshold))
        plt.grid()
        plt.title('Cummulative Map Error')
        plt.xlabel('horizontal distance to GNSS [m]')
        plt.ylabel('percentage of keyframes')
        plt.legend()
        plt.show(block=False)
        if save_figs:
            plt.savefig(output_dir + 'map_accuracy_cummulative_error.png')

        plt.figure(dpi=120)
        plt.plot(gnss_points_enu[:, 1],
                 gnss_points_enu[:, 2], 'g*', label='GNSS')
        plt.plot(eval_points_enu_full_trajectory[:, 1],
                 eval_points_enu_full_trajectory[:, 2], 'r', label='Keyframe')
        plt.title('Trajectory')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.legend()
        if save_figs:
            plt.show(block=False)
            plt.savefig(output_dir + 'map_accuracy_trajectory_coverage.png')

        plt.figure(dpi=120, figsize=(15, 10))
        plt.plot(gnss_points_enu[:, 1],
                 gnss_points_enu[:, 2], 'o', color='g', label='GNSS', markersize=8)
        plt.plot(good_eval_points_enu[:, 1],
                 good_eval_points_enu[:, 2], '.', color='r', label='Keyframe',  markersize=5)
        plt.title('trajectory coverage good Keyframe vs. GNSS')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.legend()
        if save_figs:
            plt.show(block=False)
            plt.savefig(
                output_dir + 'map_accuracy_good_keyframe_vs_gnss_coverage.png')
        if plot_results:
            plt.show()

    return error_vec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Global Map Accuracy Evaluation')
    parser.add_argument(
        'input', type=str, help='DSO results')
    parser.add_argument(
        'nmea', type=str, help='GNSS measurements')
    parser.add_argument('--gps_config', nargs='?', type=str, default='',
                        help='GPS config file including calibration of GNSS antenna')
    parser.add_argument('--output', nargs='?', type=str, default='',
                        help='Directory to store keyframe_accuracy.txt')
    parser.add_argument('--gps_var_threshold', nargs='?', type=float, default=0.0025,
                        help='Horizontal GNSS variance threshold (default = 0.0025)')
    parser.add_argument('--only_accurate_kf', action='store_true',
                        help='Store only keyframes with accurate poses')
    parser.add_argument('--plot', action='store_true',
                        help='Plot result graphs')
    parser.add_argument('--save_figs', action='store_true',
                        help='Save figures to output directory')

    args = parser.parse_args()
    input_file = args.input
    nmea_file = args.nmea
    gps_config = args.gps_config
    var_threshold = args.gps_var_threshold
    only_accurate_kf = args.only_accurate_kf
    plot_results = args.plot
    save_figs = args.save_figs
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    errors = evalMapAccuracy(input_file, nmea_file, gps_config,
                             var_threshold, only_accurate_kf, output_dir, plot_results, save_figs)
