import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import os


def cumPlot(ax, data_vec, low_lim, up_lim, label, n_bins=1000):
    cum_data = stats.cumfreq(data_vec, numbins=n_bins + 1,
                             defaultreallimits=(low_lim - (up_lim - low_lim) / n_bins, up_lim))
    # plot with n+1 bins is generated, since we also want to know the number of measurements below the lower threshold
    x = np.linspace(cum_data.lowerlimit + cum_data.binsize,
                    cum_data.lowerlimit + cum_data.binsize + cum_data.binsize * cum_data.cumcount.size,
                    cum_data.cumcount.size)
    cum_data = cum_data.cumcount
    ax.plot(x, cum_data / len(data_vec) * 100, label=label, rasterized=False, linewidth=2.0)


def plotOdometryError(dir_error, plot_figures):
    errors_list = list()
    with os.scandir(dir_error) as entries:
        for entry in entries:
            ext = os.path.splitext(entry.name)[1]
            if ext != '.txt':
                continue

            errors = list()
            with open(entry.path, 'r') as file:
                for line in file.readlines():
                    items = line.split()
                    assert (len(items) == 4)
                    dist = float(items[0])
                    e_trans = float(items[1])
                    e_rot = float(items[2])
                    e_scale = float(items[3])
                    errors.append((dist, e_trans, e_rot, e_scale))

            file_name = entry.name.split('.')

            if file_name[0] == 'dso_stereo_inertial':
                name = 'ArtiSLAM (stereo-inertial)'
            elif file_name[0] == 'basalt_stereo':
                name = 'Basalt (stereo)'
            elif file_name[0] == 'basalt_stereo_inertial':
                name = 'Basalt (stereo-inertial)'
            elif file_name[0] == 'orb_slam_stereo':
                name = 'ORB-SLAM (stereo)'
            else:
                continue

            #errors_list.append((file_name[0], np.array(errors)))
            errors_list.append((name, np.array(errors)))

    def sortName(val):
        return val[0]

    errors_list.sort(key=sortName)

    #f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False, figsize=(15, 3.5))
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(15, 3.5))

    for i in range(len(errors_list)):
        label = errors_list[i][0]
        cumPlot(ax1, errors_list[i][1][:, 1], 0, 3, label)
    ax1.grid(True)
    ax1.set_xlim([0, 3])
    ax1.set_ylim([0, 100])
    ax1.set_ylabel('occurrence [%]')
    ax1.set_xlabel('translational error [%]')
#    ax1.legend(loc='upper center', bbox_to_anchor=(1.7, -0.2), ncol=5, frameon=False, prop={'size': 12})
    ax1.legend(loc='upper left', bbox_to_anchor=(0, -0.2), ncol=5, frameon=False, prop={'size': 12})

    for i in range(len(errors_list)):
        label = errors_list[i][0]
        cumPlot(ax2, errors_list[i][1][:, 2] * 1000, 0, 0.025 * 1000, label)
    ax2.grid(True)
    ax2.set_xlim([0, 0.025 * 1000])
    ax2.set_ylim([0, 100])
    ax2.set_ylabel('occurrence [%]')
    ax2.set_xlabel('rotational error [mdeg/m]')

    # for i in range(len(errors_list)):
    #     label = errors_list[i][0]
    #     cumPlot(ax3, errors_list[i][1][:, 3], 1.0, 1.02, label)
    # ax3.grid(True)
    # ax3.set_xlim([1.0, 1.02])
    # ax3.set_ylim([0, 100])
    # ax3.set_ylabel('occurrence [%]')
    # ax3.set_xlabel('scale error (multiplier)')
    plt.savefig('odometry_results.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('odometry_results.png', dpi=300, bbox_inches='tight')

    if plot_figures:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Odometry Evaluation')
    parser.add_argument(
        'odometry_errors', type=str)
    parser.add_argument('--plot', action='store_true', help='Plot result graphs')
    args = parser.parse_args()

    plotOdometryError(args.odometry_errors, args.plot)
