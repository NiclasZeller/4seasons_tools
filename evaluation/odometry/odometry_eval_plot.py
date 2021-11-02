import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import os

th_scale = np.array([1.005, 1.01, 1.02])
th_rot = np.array([0.005, 0.01, 0.02])
th_trans = np.array([0.5, 1.0, 2.0])


def calcPercentThreshold(data_vec):
    percent = np.zeros(len(th_trans))
    for i in range(len(th_trans)):
        percent[i] = sum(1 for j in range(len(data_vec[:, 0])) if
                         ((data_vec[j, 0] <= th_trans[i]) and (data_vec[j, 1] <= th_rot[i]) and (data_vec[j, 2] <= th_scale[i])))

    return percent / len(data_vec) * 100.0


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
            errors_list.append((file_name[0], np.array(errors)))

    def sortName(val):
        return val[0]

    errors_list.sort(key=sortName)

    file = open("odometry_results.txt", "w+")
    for i in range(len(errors_list)):
        label = errors_list[i][0]
        percent = calcPercentThreshold((errors_list[i][1][:, 1:]).squeeze())
        file.write("{} {} {} {}\n".format(label.ljust(30), percent[0], percent[1], percent[2]))
    file.close()

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False, figsize=(15, 3.5))

    for i in range(len(errors_list)):
        label = errors_list[i][0]
        cumPlot(ax1, errors_list[i][1][:, 1], 0, 3, label)
    ax1.grid(True)
    ax1.set_xlim([0, 3])
    ax1.set_ylim([0, 100])
    ax1.set_ylabel('occurrence [%]')
    ax1.set_xlabel('translational error [%]')
    ax1.legend(loc='upper center', bbox_to_anchor=(1.7, -0.2), ncol=5, frameon=False, prop={'size': 12})

    for i in range(len(errors_list)):
        label = errors_list[i][0]
        cumPlot(ax2, errors_list[i][1][:, 2] * 1000, 0, 0.025 * 1000, label)
    ax2.grid(True)
    ax2.set_xlim([0, 0.025 * 1000])
    ax2.set_ylim([0, 100])
    ax2.set_ylabel('occurrence [%]')
    ax2.set_xlabel('rotational error [mdeg/m]')

    for i in range(len(errors_list)):
        label = errors_list[i][0]
        cumPlot(ax3, errors_list[i][1][:, 3], 1.0, 1.02, label)
    ax3.grid(True)
    ax3.set_xlim([1.0, 1.02])
    ax3.set_ylim([0, 100])
    ax3.set_ylabel('occurrence [%]')
    ax3.set_xlabel('scale error (multiplier)')
    plt.savefig('odometry_results.pdf', dpi=300, bbox_inches='tight')

    if plot_figures:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Odometry Evaluation')
    parser.add_argument(
        'odometry_errors', type=str)
    parser.add_argument('--plot', action='store_true', help='Plot result graphs')
    args = parser.parse_args()

    plotOdometryError(args.odometry_errors, args.plot)
