import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage import gaussian_filter1d
import numpy as np
import seaborn as sns
from scipy import interpolate
import os

from trackobjects.simulationconstants import SimulationConstants
from trackobjects.symmetricmerge import SymmetricMergingTrack


def vehicle_xy_coordinates(intcolumn, data_csv):
    list_x = []
    list_y = []
    for i in range(len(data_csv.iloc[:, intcolumn])):
        transform_vehicle1 = eval(data_csv.iloc[i, intcolumn])
        x_loc = transform_vehicle1[0]
        y_loc = transform_vehicle1[1]

        list_x.append(x_loc)
        # list_x = list(dict.fromkeys(list_x))
        list_y.append(y_loc)
        # list_y = list(dict.fromkeys(list_y))

    return list_x, list_y


def plot_varjo(path_to_csv_folder, condition):
    simulation_constants = SimulationConstants(vehicle_width=2,
                                               vehicle_length=4.7,
                                               tunnel_length=118,
                                               track_width=8,
                                               track_height=195,
                                               track_start_point_distance=390,
                                               track_section_length_before=275.77164466275354,
                                               track_section_length_after=200)  # goes until 400
    track = SymmetricMergingTrack(simulation_constants)

    files_directory = path_to_csv_folder
    trails = []
    for file in Path(files_directory).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails.append(file)

    all_pds_list = []
    for i in range(len(trails)):
        all_pds = pd.read_csv(trails[i], sep=',')
        all_pds_list.append(all_pds)

    xy_coordinates_for_trace = []
    for i in range(len(all_pds_list)):
        xy_coordinates_vehicle = vehicle_xy_coordinates(2, all_pds_list[i])
        xy_coordinates_zip = [list(a) for a in zip(xy_coordinates_vehicle[0], xy_coordinates_vehicle[1])]
        xy_coordinates_for_trace.append(xy_coordinates_zip)

    # print(xy_coordinates_for_trace[0])

    travelled_distance = []
    for list_index in range(len(xy_coordinates_for_trace)):
        individual_xy_vehicle = xy_coordinates_for_trace[list_index]
        inner_xy_list = []
        for xy in range(len(individual_xy_vehicle)):
            traveled_distance_vehicle = track.coordinates_to_traveled_distance(individual_xy_vehicle[xy])
            inner_xy_list.append(traveled_distance_vehicle)
        travelled_distance.append(inner_xy_list)

    indexes_of_tunnel_and_merge = []
    for list_index in range(len(travelled_distance)):
        individual_travelled_distance_vehicle = travelled_distance[list_index]
        inner_tunnel_merge = []
        for i in range(0, 1):
            index_of_tunnel_vehicle = min(range(len(individual_travelled_distance_vehicle)), key=lambda i: abs(
                individual_travelled_distance_vehicle[i] - track.tunnel_length))
            index_of_mergepoint_vehicle = min(range(len(individual_travelled_distance_vehicle)), key=lambda i: abs(
                individual_travelled_distance_vehicle[i] - simulation_constants.track_section_length_before))
            inner_tunnel_merge.append(index_of_tunnel_vehicle)
            inner_tunnel_merge.append(index_of_mergepoint_vehicle)
        indexes_of_tunnel_and_merge.append(inner_tunnel_merge)

    interactive_area_travelled_traces = []
    hmd_rot_interactive_area = []

    for i in range(len(indexes_of_tunnel_and_merge)):
        hmd_rot = list(all_pds_list[i]['HMD_rotation_vehicle1'][
                       indexes_of_tunnel_and_merge[i][0]:indexes_of_tunnel_and_merge[i][1]])
        interactive_trace = travelled_distance[i][indexes_of_tunnel_and_merge[i][0]:indexes_of_tunnel_and_merge[i][1]]
        hmd_rot_interactive_area.append(hmd_rot)
        interactive_area_travelled_traces.append(interactive_trace)

    on_ramp_vs_opponent = []
    for list_index in range(len(hmd_rot_interactive_area)):
        individual_hmd_rot_list = hmd_rot_interactive_area[list_index]
        inner_attention_list = []
        for i in range(len(individual_hmd_rot_list)):
            if individual_hmd_rot_list[i] > 0.94:  # this we need to know better
                inner_attention_list.append(1)
            else:
                inner_attention_list.append(0)

        on_ramp_vs_opponent.append(inner_attention_list)

    df_traces = pd.DataFrame(interactive_area_travelled_traces)
    x_mean_traces = df_traces.mean()
    df_hmd_rotations = pd.DataFrame(on_ramp_vs_opponent)
    y_mean = df_hmd_rotations.mean()

    # ----------------------option1
    fig, (ax1) = plt.subplots(1)
    #
    # ax1.title('Area of interest over travelled distance')
    ax1.set_xlabel('average travelled distance [m]')
    ax1.set_ylabel('% fixated on AOI')
    # ax1.plot(x_mean_traces[0:500], y_mean_traces[0:500])
    # print(len(x_mean_traces))
    # print(len(y_mean_traces))

    # ysmoothed = gaussian_filter1d(y_mean[0:1100], sigma=4)
    # x = np.linspace(120, 275, len(y_mean[0:1100]))
    # ax1.plot(x, ysmoothed)  # see x below

    # # ----------------------option2
    # d = {'Average travelled distance': [], '% fixation on opponent': []}
    #
    # for i in range(len(x)):
    #     d['Average travelled distance'].append(x[i])
    #     d['% fixation on opponent'].append(y_mean[i])
    #
    # data = pd.DataFrame(d)

    # sns.lineplot(
    #     data=data,
    #     x="Average travelled distance",
    #     y='% fixation on opponent',
    #     palette="tab10", linewidth=1
    # )

    # pd.set_option('display.max_rows', None)
    # find longest column and put in gaussian
    # print(df_traces.iloc[:, 0:1000])
    #
    # print(y_mean.mean())
    # print(y_mean[0:1175])

    ##get median lines
    path_to_data_csv = os.path.join('..', 'data_folder', 'medians_crt_index.csv')
    global_crt_index = pd.read_csv(path_to_data_csv, sep=',')

    if condition == '50-50':
        travelled_distances_on_crt_index = []
        for i in range(len(travelled_distance)):
            value_on_crt_index = travelled_distance[i][global_crt_index.iloc[0][0]]
            travelled_distances_on_crt_index.append(value_on_crt_index)
        average_travelled_distance_on_index_crt = sum(travelled_distances_on_crt_index) / len(
            travelled_distances_on_crt_index)
        plt.axvline(average_travelled_distance_on_index_crt, 0, 1, color='r', label='Average crt')
        plt.title("Condition 50-50")
        ysmoothed = gaussian_filter1d(y_mean[0:1150], sigma=4)
        x = np.linspace(120, 275, len(y_mean[0:1150]))
        ax1.plot(x, ysmoothed)  # see x below

        index_of_average_crt = min(range(len(x)), key=lambda i: abs(
            x[i] - average_travelled_distance_on_index_crt))

        before_crt_average = y_mean[0:index_of_average_crt].mean()
        after_crt_average = y_mean[index_of_average_crt:len(y_mean)].mean()


    elif condition == '55-45':
        travelled_distances_on_crt_index = []
        for i in range(len(travelled_distance)):
            value_on_crt_index = travelled_distance[i][global_crt_index.iloc[0][1]]
            travelled_distances_on_crt_index.append(value_on_crt_index)
        average_travelled_distance_on_index_crt = sum(travelled_distances_on_crt_index) / len(
            travelled_distances_on_crt_index)
        plt.axvline(average_travelled_distance_on_index_crt, 0, 1, color='r', label='Average crt')
        plt.title("Condition 55-45")
        ysmoothed = gaussian_filter1d(y_mean[0:1175], sigma=4)
        x = np.linspace(120, 275, len(y_mean[0:1175]))
        ax1.plot(x, ysmoothed)  # see x below

        index_of_average_crt = min(range(len(x)), key=lambda i: abs(
            x[i] - average_travelled_distance_on_index_crt))

        before_crt_average = y_mean[0:index_of_average_crt].mean()
        after_crt_average = y_mean[index_of_average_crt:len(y_mean)].mean()

    elif condition == '60-40':
        travelled_distances_on_crt_index = []
        for i in range(len(travelled_distance)):
            value_on_crt_index = travelled_distance[i][global_crt_index.iloc[0][2]]
            travelled_distances_on_crt_index.append(value_on_crt_index)
        average_travelled_distance_on_index_crt = sum(travelled_distances_on_crt_index) / len(
            travelled_distances_on_crt_index)
        plt.axvline(average_travelled_distance_on_index_crt, 0, 1, color='r', label='Average crt')
        plt.title("Condition 60-40")
        ysmoothed = gaussian_filter1d(y_mean[0:1175], sigma=4)
        x = np.linspace(120, 275, len(y_mean[0:1175]))
        ax1.plot(x, ysmoothed)  # see x below

        index_of_average_crt = min(range(len(x)), key=lambda i: abs(
            x[i] - average_travelled_distance_on_index_crt))

        before_crt_average = y_mean[0:index_of_average_crt].mean()
        after_crt_average = y_mean[index_of_average_crt:len(y_mean)].mean()



    # plt.plot(y, c='red')
    ax1.fill_between(x, ysmoothed, color='blue', alpha=0.1, label='Fixation on road')
    ax1.fill_between(x, ysmoothed, 1, color='red', alpha=0.1, label='Fixation on opponent')

    ax1.plot([], [], ' ', label='Average before crt: ' + str(round(before_crt_average, 2)))
    ax1.plot([], [], ' ', label='Average after crt: ' + str(round(after_crt_average, 2)))

    ax1.set_xlim([120, 275])
    ax1.set_ylim([0, 1])
    ax1.legend(loc='lower right')

    plt.show()


if __name__ == '__main__':
    #
    plot_varjo(
        r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\condition_60_40',
        '60-40')
    plot_varjo(
        r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\condition_55_45',
        '55-45')
    plot_varjo(
        r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\condition_50_50',
        '50-50')

    # #test
    # plot_varjo(r'C:\Users\loran\Desktop\ExperimentOlgerArkady\Joan.Varjo.combined', '60-40')
