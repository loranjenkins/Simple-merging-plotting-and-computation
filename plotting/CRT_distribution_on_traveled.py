import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from natsort import natsorted
import seaborn as sns
import numpy as np

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


def interpolate(inp, fi):
    i, f = int(fi // 1), fi % 1  # Split floating-point index into whole & fractional parts.
    j = i + 1 if f > 0 else i  # Avoid index error.
    return (1 - f) * inp[i] + f * inp[j]


def average(l):
    llen = len(l)

    def divide(x): return x / llen

    return map(divide, map(sum, zip(*l)))


def check_if_on_collision_course_for_point(travelled_distance_collision_point, data_dict, simulation_constants):
    track = SymmetricMergingTrack(simulation_constants)
    point_predictions = {'vehicle1': [], 'vehicle2': []}

    point_predictions['vehicle1'] = np.array(data_dict['distance_traveled_vehicle1']) + np.array(
        data_dict['velocity_vehicle1']) * (travelled_distance_collision_point - np.array(
        data_dict['distance_traveled_vehicle2'])) / np.array(
        data_dict['velocity_vehicle2'])
    point_predictions['vehicle2'] = np.array(data_dict['distance_traveled_vehicle2']) + np.array(
        data_dict['velocity_vehicle2']) * (travelled_distance_collision_point - np.array(
        data_dict['distance_traveled_vehicle1'])) / np.array(
        data_dict['velocity_vehicle1'])

    lb, ub = track.get_collision_bounds(travelled_distance_collision_point, simulation_constants.vehicle_width,
                                        simulation_constants.vehicle_length)

    on_collision_course = ((lb < point_predictions['vehicle1']) & (point_predictions['vehicle1'] < ub)) | \
                          ((lb < point_predictions['vehicle2']) & (point_predictions['vehicle2'] < ub))

    return on_collision_course[0]


def calculate_conflict_resolved_time(data_dict, simulation_constants, condition):
    time = data_dict['time']
    track = SymmetricMergingTrack(simulation_constants)

    merge_point_collision_course = check_if_on_collision_course_for_point(track.section_length_before, data_dict,
                                                                          simulation_constants)

    threshold_collision_course = check_if_on_collision_course_for_point(track.upper_bound_threshold + 1e-3, data_dict,
                                                                        simulation_constants)
    # 1e-3 is used for straight approach (always inside)
    end_point_collision_course = check_if_on_collision_course_for_point(
        track.section_length_before + track.section_length_after,
        data_dict, simulation_constants)

    on_collision_course = merge_point_collision_course \
                          | threshold_collision_course \
                          | end_point_collision_course

    # on_collision_course = merge_point_collision_course \
    #                       | end_point_collision_course

    if condition == '50-50':
        approach_mask = ((np.array(data_dict['distance_traveled_vehicle1']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle1']) < 304)) | \
                        ((np.array(data_dict['distance_traveled_vehicle2']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle2']) < 304))
        indices_of_conflict_resolved = (on_collision_course & approach_mask)

    if condition == '55-45':
        approach_mask = ((np.array(data_dict['distance_traveled_vehicle1']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle1']) < 304)) | \
                        ((np.array(data_dict['distance_traveled_vehicle2']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle2']) < 304))
        indices_of_conflict_resolved = (on_collision_course & approach_mask)

    if condition == '60-40':
        approach_mask = ((np.array(data_dict['distance_traveled_vehicle1']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle1']) < 290)) | \
                        ((np.array(data_dict['distance_traveled_vehicle2']) > track.tunnel_length) &
                         (np.array(data_dict['distance_traveled_vehicle2']) < 290))
        indices_of_conflict_resolved = (on_collision_course & approach_mask)

    try:
        time_of_conflict_resolved = np.array(time)[indices_of_conflict_resolved]
    except IndexError:
        time_of_conflict_resolved = None

    return indices_of_conflict_resolved, time_of_conflict_resolved


def plot_varjo(path_to_csv_folder, condition):
    simulation_constants = SimulationConstants(vehicle_width=1.5,
                                               vehicle_length=4.7,
                                               tunnel_length=135,
                                               track_width=8,
                                               track_height=230,
                                               track_start_point_distance=460,
                                               track_section_length_before=325.27,
                                               track_section_length_after=120)

    track = SymmetricMergingTrack(simulation_constants)

    files_directory = path_to_csv_folder
    trails = []
    for file in Path(files_directory).glob('*.csv'):
        # trail_condition = plot_trail(file)
        trails.append(file)
    trails = natsorted(trails, key=str)

    all_pds_list = []
    for i in range(len(trails)):
        all_pds = pd.read_csv(trails[i], sep=',')
        all_pds_list.append(all_pds)

    xy_coordinates_for_vehicle1 = []
    xy_coordinates_for_vehicle2 = []

    for i in range(len(all_pds_list)):
        xy_coordinates_vehicle = vehicle_xy_coordinates(2, all_pds_list[i])
        xy_coordinates_vehicle2 = vehicle_xy_coordinates(5, all_pds_list[i])

        xy_coordinates_zip = [list(a) for a in zip(xy_coordinates_vehicle[0], xy_coordinates_vehicle[1])]
        xy_coordinates_zip2 = [list(a) for a in zip(xy_coordinates_vehicle2[0], xy_coordinates_vehicle2[1])]

        xy_coordinates_for_vehicle1.append(xy_coordinates_zip)
        xy_coordinates_for_vehicle2.append(xy_coordinates_zip2)

    # compute travelled distance for indexes tunnel and merge
    travelled_distance_vehicle1 = []
    travelled_distance_vehicle2 = []
    for list_index in range(len(xy_coordinates_for_vehicle1)):
        individual_xy_vehicle1 = xy_coordinates_for_vehicle1[list_index]
        individual_xy_vehicle2 = xy_coordinates_for_vehicle2[list_index]
        inner_xy_list_1 = []
        inner_xy_list_2 = []
        for xy in range(len(individual_xy_vehicle1)):
            traveled_distance_vehicle_1 = track.coordinates_to_traveled_distance(individual_xy_vehicle1[xy])
            traveled_distance_vehicle_2 = track.coordinates_to_traveled_distance(individual_xy_vehicle2[xy])
            inner_xy_list_1.append(traveled_distance_vehicle_1)
            inner_xy_list_2.append(traveled_distance_vehicle_2)

        travelled_distance_vehicle1.append(inner_xy_list_1)
        travelled_distance_vehicle2.append(inner_xy_list_2)

    # compute the indexes
    indexes_of_tunnel_and_merge_vehicle1 = []
    indexes_of_tunnel_and_merge_vehicle2 = []
    for list_index in range(len(travelled_distance_vehicle1)):
        individual_trace_vehicle1 = travelled_distance_vehicle1[list_index]
        individual_trace_vehicle2 = travelled_distance_vehicle2[list_index]
        inner_index_list_1 = []
        inner_index_list_2 = []
        for i in range(0, 1):
            index_of_tunnel_vehicle_1 = min(range(len(individual_trace_vehicle1)), key=lambda i: abs(
                individual_trace_vehicle1[i] - track.tunnel_length))

            index_of_tunnel_vehicle_2 = min(range(len(individual_trace_vehicle2)), key=lambda i: abs(
                individual_trace_vehicle2[i] - track.tunnel_length))

            index_of_mergepoint_vehicle1 = min(range(len(individual_trace_vehicle1)),
                                               key=lambda i: abs(
                                                   individual_trace_vehicle1[i] - 400))
            index_of_mergepoint_vehicle2 = min(range(len(individual_trace_vehicle2)),
                                               key=lambda i: abs(
                                                   individual_trace_vehicle2[i] - 400))

            inner_index_list_1.append(index_of_tunnel_vehicle_1)
            inner_index_list_1.append(index_of_mergepoint_vehicle1)

            inner_index_list_2.append(index_of_tunnel_vehicle_2)
            inner_index_list_2.append(index_of_mergepoint_vehicle2)

        indexes_of_tunnel_and_merge_vehicle1.append(inner_index_list_1)
        indexes_of_tunnel_and_merge_vehicle2.append(inner_index_list_2)

    # interactive data for each vehicle
    interactive_area_travelled_trace_vehicle1 = []
    interactive_area_travelled_trace_vehicle2 = []
    combined_trace = []

    for i in range(len(indexes_of_tunnel_and_merge_vehicle1)):
        # interactive_trace_1 = travelled_distance_vehicle1[i][
        #                       0:indexes_of_tunnel_and_merge_vehicle1[i][1]] #---- do this for crt computer full trail / crt_index_first_exit_full
        # interactive_trace_2 = travelled_distance_vehicle2[i][
        #                       0:indexes_of_tunnel_and_merge_vehicle2[i][1]]

        interactive_trace_1 = travelled_distance_vehicle1[i][
                              indexes_of_tunnel_and_merge_vehicle1[i][0]:indexes_of_tunnel_and_merge_vehicle1[i][1]]
        interactive_trace_2 = travelled_distance_vehicle2[i][
                              indexes_of_tunnel_and_merge_vehicle2[i][0]:indexes_of_tunnel_and_merge_vehicle2[i][1]]

        interactive_trace_1_ = travelled_distance_vehicle1[i]
        interactive_trace_2_ = travelled_distance_vehicle2[i]
        average_travelled_distance_trace = list((np.array(interactive_trace_1_) + np.array(
            interactive_trace_2_)) / 2.)

        interactive_area_travelled_trace_vehicle1.append(interactive_trace_1)
        interactive_area_travelled_trace_vehicle2.append(interactive_trace_2)
        combined_trace.append(average_travelled_distance_trace)

    path_to_data_csv = os.path.join('..', 'data_folder', 'crt_index_who_is_first_exit_interactive.csv')

    global_crt_index = pd.read_csv(path_to_data_csv, sep=',')

    count_50_50 = []
    if condition == '50-50':
        for i in range(len(interactive_area_travelled_trace_vehicle1)):
            count_combined = combined_trace[i][round(global_crt_index['crt_index_50_50'][i])]
            count_50_50.append(count_combined)

    count_55_45_v1 = []
    count_55_45_v2 = []

    if condition == '55-45':
        for i in range(len(interactive_area_travelled_trace_vehicle1)):
            count_v1 = interactive_area_travelled_trace_vehicle1[i][round(global_crt_index['crt_index_55_45'][i])]
            count_v2 = interactive_area_travelled_trace_vehicle2[i][round(global_crt_index['crt_index_55_45'][i])]
            count_55_45_v1.append(count_v1)
            count_55_45_v2.append(count_v2)

    count_60_40_v1 = []
    count_60_40_v2 = []
    if condition == '60-40':
        for i in range(len(interactive_area_travelled_trace_vehicle1)):
            count_v1 = interactive_area_travelled_trace_vehicle1[i][round(global_crt_index['crt_index_60_40'][i])]
            count_v2 = interactive_area_travelled_trace_vehicle2[i][round(global_crt_index['crt_index_60_40'][i])]
            count_60_40_v1.append(count_v1)
            count_60_40_v2.append(count_v2)

    return count_50_50, count_55_45_v1, count_55_45_v2, count_60_40_v1, count_60_40_v2


if __name__ == '__main__':
    fig, ax1 = plt.subplots()

    # 50-50
    fig.suptitle('CRT distribution on average traveled distance condition 1 (50-50 km/h)')
    path_to_csv_50_50 = r'D:\Thesis_data_all_experiments\Conditions\condition_50_50'
    condition = '50-50'
    Varjo_data = plot_varjo(path_to_csv_50_50, condition)
    kde = sns.kdeplot(Varjo_data[0], ax=ax1, color='r')
    plt.hist(Varjo_data[0], 30, density=True, color='dodgerblue', edgecolor='black', linewidth=1.2)

    line = kde.lines[0]
    x, y = line.get_data()
    x_index = min(range(len(x)),
                  key=lambda i: abs(x[i] - 135))
    x = x[x_index:len(x)]
    y = y[x_index:len(x)]
    maxid = y.argmax()

    # average1 = x[maxid]
    average1 = sum(Varjo_data[0]) / len(Varjo_data[0])

    ax1.scatter(x[maxid], y[maxid], c='yellow', marker='x', s=100, zorder=3)
    ax1.axvline(x[maxid], 0, 1, color='r', label='Kernel density maximum: ' + str(round(x[maxid])))
    ax1.set_xlim([120, 325])
    ax1.set(xlabel='Travelled distance [m]')
    ax1.legend(loc='best')


    # 55 - 45
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    fig.suptitle('CRT distribution on average traveled distance condition 2 (55-45 km/h)')
    fig.text(0.5, 0.05, "Travelled distance [m]", ha="center", va="center")
    path_to_csv_55_45 = r'D:\Thesis_data_all_experiments\Conditions\condition_55_45'
    condition = '55-45'
    Varjo_data = plot_varjo(path_to_csv_55_45, condition)
    axes[0].hist(Varjo_data[1], 30, density=True, color='dodgerblue', edgecolor='black', linewidth=1.2)
    axes[1].hist(Varjo_data[2], 30, density=True, color='dodgerblue', edgecolor='black', linewidth=1.2)
    kde1 = sns.kdeplot(Varjo_data[1], ax=axes[0], color='r')
    kde2 = sns.kdeplot(Varjo_data[2], ax=axes[1], color='r')

    line1 = kde1.lines[0]
    x1, y1 = line1.get_data()
    x1_index = min(range(len(x1)),
                  key=lambda i: abs(x1[i] - 125))
    x1 = x1[x1_index:len(x1)]
    y1 = y1[x1_index:len(x1)]
    maxid1 = y1.argmax()

    line2 = kde2.lines[0]
    x2, y2 = line2.get_data()
    x2_index = min(range(len(x2)),
                  key=lambda i: abs(x2[i] - 125))
    x2 = x2[x2_index:len(x2)]
    y2 = y2[x2_index:len(x2)]
    maxid2 = y2.argmax()

    # average2 = x1[maxid1]
    # average3 = x2[maxid2]
    average2 = sum(Varjo_data[1]) / len(Varjo_data[1])
    average3 = sum(Varjo_data[2]) / len(Varjo_data[2])

    axes[0].scatter(x1[maxid1], y1[maxid1], c='yellow', marker='x', s=100, zorder=3)
    axes[0].axvline(x1[maxid1], 0, 1, color='r', label='Kernel density maximum: ' + str(round(x1[maxid1])))
    axes[1].scatter(x2[maxid2], y2[maxid2], c='yellow', marker='x', s=100, zorder=3)
    axes[1].axvline(x2[maxid2], 0, 1, color='r', label='Kernel density maximum: ' + str(round(x2[maxid2])))
    axes[0].set_xlim([120, 325])
    axes[1].set_xlim([120, 325])
    axes[0].set_title('Participant is behind')
    axes[1].set_title('Participant is ahead')
    axes[0].legend(loc='best')
    axes[1].legend(loc='best')

    # # 60-40
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    fig.suptitle('CRT distribution on average traveled distance condition 3 (60-40 km/h)')
    fig.text(0.5, 0.05, "Travelled distance [m]", ha="center", va="center")
    path_to_csv_60_40 = r'D:\Thesis_data_all_experiments\Conditions\condition_60_40'
    condition = '60-40'
    Varjo_data = plot_varjo(path_to_csv_60_40, condition)

    axes[0].hist(Varjo_data[3], 30, density=True, color='dodgerblue', edgecolor='black', linewidth=1.2)
    axes[1].hist(Varjo_data[4], 30, density=True, color='dodgerblue', edgecolor='black', linewidth=1.2)
    kde1 = sns.kdeplot(Varjo_data[3], ax=axes[0], color='r')
    kde2 = sns.kdeplot(Varjo_data[4], ax=axes[1], color='r')

    line1 = kde1.lines[0]
    x1, y1 = line1.get_data()
    x1_index = min(range(len(x1)),
                  key=lambda i: abs(x1[i] - 125))
    x1 = x1[x1_index:len(x1)]
    y1 = y1[x1_index:len(x1)]
    maxid1 = y1.argmax()

    line2 = kde2.lines[0]
    x2, y2 = line2.get_data()
    x2_index = min(range(len(x2)),
                  key=lambda i: abs(x2[i] - 125))
    x2 = x2[x2_index:len(x2)]
    y2 = y2[x2_index:len(x2)]
    maxid2 = y2.argmax()

    # average4 = x1[maxid1]
    # average5 = x2[maxid2]
    average4 = sum(Varjo_data[3]) / len(Varjo_data[3])
    average5 = sum(Varjo_data[4]) / len(Varjo_data[4])

    axes[0].scatter(x1[maxid1], y1[maxid1], c='yellow', marker='x', s=100, zorder=3)
    axes[0].axvline(x1[maxid1], 0, 1, color='r', label='Kernel density maximum: ' + str(round(x1[maxid1])))
    axes[1].scatter(x2[maxid2], y2[maxid2], c='yellow', marker='x', s=100, zorder=3)
    axes[1].axvline(x2[maxid2], 0, 1, color='r', label='Kernel density maximum: ' + str(round(x2[maxid2])))
    axes[0].set_xlim([120, 325])
    axes[1].set_xlim([120, 325])
    axes[0].set_title('Participant is behind')
    axes[1].set_title('Participant is ahead')
    axes[0].legend(loc='best')
    axes[1].legend(loc='best')


    #save to csv
    path_to_saved_dict = os.path.join('..', 'data_folder', 'medians_crt_index.csv')
    df1 = pd.DataFrame({'median_50_50': [average1]})
    df2 = pd.DataFrame({'median_55_45_v1': [average2]})
    df3 = pd.DataFrame({'median_55_45_v2': [average3]})
    df4 = pd.DataFrame({'median_60_40_v1': [average4]})
    df5 = pd.DataFrame({'median_60_40_v2': [average5]})
    pd.concat([df1, df2, df3, df4, df5], axis=1).to_csv(path_to_saved_dict, index=False)

    plt.show()
