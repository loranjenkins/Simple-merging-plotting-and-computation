import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage import gaussian_filter1d
import numpy as np
import seaborn as sns
from scipy import interpolate
import os
import numpy as np
from natsort import natsorted
import scipy.stats as stats

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


def plot_varjo(path_to_csv_folder, condition, who_ahead):
    # simulation_constants = SimulationConstants(vehicle_width=2,
    #                                            vehicle_length=4.7,
    #                                            tunnel_length=100,  # original = 118 -> check in unreal
    #                                            track_width=8,
    #                                            track_height=215,
    #                                            track_start_point_distance=430,
    #                                            track_section_length_before=304.056,
    #                                            track_section_length_after=150)  # goes until 400

    simulation_constants = SimulationConstants(vehicle_width=1.5,
                                               vehicle_length=4.7,
                                               tunnel_length=130,
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

    if condition == '50-50':
        dict = {'CRT': [], '% fixation on opponent': []}
    if condition == '55-45':
        dict = {'CRT': [], '% fixation vehicle 1': [], '% fixation vehicle 2': []}
    if condition == '60-40':
        dict = {'CRT': [], '% fixation vehicle 1': [], '% fixation vehicle 2': []}

    path_to_data_csv = os.path.join('..', 'data_folder', 'crt_who_is_ahead.csv')
    global_crt_median = pd.read_csv(path_to_data_csv, sep=',')

    if condition == '50-50':
        if who_ahead == 'equal':
            for i in global_crt_median['crt_50_50'].dropna():
                dict['CRT'].append(i)
    if condition == '55-45':
        if who_ahead == 'vehicle1':
            for i in global_crt_median['crt_45_55_vehicle1'].dropna():
                dict['CRT'].append(i)
        elif who_ahead == 'vehicle2':
            for i in global_crt_median['crt_55_45_vehicle2'].dropna():
                dict['CRT'].append(i)
    if condition == '60-40':
        if who_ahead == 'vehicle1':
            for i in global_crt_median['crt_40_60_vehicle1'].dropna():
                dict['CRT'].append(i)
        elif who_ahead == 'vehicle2':
            for i in global_crt_median['crt_60_40_vehicle2'].dropna():
                dict['CRT'].append(i)

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
                                                   individual_trace_vehicle1[i] - track.section_length_before))
            index_of_mergepoint_vehicle2 = min(range(len(individual_trace_vehicle2)),
                                               key=lambda i: abs(
                                                   individual_trace_vehicle2[i] - track.section_length_before))

            inner_index_list_1.append(index_of_tunnel_vehicle_1)
            inner_index_list_1.append(index_of_mergepoint_vehicle1)

            inner_index_list_2.append(index_of_tunnel_vehicle_2)
            inner_index_list_2.append(index_of_mergepoint_vehicle2)

        indexes_of_tunnel_and_merge_vehicle1.append(inner_index_list_1)
        indexes_of_tunnel_and_merge_vehicle2.append(inner_index_list_2)

    # interactive data for each vehicle
    hmd_rot_interactive_area_vehicle1 = []
    hmd_rot_interactive_area_vehicle2 = []

    for i in range(len(indexes_of_tunnel_and_merge_vehicle1)):
        hmd_rot_1 = list(all_pds_list[i]['HMD_rotation_vehicle1'][
                         indexes_of_tunnel_and_merge_vehicle1[i][0]:indexes_of_tunnel_and_merge_vehicle1[i][1]])

        hmd_rot_2 = list(all_pds_list[i]['HMD_rotation_vehicle2'][
                         indexes_of_tunnel_and_merge_vehicle2[i][0]:indexes_of_tunnel_and_merge_vehicle2[i][1]])

        hmd_rot_interactive_area_vehicle1.append(hmd_rot_1)
        hmd_rot_interactive_area_vehicle2.append(hmd_rot_2)

        on_ramp_vs_opponent_vehicle1 = []
        on_ramp_vs_opponent_vehicle2 = []

    for list_index in range(len(hmd_rot_interactive_area_vehicle1)):
        individual_hmd_rot_list_1 = hmd_rot_interactive_area_vehicle1[list_index]
        individual_hmd_rot_list_2 = hmd_rot_interactive_area_vehicle2[list_index]

        inner_attention_list_1 = []
        inner_attention_list_2 = []

        for i in range(len(individual_hmd_rot_list_1)):
            if individual_hmd_rot_list_1[i] > 0.95:  # this we need to know better
                inner_attention_list_1.append(1)
            else:
                inner_attention_list_1.append(0)

        for i in range(len(individual_hmd_rot_list_2)):
            if individual_hmd_rot_list_2[i] > 0.95:  # this we need to know better
                inner_attention_list_2.append(1)
            else:
                inner_attention_list_2.append(0)

        on_ramp_vs_opponent_vehicle1.append(inner_attention_list_1)
        on_ramp_vs_opponent_vehicle2.append(inner_attention_list_2)

    if condition == '50-50':
        for i in range(len(on_ramp_vs_opponent_vehicle1)):
            average_v1 = 1 - sum(on_ramp_vs_opponent_vehicle1[i]) / len(on_ramp_vs_opponent_vehicle1[i])
            average_v2 = 1 - sum(on_ramp_vs_opponent_vehicle2[i]) / len(on_ramp_vs_opponent_vehicle2[i])
            average_together = np.array([average_v1, average_v2])
            average_together = np.mean(average_together, axis=0)
            dict['% fixation on opponent'].append(average_together)

    if condition == '55-45':
        for i in range(len(on_ramp_vs_opponent_vehicle1)):
            average_v1 = 1 - sum(on_ramp_vs_opponent_vehicle1[i]) / len(on_ramp_vs_opponent_vehicle1[i])
            average_v2 = 1 - sum(on_ramp_vs_opponent_vehicle2[i]) / len(on_ramp_vs_opponent_vehicle2[i])
            dict['% fixation vehicle 1'].append(average_v1)
            dict['% fixation vehicle 2'].append(average_v2)

    if condition == '60-40':
        for i in range(len(on_ramp_vs_opponent_vehicle1)):
            average_v1 = 1 - sum(on_ramp_vs_opponent_vehicle1[i]) / len(on_ramp_vs_opponent_vehicle1[i])
            average_v2 = 1 - sum(on_ramp_vs_opponent_vehicle2[i]) / len(on_ramp_vs_opponent_vehicle2[i])
            dict['% fixation vehicle 1'].append(average_v1)
            dict['% fixation vehicle 2'].append(average_v2)

    return dict


if __name__ == '__main__':
    # 55_45
    path_to_csv_vehicle1_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\vehicle 1'
    dict55_45_v1_ahead = plot_varjo(path_to_csv_vehicle1_ahead, '55-45', 'vehicle1')

    path_to_csv_vehicle2_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\vehicle 2'
    dict55_45_v2_ahead = plot_varjo(path_to_csv_vehicle2_ahead, '55-45', 'vehicle2')

    df1 = pd.DataFrame.from_dict(dict55_45_v1_ahead)
    df2 = pd.DataFrame.from_dict(dict55_45_v2_ahead)

    ahead_fixations_55_45 = pd.concat([df1['% fixation vehicle 1'], df2['% fixation vehicle 2']], axis=0).rename(
        '% fixation on opponent')
    ahead_crt_55_45 = pd.concat([df1['CRT'], df2['CRT']], axis=0)
    df_ahead_55_45 = pd.concat([ahead_crt_55_45, ahead_fixations_55_45], axis=1)

    behind_fixations_55_45 = pd.concat([df1['% fixation vehicle 2'], df2['% fixation vehicle 1']], axis=0).rename(
        '% fixation on opponent')
    behind_crt_55_45 = pd.concat([df1['CRT'], df2['CRT']], axis=0)
    df_behind_55_45 = pd.concat([behind_crt_55_45, behind_fixations_55_45], axis=1)

    r, p = stats.pearsonr(df_ahead_55_45['CRT'], df_ahead_55_45['% fixation on opponent'])
    r1, p1 = stats.pearsonr(df_behind_55_45['CRT'], df_behind_55_45['% fixation on opponent'])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Linear regression analysis on condition 2 (55-45 km/h)')
    fig.text(0.05, 0.5, "Fixation on opponent [%]", va='center', rotation='vertical')
    fig.text(0.5, 0.05, "CRT [s]", ha="center", va="center")

    sns.regplot(df_ahead_55_45, x='CRT', y='% fixation on opponent', ax=axes[0])
    sns.regplot(df_behind_55_45, x='CRT', y='% fixation on opponent', ax=axes[1])

    axes[0].set_title('Participant is behind')
    axes[1].set_title('Participant is ahead')

    axes[0].set(xlabel=None, ylabel= None)
    axes[1].set(xlabel=None, ylabel= None)

    axes[0].plot([], [], ' ', label='r: ' + str(round(r, 2)))
    axes[0].plot([], [], ' ', label='p: ' + str(round(p, 2)))
    axes[1].plot([], [], ' ', label='r: ' + str(round(r1, 2)))
    axes[1].plot([], [], ' ', label='p: ' + "{:.2e}".format(p1))

    axes[0].legend(loc='best')
    axes[1].legend(loc='best')

    # --------------------------------------------------
    # 60-40
    path_to_csv_vehicle1_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\vehicle1'
    dict60_40_v1_ahead = plot_varjo(path_to_csv_vehicle1_ahead, '60-40', 'vehicle1')

    path_to_csv_vehicle2_ahead = r'D:\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\vehicle2'
    dict60_40_v2_ahead = plot_varjo(path_to_csv_vehicle2_ahead, '60-40', 'vehicle2')

    df3 = pd.DataFrame.from_dict(dict60_40_v1_ahead)
    df4 = pd.DataFrame.from_dict(dict60_40_v2_ahead)

    ahead_fixations_60_40 = pd.concat([df3['% fixation vehicle 1'], df4['% fixation vehicle 2']], axis=0).rename(
        '% fixation on opponent')
    ahead_crt_60_40 = pd.concat([df3['CRT'], df4['CRT']], axis=0)
    df_ahead_60_40 = pd.concat([ahead_crt_60_40, ahead_fixations_60_40], axis=1)

    behind_fixations_60_40 = pd.concat([df3['% fixation vehicle 2'], df4['% fixation vehicle 1']], axis=0).rename(
        '% fixation on opponent')
    behind_crt_60_40 = pd.concat([df3['CRT'], df4['CRT']], axis=0)
    df_behind_60_40 = pd.concat([behind_crt_60_40, behind_fixations_60_40], axis=1)

    r3, p3 = stats.pearsonr(df_ahead_60_40['CRT'], df_ahead_60_40['% fixation on opponent'])
    r4, p4 = stats.pearsonr(df_behind_60_40['CRT'], df_behind_60_40['% fixation on opponent'])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Linear regression analysis on condition 3 (60-40 km/h)')
    fig.text(0.05, 0.5, "Fixation on opponent [%]", va='center', rotation='vertical')
    fig.text(0.5, 0.05, "CRT [s]", ha="center", va="center")

    sns.regplot(df_ahead_60_40, x='CRT', y='% fixation on opponent', ax=axes[0])
    sns.regplot(df_behind_60_40, x='CRT', y='% fixation on opponent', ax=axes[1])

    axes[0].set_title('Participant is behind')
    axes[1].set_title('Participant is ahead')

    axes[0].set(xlabel=None, ylabel= None)
    axes[1].set(xlabel=None, ylabel= None)

    axes[0].plot([], [], ' ', label='r: ' + str(round(r3, 2)))
    axes[0].plot([], [], ' ', label='p: ' + str(round(p3, 2)))
    axes[1].plot([], [], ' ', label='r: ' + str(round(r4, 2)))
    axes[1].plot([], [], ' ', label='p: ' + str(round(p4, 2)))

    axes[0].legend(loc='best')
    axes[1].legend(loc='best')


    # --------------------------------------------------
    # #50_50
    path_to_csv_50_50 = r'D:\Thesis_data_all_experiments\Conditions\condition_50_50'
    dict50_50 = plot_varjo(path_to_csv_50_50, '50-50', 'equal')

    df_50_50 = pd.DataFrame.from_dict(dict50_50)

    r5, p5 = stats.pearsonr(df_50_50['CRT'], df_50_50['% fixation on opponent'])
    fig, ax5 = plt.subplots(1, 1)
    fig.suptitle('Linear regression analysis on condition 1 (50-50 km/h)')

    sns.regplot(df_50_50, x='CRT', y='% fixation on opponent')
    ax5.set(xlabel='CRT [s]', ylabel='Fixation on opponent [%]')
    ax5.plot([], [], ' ', label='r: ' + str(round(r5, 2)))
    ax5.plot([], [], ' ', label='p: ' + "{:.2e}".format(p5))
    ax5.legend(loc='best')


    #----- combined plot
    # df_50_50['condition'] = ['condition 1'] * len(df_50_50)
    # df_ahead_55_45['condition'] = ['condition 2'] * len(df_ahead_55_45)
    # df_behind_55_45['condition'] = ['condition 2'] * len(df_behind_55_45)
    # df_ahead_60_40['condition'] = ['condition 3'] * len(df_ahead_60_40)
    # df_behind_60_40['condition'] = ['condition 3'] * len(df_behind_60_40)

    df_total = pd.concat([df_ahead_55_45, df_behind_55_45, df_ahead_60_40, df_behind_60_40, df_50_50])

    r6, p6 = stats.pearsonr(df_total['CRT'], df_total['% fixation on opponent'])

    fig, ax6 = plt.subplots(1, 1)
    fig.suptitle('Linear regression analysis of all conditions')
    sns.regplot(df_total, x='CRT', y='% fixation on opponent')
    ax6.set(xlabel='CRT [s]', ylabel='Fixation on opponent [%]')
    ax6.plot([], [], ' ', label='r: ' + str(round(r6, 2)))
    ax6.plot([], [], ' ', label='p: ' + str(round(p6, 2)))
    ax6.legend(loc='best')

    plt.show()


    # --------to csv
    path_to_saved_dict_crt = os.path.join('..', 'data_folder', 'boxplot_who_is_ahead.csv')
    df2 = pd.DataFrame({'ahead_fixations_55_45': list(ahead_fixations_55_45)})
    df3 = pd.DataFrame({'behind_fixations_55_45': list(behind_fixations_55_45)})
    df4 = pd.DataFrame({'ahead_fixations_60_40': list(ahead_fixations_60_40)})
    df5 = pd.DataFrame({'behind_fixations_60_40': list(behind_fixations_60_40)})
    pd.concat([df2, df3, df4, df5], axis=1).to_csv(path_to_saved_dict_crt, index=False)