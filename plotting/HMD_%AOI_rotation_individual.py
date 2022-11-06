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



def average(l):
    llen = len(l)

    def divide(x): return x / llen

    return map(divide, map(sum, zip(*l)))


def plot_varjo(path_to_csv_folder):
    simulation_constants = SimulationConstants(vehicle_width=2,
                                               vehicle_length=4.7,
                                               tunnel_length=118,  # original = 118 -> check in unreal
                                               track_width=8,
                                               track_height=215,
                                               track_start_point_distance=430,
                                               track_section_length_before=304.056,
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

    xy_coordinates_for_vehicle1 = []
    xy_coordinates_for_vehicle2 = []

    for i in range(len(all_pds_list)):
        xy_coordinates_vehicle = vehicle_xy_coordinates(2, all_pds_list[i])
        xy_coordinates_vehicle2 = vehicle_xy_coordinates(5, all_pds_list[i])

        xy_coordinates_zip = [list(a) for a in zip(xy_coordinates_vehicle[0], xy_coordinates_vehicle[1])]
        xy_coordinates_zip2 = [list(a) for a in zip(xy_coordinates_vehicle2[0], xy_coordinates_vehicle2[1])]

        xy_coordinates_for_vehicle1.append(xy_coordinates_zip)
        xy_coordinates_for_vehicle2.append(xy_coordinates_zip2)


    #compute travelled distance for indexes tunnel and merge
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


    #compute the indexes
    indexes_of_tunnel_and_merge_vehicle1 = []
    indexes_of_tunnel_and_merge_vehicle2 = []

    for list_index in range(len(travelled_distance_vehicle1)):
        individual_travelled_distance_vehicle1 = travelled_distance_vehicle1[list_index]
        individual_travelled_distance_vehicle2 = travelled_distance_vehicle2[list_index]
        inner_tunnel_merge_1 = []
        inner_tunnel_merge_2 = []
        for i in range(0, 1):
            index_of_tunnel_vehicle_1 = min(range(len(individual_travelled_distance_vehicle1)), key=lambda i: abs(
                individual_travelled_distance_vehicle1[i] - track.tunnel_length))
            index_of_mergepoint_vehicle_1 = min(range(len(individual_travelled_distance_vehicle1)), key=lambda i: abs(
                individual_travelled_distance_vehicle1[i] - simulation_constants.track_section_length_before))

            index_of_tunnel_vehicle_2 = min(range(len(individual_travelled_distance_vehicle2)), key=lambda i: abs(
                individual_travelled_distance_vehicle2[i] - track.tunnel_length))
            index_of_mergepoint_vehicle_2 = min(range(len(individual_travelled_distance_vehicle2)), key=lambda i: abs(
                individual_travelled_distance_vehicle2[i] - simulation_constants.track_section_length_before))

            inner_tunnel_merge_1.append(index_of_tunnel_vehicle_1)
            inner_tunnel_merge_1.append(index_of_mergepoint_vehicle_1)

            inner_tunnel_merge_2.append(index_of_tunnel_vehicle_2)
            inner_tunnel_merge_2.append(index_of_mergepoint_vehicle_2)

        indexes_of_tunnel_and_merge_vehicle1.append(inner_tunnel_merge_1)
        indexes_of_tunnel_and_merge_vehicle2.append(inner_tunnel_merge_2)


    #interactive data for each vehicle
    interactive_area_travelled_trace_vehicle1 = []
    hmd_rot_interactive_area_vehicle1 = []

    interactive_area_travelled_trace_vehicle2 = []
    hmd_rot_interactive_area_vehicle2 = []


    for i in range(len(indexes_of_tunnel_and_merge_vehicle1)):
        hmd_rot_1 = list(all_pds_list[i]['HMD_rotation_vehicle1'][
                       indexes_of_tunnel_and_merge_vehicle1[i][0]:indexes_of_tunnel_and_merge_vehicle1[i][1]])

        hmd_rot_2 = list(all_pds_list[i]['HMD_rotation_vehicle2'][
                         indexes_of_tunnel_and_merge_vehicle2[i][0]:indexes_of_tunnel_and_merge_vehicle2[i][1]])

        interactive_trace_1 = travelled_distance_vehicle1[i][indexes_of_tunnel_and_merge_vehicle1[i][0]:indexes_of_tunnel_and_merge_vehicle1[i][1]]
        interactive_trace_2 = travelled_distance_vehicle1[i][indexes_of_tunnel_and_merge_vehicle2[i][0]:indexes_of_tunnel_and_merge_vehicle2[i][1]]

        hmd_rot_interactive_area_vehicle1.append(hmd_rot_1)
        hmd_rot_interactive_area_vehicle2.append(hmd_rot_2)

        interactive_area_travelled_trace_vehicle1.append(interactive_trace_1)
        interactive_area_travelled_trace_vehicle2.append(interactive_trace_2)

        on_ramp_vs_opponent_vehicle1 = []
        on_ramp_vs_opponent_vehicle2 = []

        for list_index in range(len(hmd_rot_interactive_area_vehicle1)):
            individual_hmd_rot_list_1 = hmd_rot_interactive_area_vehicle1[list_index]
            individual_hmd_rot_list_2 = hmd_rot_interactive_area_vehicle2[list_index]

            inner_attention_list_1 = []
            inner_attention_list_2 = []

            for i in range(len(individual_hmd_rot_list_1)):
                if individual_hmd_rot_list_1[i] > 0.96:  # this we need to know better
                    inner_attention_list_1.append(1)
                else:
                    inner_attention_list_1.append(0)

            for i in range(len(individual_hmd_rot_list_2)):
                if individual_hmd_rot_list_2[i] > 0.96:  # this we need to know better
                    inner_attention_list_2.append(1)
                else:
                    inner_attention_list_2.append(0)

            on_ramp_vs_opponent_vehicle1.append(inner_attention_list_1)
            on_ramp_vs_opponent_vehicle2.append(inner_attention_list_2)

    average_hmd_vehicle1 = list(average(on_ramp_vs_opponent_vehicle1))
    average_hmd_vehicle2 = list(average(on_ramp_vs_opponent_vehicle2))

    average_trace_vehicle1 = sum(average_hmd_vehicle1) / len(average_hmd_vehicle1)
    average_trace_vehicle2 = sum(average_hmd_vehicle2) / len(average_hmd_vehicle2)

    ysmoothed_1 = gaussian_filter1d(average_hmd_vehicle1[0:1150], sigma=4)
    ysmoothed_2 = list(gaussian_filter1d(average_hmd_vehicle2[0:1150], sigma=4))
    x_vehicle1 = list(np.linspace(120, 275, len(average_hmd_vehicle1[0:1150])))
    x_vehicle2 = list(np.linspace(120, 275, len(average_hmd_vehicle2[0:1150])))

    return ysmoothed_1, ysmoothed_2, x_vehicle1, x_vehicle2, average_trace_vehicle1, average_trace_vehicle2

if __name__ == '__main__':

    #left ahead 45-55
    path_to_csv_45_55 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\left'
    Varjo_data = plot_varjo(path_to_csv_45_55)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Comparison gaze behavior condition 55_45')
    ax1.plot(Varjo_data[2], Varjo_data[0])  # see x below
    ax1.fill_between(Varjo_data[2], Varjo_data[0], color='blue', alpha=0.1, label='Fixation on road')
    ax1.fill_between(Varjo_data[2], Varjo_data[0], 1, color='red', alpha=0.1, label='Fixation on opponent')
    ax1.set_title('Right vehicle negative headway')
    ax1.set_xlim([120, 275])
    ax1.set_ylim([0, 1])

    ax2.plot(Varjo_data[3], Varjo_data[1])  # see x below
    ax2.fill_between(Varjo_data[3], Varjo_data[1], color='blue', alpha=0.1, label='Fixation on road')
    ax2.fill_between(Varjo_data[3], Varjo_data[1], 1, color='red', alpha=0.1, label='Fixation on opponent')
    ax2.set_title('Left vehicle positive headway')
    ax2.set_xlim([120, 275])
    ax2.set_ylim([0, 1])

    ax1.set(ylabel='% fixated on AOI')
    fig.text(0.5, 0.04, "Average travelled distance", ha="center", va="center")
    ax1.plot([], [], ' ', label='Average: ' + str(round(Varjo_data[4], 2)))
    ax2.plot([], [], ' ', label='Average: ' + str(round(Varjo_data[5], 2)))
    ax1.legend(loc='lower left')
    ax2.legend(loc='lower left')


    #right ahead 55-45
    path_to_csv_55_45 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_55_45\right'
    Varjo_data = plot_varjo(path_to_csv_55_45)

    fig, (ax3, ax4) = plt.subplots(1, 2)
    fig.suptitle('Comparison gaze behavior condition 55_45')
    ax3.plot(Varjo_data[2], Varjo_data[0])  # see x below
    ax3.fill_between(Varjo_data[2], Varjo_data[0], color='blue', alpha=0.1, label='Fixation on road')
    ax3.fill_between(Varjo_data[2], Varjo_data[0], 1, color='red', alpha=0.1, label='Fixation on opponent')
    ax3.set_title('Right vehicle positive headway')
    ax3.set_xlim([120, 275])
    ax3.set_ylim([0, 1])

    ax4.plot(Varjo_data[3], Varjo_data[1])  # see x below
    ax4.fill_between(Varjo_data[3], Varjo_data[1], color='blue', alpha=0.1, label='Fixation on road')
    ax4.fill_between(Varjo_data[3], Varjo_data[1], 1, color='red', alpha=0.1, label='Fixation on opponent')
    ax4.set_title('Left vehicle negative headway')
    ax4.set_xlim([120, 275])
    ax4.set_ylim([0, 1])

    ax3.set(ylabel='% fixated on AOI')
    fig.text(0.5, 0.04, "Average travelled distance", ha="center", va="center")
    ax3.plot([], [], ' ', label='Average: ' + str(round(Varjo_data[4], 2)))
    ax4.plot([], [], ' ', label='Average: ' + str(round(Varjo_data[5], 2)))
    ax3.legend(loc='lower left')
    ax4.legend(loc='lower left')


    #left ahead 60-40
    path_to_csv_40_60 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\left'
    Varjo_data = plot_varjo(path_to_csv_40_60)

    fig, (ax5, ax6) = plt.subplots(1, 2)
    fig.suptitle('Comparison gaze behavior condition 60_40')
    ax5.plot(Varjo_data[2], Varjo_data[0])  # see x below
    ax5.fill_between(Varjo_data[2], Varjo_data[0], color='blue', alpha=0.1, label='Fixation on road')
    ax5.fill_between(Varjo_data[2], Varjo_data[0], 1, color='red', alpha=0.1, label='Fixation on opponent')
    ax5.set_title('Right vehicle negative headway')
    ax5.set_xlim([120, 275])
    ax5.set_ylim([0, 1])

    ax6.plot(Varjo_data[3], Varjo_data[1])  # see x below
    ax6.fill_between(Varjo_data[3], Varjo_data[1], color='blue', alpha=0.1, label='Fixation on road')
    ax6.fill_between(Varjo_data[3], Varjo_data[1], 1, color='red', alpha=0.1, label='Fixation on opponent')
    ax6.set_title('Left vehicle positive headway')
    ax6.set_xlim([120, 275])
    ax6.set_ylim([0, 1])

    ax5.set(ylabel='% fixated on AOI')
    fig.text(0.5, 0.04, "Average travelled distance", ha="center", va="center")
    ax5.plot([], [], ' ', label='Average: ' + str(round(Varjo_data[4], 2)))
    ax6.plot([], [], ' ', label='Average: ' + str(round(Varjo_data[5], 2)))
    ax5.legend(loc='lower left')
    ax6.legend(loc='lower left')


    # right ahead 60-40
    path_to_csv_60_40 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_60_40\right'
    Varjo_data = plot_varjo(path_to_csv_60_40)

    fig, (ax7, ax8) = plt.subplots(1, 2)
    fig.suptitle('Comparison gaze behavior condition 60_40')
    ax7.plot(Varjo_data[2], Varjo_data[0])  # see x below
    ax7.fill_between(Varjo_data[2], Varjo_data[0], color='blue', alpha=0.1, label='Fixation on road')
    ax7.fill_between(Varjo_data[2], Varjo_data[0], 1, color='red', alpha=0.1, label='Fixation on opponent')
    ax7.set_title('Right vehicle positive headway')
    ax7.set_xlim([120, 275])
    ax7.set_ylim([0, 1])

    ax8.plot(Varjo_data[3], Varjo_data[1])  # see x below
    ax8.fill_between(Varjo_data[3], Varjo_data[1], color='blue', alpha=0.1, label='Fixation on road')
    ax8.fill_between(Varjo_data[3], Varjo_data[1], 1, color='red', alpha=0.1, label='Fixation on opponent')
    ax8.set_title('Left vehicle negative headway')
    ax8.set_xlim([120, 275])
    ax8.set_ylim([0, 1])

    ax7.set(ylabel='% fixated on AOI')
    fig.text(0.5, 0.04, "Average travelled distance", ha="center", va="center")
    ax7.plot([], [], ' ', label='Average: ' + str(round(Varjo_data[4], 2)))
    ax8.plot([], [], ' ', label='Average: ' + str(round(Varjo_data[5], 2)))
    ax7.legend(loc='lower left')
    ax8.legend(loc='lower left')

    # 50-50
    path_to_csv_50_50 = r'C:\Users\loran\Desktop\Mechanical engineering - Delft\Thesis\Thesis_data_all_experiments\Conditions\Conditions_who_is_ahead\whos_ahead_50_50'
    Varjo_data = plot_varjo(path_to_csv_50_50)

    fig, (ax9, ax10) = plt.subplots(1, 2)
    fig.suptitle('Comparison gaze behavior condition 50_50')
    ax9.plot(Varjo_data[2], Varjo_data[0])  # see x below
    ax9.fill_between(Varjo_data[2], Varjo_data[0], color='blue', alpha=0.1, label='Fixation on road')
    ax9.fill_between(Varjo_data[2], Varjo_data[0], 1, color='red', alpha=0.1, label='Fixation on opponent')
    ax9.set_title('Right vehicle equal headway')
    ax9.set_xlim([120, 275])
    ax9.set_ylim([0, 1])

    ax10.plot(Varjo_data[3], Varjo_data[1])  # see x below
    ax10.fill_between(Varjo_data[3], Varjo_data[1], color='blue', alpha=0.1, label='Fixation on road')
    ax10.fill_between(Varjo_data[3], Varjo_data[1], 1, color='red', alpha=0.1, label='Fixation on opponent')
    ax10.set_title('Left vehicle equal headway')
    ax10.set_xlim([120, 275])
    ax10.set_ylim([0, 1])

    ax9.set(ylabel='% fixated on AOI')
    fig.text(0.5, 0.04, "Average travelled distance", ha="center", va="center")
    ax9.plot([], [], ' ', label='Average: ' + str(round(Varjo_data[4], 2)))
    ax10.plot([], [], ' ', label='Average: ' + str(round(Varjo_data[5], 2)))
    ax9.legend(loc='lower left')
    ax10.legend(loc='lower left')

    plt.show()