import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage import gaussian_filter1d

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

def plot_varjo(path_to_csv_folder):
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
        for i in range(0,1):
            index_of_tunnel_vehicle = min(range(len(individual_travelled_distance_vehicle)), key=lambda i: abs(individual_travelled_distance_vehicle[i] - track.tunnel_length))
            index_of_mergepoint_vehicle = min(range(len(individual_travelled_distance_vehicle)), key=lambda i: abs(individual_travelled_distance_vehicle[i] - simulation_constants.track_section_length_before))
            inner_tunnel_merge.append(index_of_tunnel_vehicle)
            inner_tunnel_merge.append(index_of_mergepoint_vehicle)
        indexes_of_tunnel_and_merge.append(inner_tunnel_merge)

    interactive_area_travelled_traces = []
    hmd_rot_interactive_area = []

    for i in range(len(indexes_of_tunnel_and_merge)):
        hmd_rot = list(all_pds_list[i]['HMD_rotation_vehicle1'][indexes_of_tunnel_and_merge[i][0]:indexes_of_tunnel_and_merge[i][1]])
        interactive_trace = travelled_distance[i][indexes_of_tunnel_and_merge[i][0]:indexes_of_tunnel_and_merge[i][1]]
        hmd_rot_interactive_area.append(hmd_rot)
        interactive_area_travelled_traces.append(interactive_trace)

    on_ramp_vs_opponent = []
    for list_index in range(len(hmd_rot_interactive_area)):
        individual_hmd_rot_list = hmd_rot_interactive_area[list_index]
        inner_attention_list = []
        for i in range(len(individual_hmd_rot_list)):
            if individual_hmd_rot_list[i] > 0.99:
                inner_attention_list.append(1)
            else:
                inner_attention_list.append(0)

        on_ramp_vs_opponent.append(inner_attention_list)


    df_traces = pd.DataFrame(interactive_area_travelled_traces)
    x_mean_traces = df_traces.mean()
    df_hmd_rotations = pd.DataFrame(on_ramp_vs_opponent)
    y_mean_traces = df_hmd_rotations.mean()

    fig, (ax1) = plt.subplots(1)
    #
    # ax1.title('Area of interest over travelled distance')
    ax1.set_xlabel('average travelled distance [m]')
    ax1.set_ylabel('% fixated on AOI')
    # ax1.plot(x_mean_traces[0:500], y_mean_traces[0:500])
    # print(len(x_mean_traces))
    # print(len(y_mean_traces))

    #find longest column and put in gaussian
    print(df_traces)

    ysmoothed = gaussian_filter1d(y_mean_traces, sigma=10)
    ax1.plot(df_traces.iloc[5], ysmoothed)

    # spl = make_interp_spline(x_mean_traces[0:750], y_mean_traces[0:750], k=3) #type: BSpline
    # xnew = np.linspace(x_mean_traces[0:750].min(), x_mean_traces[0:750].max(), 100)
    # power_smooth = spl(xnew)
    # plt.plot(xnew, power_smooth)

    plt.show()



if __name__ == '__main__':

    plot_varjo(r'C:\Users\loran\Desktop\ExperimentOlgerArkady\Joan.Varjo.combined')


