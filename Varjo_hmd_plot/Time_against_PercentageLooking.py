import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

if __name__ == '__main__':
    pd.options.mode.chained_assignment = None  # default='warn'
    # data = pd.read_csv(
    #     r'D:\Pycharmprojects\Simple-merging-plotting-and-computation\data_folder\joan_data_experment_2vehicles.csv',
    #     sep=';', converters={'Carla Interface.agents.Ego Vehicle_1.transform': literal_eval})
    data = pd.read_csv(
        r'D:\Pycharmprojects\ObjectDetection\Plotting\Varjo_experiment_data.csv_2022-08-11 143010 - olger right - arkady left.csv',
        sep=',')
    data.values.tolist()

    #find zeros and exclude
    gaze = []
    for i in range(len(data['gaze_forward'])):
        eval_array = eval(data['gaze_forward'][i])
        gaze.append(eval_array)
    list_zeros = []
    for i in range(len(gaze)):
        zeros = gaze[i][0]
        list_zeros.append(zeros)

    data['indexes_to_exclude'] = list_zeros

    Filtered_gaze_data = data[data.indexes_to_exclude != 0]
    Filtered_gaze_data = Filtered_gaze_data.drop(columns='indexes_to_exclude')


    #find time spend looking in front
    def FindPointInsideRectangle(x1, y1, x2, y2, x, y):
        if x1 < x < x2 and y1 < y < y2:
            return True
        else:
            return False
    #define bounding box
    bounding_box_looking_front = [-0.5, -1, 0.5, 0.1] #collect this data better!
    bounding_box_interest_other_vehicle = [-1, -0.25, 1, 0.25]
    bounding_box_other = [-1, -1, 1, 1]

    #Bounding areas HMD rotation
    data_looking_front = Filtered_gaze_data[Filtered_gaze_data['HMD_rotation'].between(0.80, 1)]
    data_looking_other_vehicle = Filtered_gaze_data[Filtered_gaze_data['HMD_rotation'].between(-0.2, 0.6)]
    data_looking_other = Filtered_gaze_data[Filtered_gaze_data['HMD_rotation'].between(0.6, 0.80)]
    # print(data_looking_front)
    # print(data_looking_interest_other_vehicle)
    # print(data_looking_other)

    #data looking in front
    data_is_inside_looking_front = []
    for i in range(len(data_looking_front)):
        is_inside = FindPointInsideRectangle(bounding_box_looking_front[0],bounding_box_looking_front[1],
                                             bounding_box_looking_front[2],bounding_box_looking_front[3],
                                             eval(data_looking_front['gaze_forward'].array[i])[0],eval(data_looking_front['gaze_forward'].array[i])[1])
        data_is_inside_looking_front.append(is_inside)

    data_looking_front['is_inside'] = data_is_inside_looking_front
    data_looking_front_true = data_looking_front.loc[data_looking_front.is_inside, :]
    # print(data_looking_front_true)
    data_looking_front_false = data_looking_front[~data_looking_front["is_inside"]]
    # print(data_looking_front_false)

    #data looking other vehicle
    data_is_inside_other_vehicle = []
    for i in range(len(data_looking_other_vehicle)):
        is_inside1 = FindPointInsideRectangle(bounding_box_interest_other_vehicle[0],bounding_box_interest_other_vehicle[1],
                                             bounding_box_interest_other_vehicle[2],bounding_box_interest_other_vehicle[3],
                                             eval(data_looking_other_vehicle['gaze_forward'].array[i])[0],
                                             eval(data_looking_other_vehicle['gaze_forward'].array[i])[1])
        data_is_inside_other_vehicle.append(is_inside1)

    data_looking_other_vehicle['is_inside'] = data_is_inside_other_vehicle
    data_looking_other_vehicle_true = data_looking_other_vehicle.loc[data_looking_other_vehicle.is_inside, :]
    # print(data_looking_other_vehicle_true)
    data_looking_other_vehicle_false = data_looking_other_vehicle[~data_looking_other_vehicle["is_inside"]]
    # print(data_looking_other_vehicle_false)

    #other data
    data_is_inside_looking_other = []
    for i in range(len(data_looking_other)):
        is_inside2 = FindPointInsideRectangle(bounding_box_other[0],bounding_box_other[1],
                                             bounding_box_other[2],bounding_box_other[3],
                                             eval(data_looking_other['gaze_forward'].array[i])[0],
                                             eval(data_looking_other['gaze_forward'].array[i])[1])
        data_is_inside_looking_other.append(is_inside2)


    data_looking_other['is_inside'] = data_is_inside_looking_other
    data_is_inside_looking_other_true = data_looking_other.loc[data_looking_other.is_inside, :]
    # print(data_is_inside_looking_other_true)
    data_is_inside_looking_other_false = data_looking_other[~data_looking_other["is_inside"]]
    # print(data_is_inside_looking_other_false)

    # Here append other false values from looking front, looking other vehicle
    merged_front_and_othervehicle = pd.concat([data_looking_other_vehicle_false,data_looking_front_false])
    merged_all_looking_other = pd.concat([merged_front_and_othervehicle,data_is_inside_looking_other_true])

    #Plot %looking vs time
    # print(data_looking_front_true)
    # print(len(data_looking_other_vehicle_true))
    # print(len(merged_all_looking_other))

