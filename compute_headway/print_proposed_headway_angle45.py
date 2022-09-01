from compute_headway.conditiondefinition import ConditionDefinition
import math
import numpy as np

if __name__ == '__main__':
# def _from_difference_at_point(left_velocity, right_velocity, left_headway, headway, distance, name):


    #geometry axis: upwards y = negative upwards x = positive
    initial_start_point_y = 5
    offset_width = 20 * 100
    width = 195
    height = 200-initial_start_point_y
    length_tunnel = 120
    y_merge_point = (height+initial_start_point_y)*100
    angle_from_merge_point = math.atan(width/height) #radians
    angle_from_merge_point_degrees = angle_from_merge_point*(180/math.pi)
    print('Angle in radians:', angle_from_merge_point)
    print('Angle in degrees:', angle_from_merge_point*(180/math.pi))
    section_length = np.sqrt(width ** 2 + height ** 2)

    print('section length:', section_length)


# def _from_difference_at_point(left_velocity, right_velocity, left_headway, distance, name):
    left_velocity = 37 #vehicle1
    right_velocity = 43 #vehicle2

    headway = ConditionDefinition._from_difference_at_point(left_velocity, right_velocity, 0, section_length, 'test')

    # print('right_initial_offset:', headway.right_initial_position_offset)
    # print('left_initial_offset:', headway.left_initial_position_offset)

    #this one if right has speed advantage
    if left_velocity == right_velocity:
        difference_section_length_from_merge_point = section_length - length_tunnel/2
        x_condition_unreal = difference_section_length_from_merge_point * math.sin(angle_from_merge_point)
        y_condition_unreal = difference_section_length_from_merge_point * math.cos(angle_from_merge_point)
        print('velocity_left_vehicle:', left_velocity, 'new_x_left_vehicle:', x_condition_unreal * 100 + offset_width, 'new_y_left_vehicle:', y_merge_point-y_condition_unreal * 100, 'Angle:', 90+angle_from_merge_point_degrees)
        print('velocity_right_vehicle:', right_velocity, 'new_x_right_vehicle:', -x_condition_unreal*100 - offset_width, 'new_y_right_vehicle:', y_merge_point-y_condition_unreal*100, 'Angle:', 90-angle_from_merge_point_degrees)

    elif right_velocity > left_velocity:
        difference_section_length_from_merge_point = section_length - headway.left_initial_position_offset
        x_condition_unreal = difference_section_length_from_merge_point * math.sin(angle_from_merge_point)
        y_condition_unreal = difference_section_length_from_merge_point * math.cos(angle_from_merge_point)
        print('velocity_left_vehicle:', left_velocity, 'new_x_left_vehicle:', x_condition_unreal*100 + offset_width,'new_y_left_vehicle:', y_merge_point-y_condition_unreal*100, 'Angle:', 90+angle_from_merge_point_degrees)
        print('velocity_right_vehicle:', right_velocity, 'new_x_right_vehicle:', -width*100 - offset_width, 'new_y_right_vehicle:', initial_start_point_y*100, 'Angle:', 90-angle_from_merge_point_degrees)

    elif left_velocity > right_velocity:
        difference_section_length_from_merge_point = section_length - headway.right_initial_position_offset
        x_condition_unreal = difference_section_length_from_merge_point * math.sin(angle_from_merge_point)
        y_condition_unreal = difference_section_length_from_merge_point * math.cos(angle_from_merge_point)
        print('velocity_left_vehicle:', left_velocity, 'new_x_left_vehicle:', width*100 + offset_width , 'new_y_left_vehicle:', initial_start_point_y*100, 'Angle:', 90+angle_from_merge_point_degrees)
        print('velocity_right_vehicle:', right_velocity, 'new_x_right_vehicle:', -x_condition_unreal*100 - + offset_width, 'new_y_right_vehicle:', y_merge_point-y_condition_unreal*100, 'Angle:', 90-angle_from_merge_point_degrees)








    # def ComputeX_right(theta, right_difference):
    #     desired_x = width-right_difference*math.sin(theta)
    #     return desired_x
    #
    # def ComputeY_right(theta, right_difference):
    #     desired_y = height-right_difference*math.cos(theta)
    #     return desired_y
    #
    # desired_x_right = ComputeX(angle_from_merge_point, headway.right_initial_position_offset)
    # desired_y_right = ComputeY(angle_from_merge_point, headway.right_initial_position_offset)
    # desired_x_left = ComputeX(angle_from_merge_point, headway.left_initial_position_offset)
    # desired_y_left = ComputeY(angle_from_merge_point, headway.left_initial_position_offset)
    # print("The x value is for right vehicle: ", desired_x_right)
    # print("The y value is for right vehicle: ", (300-desired_y_right)*100)
    # print("The x value is for left vehicle: ", desired_x_left*100)
    # print("The y value is for left vehicle: ", desired_y_left*100)

#commit this2 -> this is wrong