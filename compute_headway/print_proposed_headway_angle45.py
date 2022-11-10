from compute_headway.conditiondefinition import ConditionDefinition
import math
import numpy as np

if __name__ == '__main__':
# def _from_difference_at_point(left_velocity, right_velocity, left_headway, headway, distance, name):


    #geometry axis: upwards y = negative upwards x = positive
    initial_start_point_y = 5
    offset_width = 0
    width = 215
    height = 215
    length_tunnel = 120
    y_merge_point = (height+initial_start_point_y)*100
    angle_from_merge_point = math.atan(width/height) #radians
    angle_from_merge_point_degrees = angle_from_merge_point*(180/math.pi)
    print('Angle in radians:', angle_from_merge_point)
    print('Angle in degrees:', angle_from_merge_point*(180/math.pi))
    section_length = np.sqrt(width ** 2 + height ** 2)

    print('section length:', section_length)


# def _from_difference_at_point(left_velocity, right_velocity, left_headway, distance, name):
    left_velocity = 45 #vehicle1
    right_velocity = 55 #vehicle2

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

