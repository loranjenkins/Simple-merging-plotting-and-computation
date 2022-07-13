from compute_headway.conditiondefinition import ConditionDefinition
import math


# def _from_difference_at_point(left_velocity, right_velocity, left_headway, headway distance, name):

headway = ConditionDefinition._from_difference_at_point(50, 50, 0, 294, 'test')

print('right_initial_offset:', headway.right_initial_position_offset)
print('left_initial_offset:', headway.left_initial_position_offset)

def ComputeX(theta, right_difference):
    desired_x = 30-right_difference*math.sin(theta)
    return desired_x

def ComputeY(theta, right_difference):
    desired_y = 292.5-right_difference*math.cos(theta)
    return desired_y

desired_x_right = ComputeX(0.102206718, headway.right_initial_position_offset)
desired_y_right = ComputeY(0.102206718, headway.right_initial_position_offset)
desired_x_left = ComputeX(0.102206718, headway.left_initial_position_offset)
desired_y_left = ComputeY(0.102206718, headway.left_initial_position_offset)
print("The x value is for right vehicle: ", desired_x_right)
print("The y value is for right vehicle: ", desired_y_right)
print("The x value is for left vehicle: ", desired_x_left)
print("The y value is for left vehicle: ", desired_y_left)