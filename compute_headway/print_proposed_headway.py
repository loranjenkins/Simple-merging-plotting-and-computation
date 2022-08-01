from compute_headway.conditiondefinition import ConditionDefinition
import math

if __name__ == '__main__':
# def _from_difference_at_point(left_velocity, right_velocity, left_headway, headway, distance, name):

    headway = ConditionDefinition._from_difference_at_point(45, 55, 0, 294, 'test')

    #geometry
    angle_from_merge_point = 0.10292525974279428 #radians
    width = 30
    height = 292.5

    print('right_initial_offset:', headway.right_initial_position_offset)
    print('left_initial_offset:', headway.left_initial_position_offset)

    def ComputeX(theta, right_difference):
        desired_x = width-right_difference*math.sin(theta)
        return desired_x

    def ComputeY(theta, right_difference):
        desired_y = height-right_difference*math.cos(theta)
        return desired_y

    desired_x_right = ComputeX(angle_from_merge_point, headway.right_initial_position_offset)
    desired_y_right = ComputeY(angle_from_merge_point, headway.right_initial_position_offset)
    desired_x_left = ComputeX(angle_from_merge_point, headway.left_initial_position_offset)
    desired_y_left = ComputeY(angle_from_merge_point, headway.left_initial_position_offset)
    print("The x value is for right vehicle: ", desired_x_right)
    print("The y value is for right vehicle: ", desired_y_right)
    print("The x value is for left vehicle: ", desired_x_left)
    print("The y value is for left vehicle: ", desired_y_left)

#commit this2