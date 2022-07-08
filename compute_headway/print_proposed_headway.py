from compute_headway.conditiondefinition import ConditionDefinition

headway = ConditionDefinition._from_difference_at_point(80, 30, 0, 100, 'test')

print('right_initial_offset:', headway.right_initial_position_offset)
print('left_initial_offset:', headway.left_initial_position_offset)