from __future__ import annotations

import numpy as np

from constants import FULL_JUMP_FALLING_COEFFICIENTS, MAX_JUMP_HEIGHT, MAX_JUMP_INCREASING_DISTANCE
from jumps import get_selected_jumps
from room import Room

straight_up_jumps = [
    np.array([[0,0], [0,MAX_JUMP_HEIGHT], [0.5, MAX_JUMP_HEIGHT]]),
    np.array([[0,0], [0,MAX_JUMP_HEIGHT*.8], [0.5, MAX_JUMP_HEIGHT*.8]]),
    np.array([[0,0], [0,MAX_JUMP_HEIGHT*.5], [0.5, MAX_JUMP_HEIGHT*.5]]),
    np.array([[0,0], [0,MAX_JUMP_HEIGHT*.8], [0.5, MAX_JUMP_HEIGHT*.3]]),
    np.array([[0,0], [0.5, 0]]),
]

# np.array([[0.0, 0.0], [0.828, 1.36], [1.658, 2.93], [2.48, 4.27], [3.308, 5.148], [4.143, 5.555]]),
# far_jumps = [
#     np.array([[0.0, 0.0], [0.828, 1.44], [1.644, 2.99], [2.43, 4.343], [3.318, 5.187], [4.137, 5.562], [4.975, 5.468], [5.812, 4.892], [6.64, 3.848], [7.456, 2.356], [8.289, 0.382], [9.128, 0.0]])
# ]
far_jumps = get_selected_jumps()

def flip_jump(jump):
    jump = jump.copy()
    jump[:,0] = -jump[:,0]
    return jump

class Floor:

    def __init__(self, x_min, y_level, x_max):

        self.x_min = x_min
        self.y_level = y_level
        self.x_max = x_max

        self.floor_neighbors: list[Floor] = []

    def is_other_floor_potentially_in_range(self, other: Floor):
        if other.y_level > self.y_level + MAX_JUMP_HEIGHT:
            # too high
            return False
        if other.x_min > self.x_max:
            # other is to the right
            # true if can reach during a jump ascend
            delta_x = other.x_min - self.x_max
            if delta_x <= MAX_JUMP_INCREASING_DISTANCE:
                return True
            # test the difference from the max jump height
            delta_y = other.y_level - (self.y_level + MAX_JUMP_HEIGHT)
            delta_x = delta_x - MAX_JUMP_INCREASING_DISTANCE
            fall_y = FULL_JUMP_FALLING_COEFFICIENTS[0] * delta_x + FULL_JUMP_FALLING_COEFFICIENTS[1] * delta_x**2 *0.8
            print('fall right', delta_x, delta_y, fall_y, delta_y < fall_y)
            return delta_y < fall_y
        elif other.x_max < self.x_min:
            # other is to the left
            delta_x = self.x_min - other.x_max
            if delta_x <= MAX_JUMP_INCREASING_DISTANCE:
                return True
            # test the difference from the max jump height
            delta_y = self.y_level + MAX_JUMP_HEIGHT - other.y_level
            delta_x = delta_x - MAX_JUMP_INCREASING_DISTANCE
            fall_y = FULL_JUMP_FALLING_COEFFICIENTS[0] * delta_x + FULL_JUMP_FALLING_COEFFICIENTS[1] * delta_x**2*0.8
            print('fall left', delta_x, delta_y, fall_y, delta_y < fall_y)
            return delta_y < fall_y
        elif other.y_level > self.y_level:
            # other is directly above
            return True
        elif other.x_min < self.x_min or other.x_max > self.x_max:
            # other is below, but juts out
            return True 
        return False
    
    def is_other_floor_in_range(self, other: Floor, room: Room):
        if other.y_level > self.y_level + MAX_JUMP_HEIGHT:
            # too high
            return False
        if other.x_min > self.x_max:
            # other is to the right
            # true if can reach during a jump ascend
            delta_x = other.x_min - self.x_max
            if delta_x <= MAX_JUMP_INCREASING_DISTANCE:
                return True
            # test the difference from the max jump height
            delta_y = other.y_level - (self.y_level + MAX_JUMP_HEIGHT)
            delta_x = delta_x - MAX_JUMP_INCREASING_DISTANCE
            fall_y = FULL_JUMP_FALLING_COEFFICIENTS[0] * delta_x + FULL_JUMP_FALLING_COEFFICIENTS[1] * delta_x**2 *0.8
            print('fall right', delta_x, delta_y, fall_y, delta_y < fall_y)
            return delta_y < fall_y
        elif other.x_max < self.x_min:
            # other is to the left
            delta_x = self.x_min - other.x_max
            if delta_x <= MAX_JUMP_INCREASING_DISTANCE:
                return True
            # test the difference from the max jump height
            delta_y = self.y_level + MAX_JUMP_HEIGHT - other.y_level
            delta_x = delta_x - MAX_JUMP_INCREASING_DISTANCE
            fall_y = FULL_JUMP_FALLING_COEFFICIENTS[0] * delta_x + FULL_JUMP_FALLING_COEFFICIENTS[1] * delta_x**2*0.8
            print('fall left', delta_x, delta_y, fall_y, delta_y < fall_y)
            return delta_y < fall_y
        elif other.y_level > self.y_level:
            # other is directly above

            # try jump from left
            xpos = max(self.x_min, other.x_min) - 0.001
            status, cast_x, cast_y, final_region = room.region_raycast(xpos, self.y_level, xpos, self.y_level + MAX_JUMP_HEIGHT)
            print('try jump left', status, cast_x, self.y_level, cast_y, self.y_level + MAX_JUMP_HEIGHT, final_region)
            if cast_y >= other.y_level:
                status, cast_x, cast_y, final_region = room.region_raycast(cast_x, cast_y, cast_x + 0.1, cast_y)
                print('try jump left2', status, cast_x, cast_y, final_region, "NR" if (final_region is None) else final_region.over_floor, other)
                if final_region is not None and final_region.over_floor == other:
                    return True
                
            # try jump from right
            xpos = min(self.x_max, other.x_max) + 0.001
            status, cast_x, cast_y, final_region = room.region_raycast(xpos, self.y_level, xpos, self.y_level + MAX_JUMP_HEIGHT)
            print('try jump right', status, cast_x, self.y_level, cast_y, self.y_level + MAX_JUMP_HEIGHT, final_region)
            if cast_y >= other.y_level:
                status, cast_x, cast_y, final_region = room.region_raycast(cast_x, cast_y, cast_x - 0.1, cast_y)
                print('try jump right2', status, cast_x, cast_y, final_region, "NR" if (final_region is None) else final_region.over_floor, other)
                if final_region is not None and final_region.over_floor == other:
                    return True
            return False
        elif other.x_min < self.x_min or other.x_max > self.x_max:
            # other is below, but juts out
            return True 
        return False

    def check_other_floor(self, other: Floor, room: Room):
        if other.y_level > self.y_level + MAX_JUMP_HEIGHT:
            # too high
            return False
        if self.x_max < other.x_max:
            # other is to the right
            right_x_check = self.x_max - 0.1 # add a bit of padding
            for jump in far_jumps:
                if self.check_jump_over_floor((right_x_check, self.y_level+0.01), jump, other, room):
                    return True
                
        if other.x_min < self.x_min:
            # other is to the left
            left_x_check = self.x_min + 0.1 # add a bit of padding
            for jump in far_jumps:
                if self.check_jump_over_floor((left_x_check, self.y_level+0.01), flip_jump(jump), other, room):
                    return True
        
        if other.y_level > self.y_level:
            if self.x_min < other.x_min <= self.x_max:
                # other is directly above
                left_x_check = max(self.x_min, other.x_min - 0.1) # add a bit of padding
                if left_x_check < other.x_min:
                    for jump in straight_up_jumps:
                        if self.check_jump_over_floor((left_x_check, self.y_level+0.01), jump, other, room):
                            return True
                    
            if self.x_min <= other.x_max < self.x_max:
                right_x_check = min(self.x_max, other.x_max + 0.1) # add a bit of padding
                if other.x_max < right_x_check:
                    for jump in straight_up_jumps:
                        if self.check_jump_over_floor((right_x_check, self.y_level+0.01), flip_jump(jump), other, room):
                            return True
        # elif other.x_min < self.x_min or other.x_max > self.x_max:
        #     # other is below, but juts out
        #     return False 
        return False
    
    def check_jump_over_floor(self, from_location, jump, floor: Floor, room: Room):
        jump_path = jump + np.array(from_location)
        raygen = room.region_pathcast_generator(jump_path)
        for status, x, y, region in raygen:
            if status == 'failed':
                return False
            if region is not None and region.over_floor == floor:
                return True
        return False
        
    def connect_to_other_floors(self, floors, room: Room):
        for other in floors:
            if self.check_other_floor(other, room):
                self.floor_neighbors.append(other)

    def __lt__(self, other): 
        return False