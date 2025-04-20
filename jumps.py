import json
import os

import numpy as np

import control
from visualize import interactive_jumps


prev_state = None
was_grounded = True
current_jump = []
prev_room = None

logged_jumps = []
jump_file = 'jumps.txt'
selected_jumps_file = 'selected_jumps.txt'

def log_knight_state(room, knight, player_state, time):
    global prev_state, was_grounded, current_jump, prev_room, logged_jumps
    if room != prev_room:
        save_jumps(logged_jumps)
        current_jump = []
        was_grounded = False
    kx, ky = knight['x'], knight['y']
    grounded = room.is_grounded(kx, ky)
    state = (time, kx, ky)
    if prev_state is not None and was_grounded and not grounded:
        # new jump
        current_jump = [prev_state, state]
    elif not grounded:
        current_jump.append(state)
    elif not was_grounded and len(current_jump) > 0:
        current_jump.append(state)
        # print('jump:', current_jump)
        process_jump(current_jump)
    prev_state = state
    was_grounded = grounded
    prev_room = room
    # print('knight', kx, ky, grounded, [float(x) for x in floor[:3]], region)
    # draw_region(screen_array, screen_transform, region)

def process_jump(jump):
    if len(jump) < 3:
        return
    t0, x0, y0 = jump[0]
    xlast = jump[-1][1]
    if xlast >= x0:
        tranformed = [(t-t0, x-x0, y-y0) for t, x, y in jump]
    else:
        tranformed = [(t-t0, x0-x, y-y0) for t, x, y in jump]
    print(tranformed)
    logged_jumps.append(tranformed)

def save_jumps(jumps):
    if os.path.exists(jump_file):
        with open(jump_file, 'r') as file:
            data = json.load(file)
    else:
        data = []
    all_jumps = [*data, *jumps]
    with open(jump_file, 'w') as file:
        json.dump(all_jumps, file, indent=2)

def get_selected_jumps():
    if os.path.exists(jump_file):
        with open(selected_jumps_file, 'r') as file:
            data = json.load(file)
    else:
        data = []
    falling = np.array([(0.0, 0.0), (0.8389999999999986, -0.5989999999999966), (1.6660000000000004, -1.661999999999999), (2.4919999999999973, -3.1949999999999985), (3.317999999999998, -5.170999999999998)])
    jumps = [np.array(jump)[:,1:] for jump in data]
    jumps = [np.concatenate((jump, falling + jump[-1,:]), axis=0) for jump in jumps]
    jumps = [*jumps, *[ np.array(list(zip(-jump[:,0], jump[:,1]))) for jump in jumps]]
    print(jumps)
    return jumps

def do_jump(game, peak_height, target_x, min_y_before_x):
    x0, y0 = game.knight_position
    t0 = game.time
    if x0 < target_x:
        press_move = control.press_right
        release_move = control.release_right
    else:
        press_move = control.press_left
        release_move = control.release_left
    target_dir = 1 if target_x > x0 else -1

    x, y, t = x0, y0, t0
    dx, dy = 0, 0
    # condition (time, x_rel, y_rel, y_vel)
    # def min_height_condition(t, dx, dy, vx, vy):
    # action: 
    control.press_jump()
    yield False

    is_pressing_jump = True
    has_left_ground = False
    is_grounded = game.is_grounded(x, y)
    num_iters = 0
    while not (has_left_ground and is_grounded):
        x, y = game.knight_position
        vx, vy = (x - x0 - dx), (y - y0 - dy)
        dx, dy = (x - x0, y - y0)
        t = game.time

        padding = 0.2 if abs(vx) < 0.1 else 2
        if (target_x - x) * target_dir < padding:
            release_move()
        elif dy >= min_y_before_x:
            press_move()
        if dy >= peak_height:
            control.release_jump()
            is_pressing_jump = False
        yield False
        num_iters += 1
        is_grounded = game.is_grounded(x, y)
        has_left_ground = has_left_ground or (not is_grounded)
    print('jumped for ', num_iters, 'is grounded', game.is_grounded(x,y))
    release_move()
    control.release_jump()
    yield True


if __name__ == "__main__":
    if os.path.exists(jump_file):
        with open(jump_file, 'r') as file:
            data = json.load(file)
    else:
        data = []

    max_y = 0
    up_jump_segments = []
    down_jump_segments = []
    segments = []
    for jump in data:
        idx = 0
        y_level = 0
        while idx < len(jump) and y_level <= jump[idx][2]:
            y_level = jump[idx][2]
            max_y = max(max_y, y_level)
            idx += 1
        up_idx = idx
        while idx < len(jump) and y_level >= jump[idx][2]:
            y_level = jump[idx][2]
            idx += 1
        if idx == len(jump):
            up_jump_segments.append(jump[:up_idx])
            if up_idx < len(jump):
                t0, x0, y0 = jump[up_idx]
                down_jump_segments.append([(t-t0, x-x0, y-y0) for t, x, y in jump[up_idx:]])
            segments.append(jump)
        

        # vy = [s1[1] - s0[2] for s0, s1 in zip(jump[:-1], jump[1:])]

        # ay = [v1 - v0 for v0, v1 in zip(vy[:-1], vy[1:])]
        # if len([a for a in ay if a > 0]) <= 2:
        #     filtered_jumps.append(jump)
    # print('max jump height', max_y)
    # interactive_jumps(up_jump_segments)
    # interactive_jumps(down_jump_segments)
    interactive_jumps(segments)