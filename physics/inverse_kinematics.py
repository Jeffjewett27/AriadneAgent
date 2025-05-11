import numpy as np
from motion import Jump, Walk
from physics.environment import Terrain
from physics.hero_controller import (
    FIXED_DELTA_TIME,
    GRAVITY,
    JUMP_SPEED,
    JUMP_TIME,
    JUMP_TIME_MIN,
    MAX_FALL_VELOCITY,
    RUN_SPEED,
)
from physics.hero_controller_state import HeroControllerStates


def inverse_kinematics(
    from_state: HeroControllerStates, to_state: HeroControllerStates, terrain: Terrain
):
    """Find a motion from from_state to to_state. Meant for minor adjustments without obstacles.

    Input:
    from_state (HeroControllerStates): the original knight state
    to_state (HeroControllerStates): the target knight state
    terrain (Terrain): the terrain for collision checks

    Returns: (success: bool, motion: MotionPrimitive)
    """
    if from_state.onGround and to_state.onGround:
        if np.abs(from_state.y_pos - to_state.y_pos) < 1e-2:
            hit, _, _, _ = terrain.linecast(
                from_state.x_pos,
                from_state.y_pos + 0.01,
                to_state.x_pos,
                to_state.y_pos + 0.01,
            )
            if hit:
                return False, None
            # grounded on same level
            displacement = to_state.x_pos - from_state.x_pos
            time = np.abs(displacement) / RUN_SPEED
            is_right = displacement > 0
            return True, Walk(time, is_right)
        # else:
        #     # grounded on different level; don't bother with IK
        #     return False, None

    jump_time, total_time = inverse_jump_time2(
        from_state, to_state.y_pos, to_state.y_velocity
    )
    print("inv jump", jump_time, total_time)
    if jump_time is None:
        # could not reach target
        return False, None
    displacement = to_state.x_pos - from_state.x_pos
    x_travel_time = np.abs(displacement) / RUN_SPEED
    is_right = displacement > 0
    # if x_travel_time > total_time:
    #     # cannot reach target before falling too far
    #     return False, None

    return True, Jump(jump_time, is_right, 0, x_travel_time, total_time - x_travel_time)


def get_apex_height_range(state: HeroControllerStates, dt: float = FIXED_DELTA_TIME):
    """
    Get the minimum and maximum apex y_pos according to a jump from this state.

    If grounded and can jump, calculates from a fresh jump. For maximum, holds
    jump button. For minimum, does not hold jump. Assume no ceiling.

    Input:
    state (HeroControllerStates): the state to calculate from
    dt (float): the delta time for integration
    """

    jump_time = state.jump_time
    if state.onGround:
        jump_time = JUMP_TIME
    elif state.y_velocity < 0:
        # knight is falling. apex reached
        return (state.y_pos, state.y_pos)

    min_y_pos = None
    y_pos = state.y_pos
    y_vel = state.y_velocity
    while jump_time > 0:
        jump_time -= dt
        if min_y_pos is None and jump_time < JUMP_TIME - JUMP_TIME_MIN:
            # minimum apex reached
            min_y_pos = y_pos
        y_vel = JUMP_SPEED - GRAVITY * dt
        y_pos += y_vel * dt
    if min_y_pos is None:
        min_y_pos = y_pos

    while y_vel > 0:
        y_vel -= GRAVITY * dt
        y_pos += y_vel * dt

    return (min_y_pos, y_pos)


def get_inverse_best_apex(
    state: HeroControllerStates,
    target_y_pos: float,
    target_y_vel: float | None,
    dt: float = FIXED_DELTA_TIME,
):
    """
    Given a state and target, find the optimal height of a simple jump.

    If optimal height cannot be achieved, give the closest height.
    If already falling, predict velocity at target.

    Return (best_height, predicted_velocity, jump_hold_time, total_time)
    """

    apex_min, apex_max = get_apex_height_range(state, dt=dt)

    best_y_pos = target_y_pos
    y_vel = target_y_vel if target_y_vel is not None else -100000
    y_vel_thresh = min(0, state.y_velocity)
    fall_time = 0
    while y_vel < y_vel_thresh and best_y_pos < apex_max:
        best_y_pos -= y_vel * dt
        y_vel += GRAVITY * dt
        fall_time += dt

    # if target_y_vel is None:
    #     #

    if y_vel >= y_vel_thresh and apex_min <= best_y_pos <= apex_max:
        # optimal solution found
        jump_time = (best_y_pos - state.y_pos) / (JUMP_SPEED - GRAVITY * dt) * dt
        return best_y_pos, target_y_vel, jump_time, jump_time + fall_time

    if y_vel < y_vel_thresh:
        # can not reach high enough to achieve fall velocity
        jump_time = (apex_max - state.y_pos) / (JUMP_SPEED - GRAVITY * dt) * dt
        return (
            apex_max,
            y_vel_thresh + (target_y_vel - y_vel),
            jump_time,
            jump_time + fall_time,
        )

    if best_y_pos < apex_min:
        # current jump is too high. will overshoot fall velocity
        jump_time = 0
        return apex_min, y_vel_thresh + (target_y_vel - y_vel), jump_time, fall_time

    if best_y_pos > apex_max:
        # current jump is not high enough. will undershoot fall veloicity
        jump_time = (apex_max - state.y_pos) / (JUMP_SPEED - GRAVITY * dt) * dt
        return (
            apex_max,
            y_vel_thresh + (target_y_vel - y_vel),
            jump_time,
            jump_time + fall_time,
        )

    raise Exception("Edge case")


def inverse_jump_time(
    state: HeroControllerStates,
    target_y_pos: float,
    target_y_vel: float | None,
    dt: float = FIXED_DELTA_TIME,
):
    tolerance = GRAVITY * dt

    def within_tolerance(y_vel):
        return np.abs(y_vel - target_y_vel) < tolerance

    max_hold_time = 0.6 if state.onGround else state.jump_time
    min_vel, max_time = get_velocity_at_target_after_jump(
        state, target_y_pos, max_hold_time
    )
    if min_vel is None:
        # could not reach target_y_pos
        return None, None
    if target_y_vel is None or within_tolerance(min_vel):
        # if target velocity is none, return max hold time
        return max_hold_time, max_time

    min_hold_time = 0.08
    max_vel, min_time = get_velocity_at_target_after_jump(
        state, target_y_pos, min_hold_time
    )
    if max_vel is not None and within_tolerance(max_vel):
        return 0, min_time

    if target_y_vel < min_vel or (max_vel is not None and target_y_vel > max_vel):
        # cannot achieve target
        return None, None

    print('search range', min_vel, max_vel)
    best_vel, best_time, best_hold_time = min_vel, max_time, max_hold_time
    while max_hold_time - min_hold_time >= 1.5*dt:
        # binary search for best hold time
        mid_hold_time = (max_hold_time + min_hold_time) / 2
        mid_hold_time = round(mid_hold_time / dt) * dt
        mid_vel, mid_time = get_velocity_at_target_after_jump(
            state, target_y_pos, mid_hold_time
        )
        print('mid', mid_vel, mid_hold_time, (min_hold_time, max_hold_time), mid_time)
        if mid_vel is None:
            mid_vel = np.inf
        if np.abs(target_y_vel - mid_vel) <= np.abs(target_y_vel - best_vel):
            best_vel = mid_vel
            best_time = mid_time
        if within_tolerance(mid_vel):
            return mid_hold_time, mid_time
        if mid_vel > target_y_vel:
            min_hold_time = mid_hold_time
        else:
            max_hold_time = mid_hold_time

    # relax tolerance
    tolerance *= 3
    if within_tolerance(mid_vel) or True:
        return best_hold_time, best_time
    raise Exception("Velocity was in range, but could not be found")

def inverse_jump_time2(
    state: HeroControllerStates,
    target_y_pos: float,
    target_y_vel: float | None,
    dt: float = FIXED_DELTA_TIME,
):
    hold_time = 0.08
    max_hold_time = 0.6 if state.onGround else state.jump_time

    if target_y_vel is None:
        vel, time = get_velocity_at_target_after_jump(
            state, target_y_pos, max_hold_time
        )
        if vel is None:
            return None, None
        return max_hold_time, time

    best_vel, best_time, best_hold_time = np.inf, None, None
    while hold_time <= max_hold_time:
        vel, time = get_velocity_at_target_after_jump(
            state, target_y_pos, hold_time
        )
        print('search', hold_time, vel, time)
        
        if vel is None:
            hold_time += dt
            continue
        if np.abs(target_y_vel - vel) <= np.abs(target_y_vel - best_vel):
            best_hold_time = hold_time
            best_vel = vel
            best_time = time
        hold_time += dt

    if best_time is None:
        return None, None
        
    return best_hold_time, best_time


def get_velocity_at_target_after_jump(
    state: HeroControllerStates,
    target_y_pos: float,
    hold_jump_time: float,
    dt: float = FIXED_DELTA_TIME,
):
    """Simulate vertical motion holding jump for a duration.
    Get the resultant y velocity at a target."""
    jump_time = state.jump_time
    if state.onGround:
        jump_time = JUMP_TIME

    y_pos = state.y_pos
    y_vel = state.y_velocity
    total_time = 0

    if state.y_velocity >= 0:
        # Stage 1: Sustained ascend
        y_vel = JUMP_SPEED - GRAVITY * dt
        while jump_time > 0:
            if hold_jump_time <= 0 and jump_time < JUMP_TIME - JUMP_TIME_MIN:
                y_vel = 0
                break
            jump_time -= dt
            hold_jump_time -= dt
            y_pos += y_vel * dt
            total_time += dt
            # print('s1', y_pos, y_vel, y_vel*dt)

        # Stage 2: Decelerating ascend
        while y_vel > 0:
            if hold_jump_time <= 0:
                y_vel = 0
                break
            hold_jump_time -= dt
            y_vel -= GRAVITY * dt
            y_pos += y_vel * dt
            total_time += dt
            # print('s2', y_pos, y_vel, y_vel*dt)

    if target_y_pos > y_pos:
        # didn't reach target position
        return None, None

    # Stage 2: Fall
    y_vel = 0
    while y_pos > target_y_pos:
        if y_vel < -MAX_FALL_VELOCITY:
            y_vel = -MAX_FALL_VELOCITY
        y_vel -= GRAVITY * dt
        y_pos += y_vel * dt
        total_time += dt
        # print('s3', y_pos, y_vel, y_vel*dt)

    return y_vel, total_time
