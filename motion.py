"""Define parameterized motion primitives"""

from collections import deque
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np

from physics.hero_controller import RUN_SPEED
from physics.hero_controller_state import HeroControllerStates
from physics.player_input import PlayerInput


class MotionPrimitive(ABC):

    @property
    def unwrapped(self):
        return self

    @property
    def is_terminated(self):
        return False

    @abstractmethod
    def reset_state(self, hero_state: HeroControllerStates) -> None:
        pass

    @abstractmethod
    def get_action(self, hero_state: HeroControllerStates) -> PlayerInput:
        pass

    @classmethod
    def sample(cls, duration=None):
        if duration is None:
            duration = np.random.uniform(0.06, 1)
        # motion probabilities
        probs = [0.75, 0.25, 0]
        motion_idx = np.random.choice(len(probs), p=probs)
        if motion_idx == 0:
            return JumpDistance.sample(duration)
        if motion_idx == 1:
            return WalkDistance.sample(duration)
        if motion_idx == 2:
            return Wait.sample(duration)
        raise Exception()


class OpenLoopMotionPrimitive(MotionPrimitive):

    @property
    def is_terminated(self):
        return self._is_terminated
    
    def reset_state(self, hero_state: HeroControllerStates):
        self._control_iter = PlayerInput.get_action_iterator(
            self.control_sequence(), PlayerInput.FRAMERATE
        )
        try:
            self._next_control = next(self._control_iter)
            self._is_terminated = False
        except StopIteration:
            self._is_terminated = True 

    def get_action(self, hero_state):
        if self.is_terminated:
            return PlayerInput(), True
        try:
            new_control = next(self._control_iter)
            old_control, self._next_control = self._next_control, new_control
            self._is_terminated = False
            return old_control, False
        except StopIteration:
            self._is_terminated = True
            return self._next_control, True

    @abstractmethod
    def control_sequence(self) -> list[tuple[PlayerInput, float]]:
        pass

    @abstractmethod
    def trim_duration(self, duration: float) -> "MotionPrimitive":
        pass


class CompositePrimitive(OpenLoopMotionPrimitive):
    def __init__(self, primitives: list[MotionPrimitive], max_duration: float = None):
        if primitives is None:
            primitives = []
        self._primitives = primitives
        self._max_duration = max_duration

    def control_sequence(self):
        combined_cs = [
            control for prim in self._primitives for control in prim.control_sequence()
        ]

        if self._max_duration is not None:
            i = 0
            cum_duration = 0
            clipped_cs = []
            while i < len(combined_cs):
                duration = combined_cs[i][1]
                cum_duration += duration
                if cum_duration < self._max_duration:
                    clipped_cs.append(combined_cs[i])
                else:
                    clipped_cs.append(
                        (combined_cs[i][0], self._max_duration - cum_duration)
                    )
                    break
                i += 1
            combined_cs = clipped_cs
        return combined_cs

    def trim_duration(self, duration):
        return CompositePrimitive(self._primitives, min(self._max_duration, duration))
    
    @classmethod
    def merge_primitives(cls, primitives):
        if len(primitives) == 0:
            return []
        prev_prim = primitives[0]
        primitives = deque(primitives)
        merged = []
        while len(primitives) > 0:
            cur_prim = primitives.pop()
            if isinstance(cur_prim, CompositePrimitive):
                for cprim in cur_prim._primitives.reverse():
                    primitives.appendleft(cprim)
                continue
            if isinstance(prev_prim.unwrapped, Walk) and isinstance(cur_prim.unwrapped, Walk):
                turn_threshold = 0.3
                cur_prim = cur_prim.unwrapped
                prev_prim = prev_prim.unwrapped
                if prev_prim.is_right == cur_prim.is_right:
                    cur_prim = Walk(prev_prim.walk_time + cur_prim.walk_time, prev_prim.is_right)
                elif prev_prim.walk_time < turn_threshold or cur_prim.walk_time < turn_threshold:
                    if prev_prim.walk_time > cur_prim.walk_time:
                        cur_prim = Walk(prev_prim.walk_time - cur_prim.walk_time, prev_prim.is_right)
                    else:
                        cur_prim = Walk(cur_prim.walk_time - prev_prim.walk_time, cur_prim.is_right)
                else:
                    merged.append(prev_prim)
            elif isinstance(prev_prim.unwrapped, WalkDistance) and isinstance(cur_prim.unwrapped, WalkDistance):
                turn_threshold = 0.3 * RUN_SPEED
                cur_prim = cur_prim.unwrapped
                prev_prim = prev_prim.unwrapped
                if prev_prim.is_right == cur_prim.is_right:
                    cur_prim = WalkDistance(prev_prim.distance + cur_prim.distance, prev_prim.is_right)
                elif prev_prim.distance < turn_threshold or cur_prim.distance < turn_threshold:
                    if prev_prim.distance > cur_prim.distance:
                        cur_prim = Walk(prev_prim.distance - cur_prim.distance, prev_prim.is_right)
                    else:
                        cur_prim = Walk(cur_prim.distance - prev_prim.distance, cur_prim.is_right)
                else:
                    merged.append(prev_prim)
            else:
                merged.append(prev_prim)
            prev_prim = cur_prim
        merged.append(prev_prim)
        return merged
        



@dataclass
class Wait(OpenLoopMotionPrimitive):
    # How long to do nothing
    wait_time: float

    def control_sequence(self):
        return [(PlayerInput(horizontal=0), self.wait_time)]

    def trim_duration(self, duration):
        return Wait(min(self.wait_time, duration))

    @classmethod
    def sample(cls, duration):
        return Wait(duration)


@dataclass
class Walk(OpenLoopMotionPrimitive):
    # How long to hold horizontal
    walk_time: float
    # Whether the walk is to the right
    is_right: bool

    def control_sequence(self):
        dir = 1 if self.is_right else -1
        return [(PlayerInput(horizontal=dir), self.walk_time)]

    def trim_duration(self, duration):
        return Walk(min(self.walk_time, duration), self.is_right)

    @classmethod
    def sample(cls, duration):
        is_right = np.random.choice([True, False])
        return Walk(walk_time=duration, is_right=is_right)


class JumpDistance(MotionPrimitive):

    def __init__(self, 
        jump_hold_height: float, 
        is_right: bool,
        x_wait_height: float,
        x_distance: float
    ):
        self.jump_hold_height = jump_hold_height
        self.is_right = is_right
        self.x_wait_height = x_wait_height
        self.x_distance = x_distance

    def reset_state(self, hero_state):
        self._x_start = hero_state.x_pos
        self._y_start = hero_state.y_pos
        self._is_x_waiting = True
        self._is_holding_jump = True
        self._is_terminated = False
    
    def get_action(self, hero_state):
        if hero_state.onGround and not self._is_holding_jump:
            self._is_terminated = True
        if self._is_terminated:
            return PlayerInput(), True
        delta_x = (hero_state.x_pos - self._x_start)
        delta_y = (hero_state.y_pos - self._y_start)
        if delta_y >= self.jump_hold_height or hero_state.y_velocity < 0:
            self._is_holding_jump = False
        if delta_y >= self.x_wait_height or hero_state.y_velocity < 0:
            self._is_x_waiting = False
        x_err_thresh = 0.3
        horizontal = 0
        if delta_x < -x_err_thresh:
            horizontal = 1
        elif delta_x > x_err_thresh:
            horizontal = -1
        return PlayerInput(jump=self._is_holding_jump, horizontal=horizontal), False
    
    @property
    def is_terminated(self):
        return self._is_terminated
    
    @classmethod
    def sample(cls, duration):
        jump = min(np.random.uniform(0.5, 10), 5.2) # sample more at max height
        is_right = np.random.choice([True, False])
        x_wait_height = max(np.random.uniform(-10, 5.2), 0) # sample more at 0 height
        x_distance = np.random.uniform(0.5, 15)
        return JumpDistance(jump, is_right, x_wait_height, x_distance)
        

@dataclass
class Jump(OpenLoopMotionPrimitive):
    # How long to hold jump
    jump_time: float
    # Whether the jump is to the right
    is_right: bool
    # How long to not move horizontally
    x_wait: float
    # After x_wait, how long to move horizontally
    x_hold_time: float
    # After x_wait+x_hold_time, how long to not move horizontally
    x_pause: float

    def control_sequence(self):
        dir = 1 if self.is_right else -1
        actions = []
        jump_time = self.jump_time
        if self.x_wait > 0:
            x_wait = self.x_wait
            if 0 < jump_time < x_wait:
                actions.append((PlayerInput(horizontal=0, jump=True), jump_time))
                x_wait -= jump_time
                jump_time = 0
            actions.append((PlayerInput(horizontal=0, jump=jump_time > 0), x_wait))
            jump_time = max(jump_time - x_wait, 0)
        if self.x_hold_time > 0:
            x_hold = self.x_hold_time
            if 0 < jump_time < x_hold:
                actions.append((PlayerInput(horizontal=dir, jump=True), jump_time))
                x_hold -= jump_time
                jump_time = 0
            actions.append((PlayerInput(horizontal=dir, jump=jump_time > 0), x_hold))
            jump_time = max(jump_time - x_hold, 0)
        if self.x_pause > 0:
            x_wait = self.x_pause
            if 0 < jump_time < x_wait:
                actions.append((PlayerInput(horizontal=0, jump=True), jump_time))
                x_wait -= jump_time
                jump_time = 0
            actions.append((PlayerInput(horizontal=0, jump=jump_time > 0), x_wait))
            jump_time = max(jump_time - x_wait, 0)
        return actions

    def trim_duration(self, duration):
        cum_time = self.x_wait
        if duration < cum_time:
            return Jump(
                min(self.jump_time, duration),
                self.is_right,
                min(self.x_wait, duration),
                0,
                0,
            )
        cum_time += self.x_hold_time
        if duration < cum_time:
            return Jump(
                min(self.jump_time, duration),
                self.is_right,
                self.x_wait,
                min(self.x_hold_time, duration),
                0,
            )
        cum_time += self.x_pause
        if duration < cum_time:
            return Jump(
                min(self.jump_time, duration),
                self.is_right,
                self.x_wait,
                self.x_hold_time,
                min(self.x_pause, duration),
            )
        return self

    @classmethod
    def sample(cls, duration):
        jump = np.random.uniform(0, 1)
        is_right = np.random.choice([True, False])
        bases = [
            np.array([0, 1, 0]),  # hold horizontal
            np.array([0.8, 0.2, 0]),  # long pause before moving slightly
            np.array([0.2, 0.8, 0]),  # slight pause before moving long
            np.array([0, 0.5, 0.5]),  # move a bit then stop
        ]
        probs = [0.75, 0.1, 0.1, 0.05]
        rand_idx = np.random.choice(len(bases), p=probs)
        basis = bases[rand_idx] * duration
        return Jump(jump, is_right, *basis[:3])
    
class MotionPrimitiveWrapper(MotionPrimitive):

    def __init__(self, primitive: MotionPrimitive):
        self._primitive = primitive

    @property
    def unwrapped(self):
        return self._primitive.unwrapped
    
    @property
    def is_terminated(self):
        return self._primitive.is_terminated
    
    def reset_state(self, hero_state):
        return self._primitive.reset_state(hero_state)
    
    def get_action(self, hero_state):
        return self._primitive.get_action(hero_state)
    

class TerminateOnGroundWrapper(MotionPrimitiveWrapper):

    def __init__(self, primitive: MotionPrimitive):
        super().__init__(primitive)
        self._has_landed = False
        self._has_left_ground = False

    def reset_state(self, hero_state):
        super().reset_state(hero_state)
        self._has_landed = False
        self._has_left_ground = False

    def get_action(self, hero_state: HeroControllerStates):
        if hero_state.onGround:
            if self._has_left_ground:
                self._has_landed = True
        else:
            self._has_left_ground = True
        if self._has_landed:
            return PlayerInput(), True
        if not self._primitive.is_terminated:
            action, terminated = self._primitive.get_action(hero_state)
        else:
            action, terminated = PlayerInput(), False
        if terminated and not self._has_left_ground:
            self._has_landed = True
        return action, self.is_terminated
    
    @property
    def is_terminated(self):
        return self._has_landed
        
    def __str__(self):
        return f'TerminateOnGround[{self._primitive}]'

class WalkDistance(MotionPrimitive):

    def __init__(self, distance: float, is_right: bool):
        self.distance = distance
        self.is_right = is_right

    def reset_state(self, hero_state):
        self._start_x = hero_state.x_pos
        self._num_failed = 0
        self._prev_x = self._start_x
        self._is_terminated = False

    def get_action(self, hero_state):
        dir = 1 if self.is_right else -1
        if (hero_state.x_pos - self._prev_x) * dir <= 0:
            # knight must keep moving right
            self._num_failed += 1
        if (hero_state.x_pos - self._start_x) * dir >= self.distance:
            self._is_terminated = True
            dir = 0
        if self._num_failed > 20: # roughly 0.4 seconds
            self._is_terminated = True
            dir = 0
        return PlayerInput(horizontal=dir), self.is_terminated

    @property
    def is_terminated(self):
        return self._is_terminated
    
    @classmethod
    def sample(cls, duration):
        is_right = np.random.choice([True, False])
        distance = RUN_SPEED / duration
        return WalkDistance(distance=distance, is_right=is_right)
    
