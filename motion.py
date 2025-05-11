"""Define parameterized motion primitives"""

from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np

from physics.player_input import PlayerInput


class MotionPrimitive(ABC):
    @abstractmethod
    def control_sequence(self):
        pass

    @classmethod
    def sample(cls, duration=None):
        if duration is None:
            duration = np.random.uniform(0.06, 1)
        # motion probabilities
        probs = [0.65, 0.25, 0.1]
        motion_idx = np.random.choice(len(probs), p=probs)
        if motion_idx == 0:
            return Jump.sample(duration)
        if motion_idx == 1:
            return Walk.sample(duration)
        if motion_idx == 2:
            return Wait.sample(duration)
        raise Exception()

@dataclass
class Wait(MotionPrimitive):
    # How long to do nothing
    wait_time: float

    def control_sequence(self):
        return [(PlayerInput(horizontal=0), self.wait_time)]
    
    @classmethod
    def sample(cls, duration):
        return Wait(duration)

@dataclass
class Walk(MotionPrimitive):
    # How long to hold horizontal
    walk_time: float
    # Whether the walk is to the right
    is_right: bool

    def control_sequence(self):
        dir = 1 if self.is_right else -1
        return [(PlayerInput(horizontal=dir), self.walk_time)]

    @classmethod
    def sample(cls, duration):
        is_right = np.random.choice([True, False])
        return Walk(walk_time=duration, is_right=is_right)


@dataclass
class Jump(MotionPrimitive):
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
