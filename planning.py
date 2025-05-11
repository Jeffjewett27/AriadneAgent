from dataclasses import dataclass
from typing import NewType

import numpy as np

from motion import MotionPrimitive
from physics.environment import Segment, Terrain
from physics.hero_controller import FIXED_UPDATE_FRAMERATE, forward_dynamics
from physics.hero_controller_state import HeroControllerStates
from physics.player_input import PlayerInput


MotionStateType = NewType("MotionStateType", np.ndarray)


@dataclass
class RRTNode:
    rrt_state: MotionStateType
    sim_state: HeroControllerStates
    parent: "RRTNode"
    control: MotionPrimitive
    cost_to_come: float


def sample_segment_connections(
    segment: Segment, terrain: Terrain, num_samples: int = 100
):
    pass
    # 1. segment
    start_coords = np.linspace(
        (segment.x_min, segment.y_min, 0), (segment.x_max, segment.y_max, 0), 5
    )
    nodes = [
        RRTNode(
            coord,
            HeroControllerStates(x_pos=coord[0], y_pos=coord[1], y_velocity=coord[2]),
            None,
            None,
            0,
        )
        for coord in start_coords
    ]
    bound_min = start_coords.min(axis=0)
    bound_max = start_coords.max(axis=0)

    def duration_schedule(i):
        # TODO anneal duration
        return np.random.random()

    for i in range(num_samples):
        state_rand = np.random.uniform(bound_min, bound_max)

        parent: RRTNode = min(nodes, key=lambda n: np.linalg.norm(n.rrt_state, state_rand))

        duration = duration_schedule(i)
        control = MotionPrimitive.sample(duration)
        full_state = parent.sim_state
        for action in PlayerInput.get_action_iterator(
            control.control_sequence(), FIXED_UPDATE_FRAMERATE
        ):
            full_state = forward_dynamics(
                full_state, terrain, action, 1 / FIXED_UPDATE_FRAMERATE
            )

        new_node = RRTNode(
            np.array([full_state.x_pos, full_state.y_pos, full_state.y_velocity]),
            full_state,
            parent,
            control,
            parent.cost_to_come + duration
        )
        nodes.append(new_node)
        bound_min = np.min([bound_min, new_node.rrt_state], axis=0)
