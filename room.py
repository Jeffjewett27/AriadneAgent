from typing import NewType

from motion import MotionPrimitive
from physics.environment import Segment, Terrain
from sklearn.neighbors import KDTree


class SegmentConnection:
    from_segment: Segment
    to_segment: Segment
    reference_states: KDTree


class Room:
    """Contain info for a room"""

    def __init__(self, id: str, terrain: Terrain):
        self.id = id
        self.terrain = terrain

    def connect_segments(self):
        for segment in self.terrain.segments:
            pass