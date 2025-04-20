import os
import numpy as np
import pyclipr
from itertools import cycle
import pickle

import pyclipr.pyclipr

from physics.environment import Terrain


def process_terrain(game_data, use_cache=True) -> Terrain:
    terrain = game_data.terrain_segmentation

    room_cache_path = os.path.join("rooms", f"{game_data.scene_name}.pkl")
    if use_cache:
        if os.path.exists(room_cache_path):
            try:
                with open(room_cache_path, "rb") as file:
                    room = pickle.load(file)
                return room
            except:
                pass
    if terrain is None:
        return None
    
    # Add room transitions as fake solids so it is always closed
    transitions = game_data.get_transitions()
    for transition in transitions:
        x, y, w, h = transition["x"], transition["y"], transition["w"], transition["h"]
        x_min, y_min = (x - w, y - h)
        x_max, y_max = (x + w, y + h)
        path = [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
            (x_min, y_min),
        ]
        terrain.append(path)

    np_terrain = [np.array(path) for path in terrain]

    # Dilate the walls so that they extend by the knight's bounding box extent, so the knight can be modeled as one point
    def scale_vertical(arr, factor):
        new_arr = arr.copy()
        new_arr[:, 1] *= factor
        return new_arr

    knight_width = 0.25
    knight_height = 0.641

    squashed = [scale_vertical(arr, knight_width / knight_height) for arr in np_terrain]
    offset = pyclipr.ClipperOffset()
    offset.scaleFactor = int(1000)
    offset.addPaths(squashed, pyclipr.JoinType.Square, pyclipr.EndType.Polygon)
    dilated = offset.execute(knight_width)
    # filter dilation artifacts. I don't know what the deal is with these, but it creates small boxes at seemingly arbitrary positions
    # no_artifacts = [arr for arr in dilated if (arr.max(axis=0) - arr.min(axis=0)).max() > knight_width]
    unsquashed = [scale_vertical(arr, knight_height / knight_width) for arr in dilated]

    # process into paths
    pc = pyclipr.Clipper()
    pc.scaleFactor = int(1000)
    pc.addPaths(unsquashed, pyclipr.Subject)

    # Clip an outer boundary
    max_coord = 1000
    outer_bounds = np.array(
        [
            (-max_coord, -max_coord),
            (-max_coord, max_coord),
            (max_coord, max_coord),
            (max_coord, -max_coord),
            (-max_coord, -max_coord),
        ]
    )
    pc.addPath(outer_bounds, pyclipr.Clip)

    # Intersect overlapping regions into one.
    clip_tree = pc.execute2(pyclipr.Intersection, pyclipr.FillRule.NonZero)

    # Extract the polygons and whether they are holes
    def recurse_clip_tree(clip_tree, out_list):
        for child in clip_tree.children:
            n = len(child.polygon)
            if n == 0:
                continue
            poly = np.array(child.polygon)
            out_list.append((poly, child.isHole))
            recurse_clip_tree(child, out_list)

    polygons = []
    recurse_clip_tree(clip_tree, polygons)

    terrain = Terrain(polygons)

    with open(room_cache_path, 'wb') as file:
        pickle.dump(terrain, file, pickle.HIGHEST_PROTOCOL)

    return terrain
