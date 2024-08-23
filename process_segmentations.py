from collections import deque
import cv2
import numpy as np
import pyclipr
from itertools import chain, cycle

from region import Region
from room import Room
from visualize import ScreenTransform, interactive_room

# input: clipper-formatted segmentations
from segmentations.Crossroads_03 import terrain

def process_segmentations(terrain):
    np_terrain = [np.array(path) for path in terrain]

    def scale_vertical(arr, factor):
        new_arr = arr.copy()
        new_arr[:,1] *= factor
        return new_arr

    knight_width = 0.25
    knight_height = 0.641

    squashed = [scale_vertical(arr, knight_width / knight_height) for arr in np_terrain]
    offset = pyclipr.ClipperOffset()
    offset.scaleFactor = int(1000)
    offset.addPaths(squashed, pyclipr.JoinType.Square, pyclipr.EndType.Polygon)
    dilated = offset.execute(knight_width)
    # filter dilation artifacts. I don't know what the deal is with these, but it creates small boxes at seemingly arbitrary positions
    no_artifacts = [arr for arr in dilated if (arr.max(axis=0) - arr.min(axis=0)).max() > knight_width]
    unsquashed = [scale_vertical(arr, knight_height / knight_width) for arr in no_artifacts]


    # process into paths
    pc = pyclipr.Clipper()
    pc.scaleFactor = int(1000)
    pc.addPaths(unsquashed, pyclipr.Subject)
    # offset = np.array([.6,.8])
    # pc.addPaths([path + offset for path in np_terrain], pyclipr.Subject)
    # pc.addPaths([path - offset for path in np_terrain], pyclipr.Subject)
    # offset[1] *= -1
    # pc.addPaths([path + offset for path in np_terrain], pyclipr.Subject)
    # pc.addPaths([path - offset for path in np_terrain], pyclipr.Subject)

    max_coord = 1000
    outer_bounds = np.array([(-max_coord, -max_coord), (-max_coord, max_coord), (max_coord, max_coord), 
        (max_coord, -max_coord), (-max_coord, -max_coord)])
    pc.addPath(outer_bounds, pyclipr.Clip)

    # clip_tree = pc.execute2(pyclipr.Union, pyclipr.FillRule.NonZero)
    clip_tree = pc.execute2(pyclipr.Intersection, pyclipr.FillRule.NonZero)

    # correspond each segment to an open or close

    def recurse_clip_tree(clip_tree, out_list):
        for child in clip_tree.children:
            n = len(child.polygon)
            if n == 0:
                continue
            poly = cycle(child.polygon)
            p0 = next(poly)
            for _ in range(n):
                p1 = next(poly)
                if p0[0] <= p1[0]:
                    out_list.append([p0[0], p0[1], p1[0], p1[1], child.isHole, p0[0] == p1[0]])
                else:
                    out_list.append([p1[0], p1[1], p0[0], p0[1], child.isHole, p0[0] == p1[0]])
                p0 = p1
            recurse_clip_tree(child, out_list)

    segments = []
    recurse_clip_tree(clip_tree, segments)

    # print(np_terrain, len(terrain))
    # print(segments, len(segments))

    def vertically_intersect(segment, x):
        if segment[5]: # vertical
            return min(segment[1], segment[3])
        m = (segment[3] - segment[1]) / (segment[2] - segment[0])
        b = (segment[1] - m * segment[0])
        return m * x + b

    floor_segments = []
    ceil_segments = []
    vert_segments = []
    for segment in segments:
        if segment[5]: # is vertical
            vert_segments.append(segment)
            continue
        sx = segment[0]
        sy = segment[1]
        count = 0
        for other in segments:
            if other == segment or other[5]:
                continue

            if other[0] <= sx and sx < other[2] and sy < vertically_intersect(other, sx):
                count += 1

        if count % 2 == 0:
            floor_segments.append(segment)
        else:
            ceil_segments.append(segment)

    # flatten slanted angles (break apart into pieces)

    def piecewise_flatten(segment, max_or_min, max_length):
        if segment[1] == segment[3]:
            return [segment]
        extreme_y = max_or_min(segment[1], segment[3])
        sample_x1 = extreme_y == segment[1]
        # term_x = segment[2] if extreme_y == segment[1] else segment[0]
        # delta_x = max_length if extreme_y == segment[1] else -max_length
        curr_x = segment[0]
        term_x = segment[2]

        partial_segments = []
        while curr_x < term_x:
            x1 = curr_x
            x2 = min(curr_x + max_length, term_x)
            sample_x = x1 if sample_x1 else x2
            y = vertically_intersect(segment, sample_x)
            partial_segments.append((x1, y, x2, y, False, x1 == x2))

            curr_x += max_length
        return partial_segments

    def round_segment(seg):
        return tuple([*[round(x,3) for x in seg[:4]], *seg[4:]])

    floor_segments = list(chain(*[piecewise_flatten(seg, max, 4) for seg in floor_segments]))
    # ceil_segments = list(chain(*[piecewise_flatten(seg, min, 4) for seg in ceil_segments]))

    floor_segments = list(map(round_segment, floor_segments))
    ceil_segments = list(map(round_segment, ceil_segments))

    # join collinear

    endpoint_dict = dict()
    for seg in floor_segments.copy():
        p0 = tuple(seg[:2])
        p1 = tuple(seg[2:4])
        seg_match = endpoint_dict.get(p0, None) or endpoint_dict.get(p1, None)
        if seg_match is not None:
            new_seg = (
                min(seg[0], seg_match[0]),
                seg[1],
                max(seg[2], seg_match[2]),
                seg[1],
                seg[4],
                seg[5]
            )
            floor_segments.remove(seg)
            floor_segments.remove(seg_match)
            floor_segments.append(new_seg)
            seg = new_seg
        endpoint_dict[p0] = seg
        endpoint_dict[p1] = seg

    # print('floor', floor_segments, len(floor_segments))
    # print('ceil', ceil_segments, len(ceil_segments))

    # above floor regions

    floor_segments.sort(key=lambda x: x[1], reverse=True) # highest floor first
    max_height = 1005
    cut_points = [(-np.inf, max_height, max_height), (np.inf, max_height, max_height)]
    floor_regions = []
    for segment in floor_segments:
        x_min = segment[0]
        x_max = segment[2]
        floor_y = segment[1]
        prev_cut = cut_points[0]
        to_remove = []
        left_y = floor_y
        right_y = floor_y
        for cut in cut_points[1:]:
            if cut[0] <= x_min:
                if cut[0] == x_min:
                    left_y = cut[1]
                    to_remove.append(cut)
                prev_cut = cut
                continue
            if cut[0] >= x_max:
                if cut[0] == x_max:
                    right_y = cut[2]
                    to_remove.append(cut)
                region = Region(
                    max(x_min, prev_cut[0]), # lower x
                    floor_y, # lower y
                    min(x_max, cut[0]),
                    min(prev_cut[2], cut[1]),
                    Region.ALL_ABOVE
                )
                if left_y == floor_y and region.x_min == x_min:
                    left_y = region.y_max
                if right_y == floor_y and region.x_max == x_max:
                    right_y = region.y_max
                floor_regions.append(region)
                break
            region = Region(
                max(x_min, prev_cut[0]), # lower x
                floor_y, # lower y
                min(x_max, cut[0]),
                min(prev_cut[2], cut[1]),
                Region.ALL_ABOVE
            )
            if region.x_min == x_min:
                left_y = region.y_max
            prev_cut = cut
            region.set_over_floor(segment)
            floor_regions.append(region)
            to_remove.append(cut)
        for rem_cut in to_remove:
            cut_points.remove(rem_cut)
        cut_points.append((x_min, left_y, floor_y))
        cut_points.append((x_max, floor_y, right_y))
        cut_points.sort(key=lambda cut: cut[0])

    def get_subregions(floor_region):
        subregions = floor_region.subregions_from_ceilings(ceil_segments)
        floor_region.overlapping_regions.update(subregions)
        return subregions

    empty_regions = list(chain(*[get_subregions(region) for region in floor_regions]))
    for region in empty_regions:
        region.connect_to_adjacent_regions(empty_regions)
    room = Room([*floor_regions, *empty_regions], segments, floor_segments, ceil_segments, vert_segments)
    return room

# room = process_segmentations(terrain)
# interactive_room(room)