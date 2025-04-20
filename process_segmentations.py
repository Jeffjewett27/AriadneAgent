import os
import numpy as np
import pyclipr
from itertools import chain, cycle
import pickle

import pyclipr.pyclipr

from floor import Floor
from region import Region
from room import Room

def process_segmentations(game_data, use_cache=True):
    terrain = game_data.terrain_segmentation

    room_cache_path = os.path.join('rooms', f'{game_data.scene_name}.pkl')
    if use_cache:
        if os.path.exists(room_cache_path):
            try:
                with open(room_cache_path, 'rb') as file:
                    room = pickle.load(file)
                return room
            except:
                pass
    if terrain is None:
        return None
        
    transitions = game_data.get_transitions()
    for transition in transitions:
        x, y, w, h = transition['x'], transition['y'], transition['w'], transition['h']
        x_min, y_min = (x - w, y - h)
        x_max, y_max = (x + w, y + h)
        path = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)]
        terrain.append(path)

    np_terrain = [np.array(path) for path in terrain]
    # np_terrain = [np_terrain[4]]

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
    # no_artifacts = [arr for arr in dilated if (arr.max(axis=0) - arr.min(axis=0)).max() > knight_width]
    unsquashed = [scale_vertical(arr, knight_height / knight_width) for arr in dilated]


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
    for floor in segments:
        if floor[5]: # is vertical
            vert_segments.append(floor)
            continue
        sx = floor[0]
        sy = floor[1]
        count = 0
        for other in segments:
            if other == floor or other[5]:
                continue

            if other[0] <= sx and sx < other[2] and sy < vertically_intersect(other, sx):
                count += 1

        if count % 2 == 0:
            floor_segments.append(floor)
        else:
            ceil_segments.append(floor)

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
    floors = [Floor(*floor[:3]) for floor in floor_segments]
    max_height = 1005
    cut_points = [(-np.inf, max_height, max_height), (np.inf, max_height, max_height)]
    floor_regions = []
    for floor in floors:
        x_min = floor.x_min
        x_max = floor.x_max
        floor_y = floor.y_level
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
                region.set_over_floor(floor)
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
            region.set_over_floor(floor)
            floor_regions.append(region)
            to_remove.append(cut)
        for rem_cut in to_remove:
            cut_points.remove(rem_cut)
        cut_points.append((x_min, left_y, floor_y))
        cut_points.append((x_max, floor_y, right_y))
        cut_points.sort(key=lambda cut: cut[0])

    def get_subregions(floor_region):
        subregions = floor_region.subregions_from_ceilings(ceil_segments)
        for subregion in subregions:
            subregion.set_over_floor(floor_region.over_floor)
        floor_region.overlapping_regions.update(subregions)
        return subregions

    empty_regions = list(chain(*[get_subregions(region) for region in floor_regions]))
    for region in empty_regions:
        region.connect_to_adjacent_regions(empty_regions)


    room = Room([*floor_regions, *empty_regions], segments, floors, ceil_segments, vert_segments)
    for floor in floors:
        floor.connect_to_other_floors(floors, room)

    with open(room_cache_path, 'wb') as file:
        pickle.dump(room, file, pickle.HIGHEST_PROTOCOL)

    return room

# room = process_segmentations(terrain)
# interactive_room(room)

if __name__ == "__main__":
    terrain = [[[910.17, -131.22], [910.0, -131.22], [910.0, 664.63], [908.2, 664.63], [908.2, 715.59], [-610.0, 715.59], [-610.0, 637.42], [-610.0, -135.37], [-610.0, -209.38], [910.17, -209.38], [910.17, -131.22]], [[-590.0, 637.42], [890.0, 637.42], [890.0, -131.22], [-590.0, -131.22], [-590.0, 637.42]], [[3.0, 16.0], [5.0, 16.0], [5.0, 27.0], [6.0, 27.0], [6.0, 30.0], [0.0, 30.0], [0.0, 15.6], [-2.38, 16.34], [-0.51, 13.29], [0.0, 13.17], [0.0, 13.0], [0.0, 12.0], [1.75, 12.0], [1.9, 11.59], [2.0, 11.59], [2.0, 11.0], [3.0, 11.0], [3.0, 16.0]], [[30.0, 72.0], [16.0, 72.0], [16.0, 71.0], [16.0, 70.0], [21.61, 70.0], [22.22, 69.83], [22.77, 69.38], [22.94, 68.53], [22.9, 67.06], [22.98, 66.33], [23.8, 66.33], [24.05, 66.76], [24.85, 66.35], [25.23, 65.63], [24.7, 67.0], [26.0, 67.0], [26.0, 64.0], [26.0, 63.0], [26.0, 62.0], [30.0, 62.0], [30.0, 64.0], [30.0, 72.0]], [[3.0, 65.99], [3.28, 66.37], [3.93, 66.52], [4.71, 66.28], [4.89, 67.0], [5.0, 67.0], [5.0, 68.73], [5.14, 69.08], [5.06, 69.24], [5.46, 69.74], [6.11, 70.0], [12.0, 70.0], [12.0, 72.0], [0.0, 72.0], [0.0, 65.0], [0.0, 64.0], [0.0, 37.99], [-0.66, 37.34], [-0.19, 35.64], [0.67, 35.89], [0.75, 36.0], [1.0, 36.0], [1.0, 36.32], [1.52, 37.0], [3.21, 37.0], [3.54, 36.96], [4.12, 35.5], [5.03, 35.5], [5.33, 36.0], [19.0, 36.0], [19.0, 38.0], [8.0, 38.0], [8.0, 40.0], [2.0, 40.0], [2.0, 47.0], [5.0, 47.0], [5.0, 46.0], [6.0, 46.0], [6.0, 44.0], [13.0, 44.0], [13.0, 43.0], [14.0, 43.0], [14.0, 42.0], [18.0, 42.0], [18.0, 46.0], [8.0, 46.0], [8.0, 51.0], [6.0, 51.0], [6.0, 56.0], [7.0, 56.0], [7.0, 59.0], [3.0, 59.0], [3.0, 64.0], [3.0, 65.99]], [[16.141, 56.031], [17.99, 56.031], [17.99, 54.202], [16.141, 54.202], [16.141, 56.031]], [[8.865, 66.558], [19.115, 66.558], [19.115, 64.282], [8.865, 64.282], [8.865, 66.558]], [[20.918, 63.045], [23.306, 63.045], [23.306, 62.351], [20.918, 62.351], [20.918, 63.045]], [[12.878, 17.045], [15.266, 17.045], [15.266, 16.351], [12.878, 16.351], [12.878, 17.045]], [[12.746, 50.926], [10.083, 50.926], [10.083, 50.256], [12.746, 50.256], [12.746, 50.926]], [[30.0, 33.0], [23.0, 33.0], [23.0, 34.0], [21.0, 34.0], [21.0, 33.0], [21.0, 32.0], [21.0, 31.0], [24.0, 31.0], [24.0, 23.0], [21.0, 23.0], [21.0, 19.0], [19.0, 19.0], [19.0, 17.0], [22.0, 17.0], [22.0, 15.0], [24.0, 15.0], [24.0, 13.0], [25.0, 13.0], [25.0, 6.0], [22.0, 6.0], [22.0, 2.0], [17.0, 2.0], [10.0, 2.0], [10.0, 4.0], [8.0, 4.0], [8.0, 7.0], [0.0, 7.0], [0.0, 1.0], [0.0, 0.0], [9.79, 0.0], [9.79, -0.35], [22.19, -0.35], [22.19, 0.0], [30.0, 0.0], [30.0, 32.0], [30.0, 33.0]], [[12.045, 60.825], [15.894, 60.825], [15.894, 59.935], [12.045, 59.935], [12.045, 60.825]], [[11.0, 10.0], [11.0, 9.0], [20.0, 9.0], [20.0, 13.0], [16.0, 13.0], [16.0, 11.0], [11.0, 11.0], [11.0, 10.0], [11.0, 10.0]], [[9.0, 25.0], [9.0, 24.0], [15.0, 24.0], [15.0, 25.0], [19.0, 25.0], [19.0, 26.0], [21.0, 26.0], [21.0, 28.0], [19.0, 28.0], [19.0, 30.0], [12.0, 30.0], [12.0, 27.0], [9.0, 27.0], [9.0, 25.0], [9.0, 25.0]], [[20.0, 53.0], [20.0, 52.0], [23.0, 52.0], [23.0, 50.0], [25.0, 50.0], [25.0, 48.0], [26.0, 48.0], [26.0, 42.0], [23.0, 42.0], [23.0, 39.0], [25.0, 39.0], [25.0, 37.0], [30.0, 37.0], [30.0, 58.0], [20.0, 58.0], [20.0, 53.0], [20.0, 53.0]]]
    terrain2 = [
        [[0,0], [0,2], [2,2], [2,0], [0,0]]
    ]
    # terrain = [[[12.878, 17.045], [12.878, 16.351], [15.266, 16.351], [15.266, 17.045], [12.878, 17.045]], terrain[2]]
    terrain = [terrain[2], terrain[8]]
    np_terrain = [np.array(path) for path in terrain]

    knight_width = 0.25
    knight_height = 0.641

    offset = np.array([.5,1])
    off1 = [path + offset for path in np_terrain]
    off2 = [path - offset for path in np_terrain]
    # process into paths
    # pc = pyclipr.Clipper()
    # pc.preserveCollinear = False
    # pc.scaleFactor = int(1000)
    # pc.addPaths(np_terrain, pyclipr.Subject)
    # pc.addPaths(off1, pyclipr.Clip)
    # pc.addPaths(off2, pyclipr.Clip)
    # offset[1] *= -1
    # pc.addPaths([path + offset for path in np_terrain], pyclipr.Clip)
    # pc.addPaths([path - offset for path in np_terrain], pyclipr.Clip)
    # result = pc.execute(pyclipr.Union, pyclipr.FillRule.EvenOdd)
    pco = pyclipr.ClipperOffset()
    pco.scaleFactor = int(1000)
    pco.addPaths(np_terrain, pyclipr.JoinType.Square, pyclipr.EndType.Polygon)
    result = pco.execute(0.5)

    pc = pyclipr.Clipper()
    pc.scaleFactor = int(1000)
    pc.addPaths(np_terrain, pyclipr.Subject)
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

    # pc.addPath(outer_bounds, pyclipr.Clip)
    import plotly.graph_objs as go
    import plotly.io as pio

    polygons = []
    for idx, arr in enumerate(result):
        polygons.append(go.Scatter(
            x=arr[:,0],
            y=arr[:,1],
            fill='toself',
            mode='lines+markers',
            name=f'Dilated_{idx}'
        ))

    for idx, arr in enumerate(np_terrain):
        polygons.append(go.Scatter(
            x=arr[:,0],
            y=arr[:,1],
            mode='lines+markers',
            name=f'Original_{idx}'
        ))

    # Create the layout
    layout = go.Layout(
        title="Multiple Disconnected Polygons",
        xaxis=dict(range=[-200, 200]),
        yaxis=dict(range=[-200, 200]),
        showlegend=True
    )

    # Combine all polygons into one figure
    fig = go.Figure(data=polygons, layout=layout)

    # Show the figure
    pio.show(fig)