from collections import deque
import cv2
import numpy as np
import pyclipr
from itertools import chain, cycle

from visualize import ScreenTransform

# input: clipper-formatted segmentations
from segmentations.Fungus1_30 import terrain
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
offset.addPaths(np_terrain, pyclipr.JoinType.Square, pyclipr.EndType.Polygon)
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

# algorithm for clipping a line segment to a box
def liang_barsky(x0, y0, x1, y1, xmin, ymin, xmax, ymax):
    def clip(p, q, u1, u2):
        if p < 0:
            r = q / p
            if r > u2:
                return False, u1, u2
            elif r > u1:
                u1 = r
        elif p > 0:
            r = q / p
            if r < u1:
                return False, u1, u2
            elif r < u2:
                u2 = r
        elif q < 0:
            return False, u1, u2
        return True, u1, u2

    dx = x1 - x0
    dy = y1 - y0
    u1, u2 = 0.0, 1.0

    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]

    for i in range(4):
        is_valid, u1, u2 = clip(p[i], q[i], u1, u2)
        if not is_valid:
            return None

    if u2 < 1:
        x1 = x0 + u2 * dx
        y1 = y0 + u2 * dy
    if u1 > 0:
        x0 = x0 + u1 * dx
        y0 = y0 + u1 * dy

    return (x0, y0, x1, y1)

class Region:
    ALL_ABOVE = 0
    EMPTY = 1

    def __init__(self, x_min, y_min, x_max, y_max, region_type):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.area = (x_max - x_min) * (y_max - y_min)
        self.zone = (x_min, y_min, x_max, y_max)

        self.region_type = region_type
        self.over_floor = None

        self.left_regions = set()
        self.right_regions = set()
        self.up_regions = set()
        self.down_regions = set()
        self.overlapping_regions = set()

    def set_over_floor(self, over_floor):
        self.over_floor = over_floor

    def contains(self, x, y):
        return self.x_min <= x <= self.x_max \
            and self.y_min <= y <= self.y_max
    
    def subregions_from_ceilings(self, ceiling_segments):
        ceiling_segments = [liang_barsky(*segment[:4], *self.zone) for segment in ceiling_segments]
        ceiling_segments = [s for s in ceiling_segments if s is not None]
        ceiling_segments = [s for s in ceiling_segments if s[0] < self.x_max and s[2] > self.x_min]
        ceiling_segments.sort(key=lambda s: s[0])
        print('ceilseg', ceiling_segments)
        
        subregions = []
        # base_region
        min_ceil_y = min([self.y_max, *[min(segment[1], segment[3]) for segment in ceiling_segments]])
        base_region = Region(
            self.x_min,
            self.y_min,
            self.x_max,
            min_ceil_y,
            Region.EMPTY
        )
        if base_region.area > 0:
            subregions.append(base_region)

        # areas that have not been processed yet
        to_resolve = deque([(self.x_min, min_ceil_y, self.x_max, self.y_max)])

        def zone_area(zone):
            return (zone[2] - zone[0]) * (zone[3] - zone[1])

        while len(to_resolve) > 0:
            zone = to_resolve.popleft()
            if zone_area(zone) == 0:
                continue
            segments_in_zone = [liang_barsky(*segment[:4], *zone) for segment in ceiling_segments]
            segments_in_zone = [s for s in segments_in_zone if s is not None]
            segments_in_zone.sort(key=lambda s: s[0])
            if len(segments_in_zone) == 0:
                continue
            max_area = 0
            selected_zone = None
            for idx, s0 in enumerate(segments_in_zone[:-1]):
                min_x = max_x = s0[0]
                max_y = s0[1]
                for s1 in segments_in_zone[idx:]:
                    max_y = min(max_y, s1[1], s1[3])
                    max_x = max(max_x, s1[2])
                    zone_option = (min_x, zone[1], max_x, max_y)
                    area = zone_area(zone_option)
                    if area > max_area:
                        max_area = area
                        selected_zone = zone_option

            if selected_zone is None:
                continue
            new_left_zone = (zone[0], zone[1], selected_zone[0], zone[3])
            new_right_zone = (selected_zone[2], zone[1], zone[2], zone[3])
            new_upper_zone = (selected_zone[0], selected_zone[3], selected_zone[2], zone[3])
            to_resolve.append(new_left_zone)
            to_resolve.append(new_right_zone)
            to_resolve.append(new_upper_zone)
            new_region = Region(
                *selected_zone,
                Region.EMPTY
            )
            new_region.set_over_floor(self.over_floor)
            subregions.append(new_region)
        return subregions
    
    def area_overlap(self, region):

        # Calculate the overlapping region's boundaries
        overlap_xmin = max(self.x_min, region.x_min)
        overlap_ymin = max(self.y_min, region.y_min)
        overlap_xmax = min(self.x_max, region.x_max)
        overlap_ymax = min(self.y_max, region.y_max)

        # Check if there is an overlap
        if overlap_xmin < overlap_xmax and overlap_ymin < overlap_ymax:
            overlap_width = overlap_xmax - overlap_xmin
            overlap_height = overlap_ymax - overlap_ymin
            overlap_area = overlap_width * overlap_height
            return overlap_area
        else:
            return 0
        
    def check_shared_edges(self, region):
        def get_edges(region):
            xmin, ymin, xmax, ymax = region.zone
            return [
                ((xmin, ymin), (xmax, ymin)),  # bottom edge
                ((xmax, ymin), (xmax, ymax)),  # right edge
                ((xmin, ymax), (xmax, ymax)),  # top edge
                ((xmin, ymin), (xmin, ymax))   # left edge
            ]

        def edges_overlap(edge1, edge2):
            (x1, y1), (x2, y2) = edge1
            (x3, y3), (x4, y4) = edge2

            if x1 == x2 == x3 == x4:  # vertical edges
                return y1 < y4 and y2 > y3
            elif y1 == y2 == y3 == y4:  # horizontal edges
                return x1 < x4 and x2 > x3
            return False

        self_edges = get_edges(self)
        region_edges = get_edges(region)
        for i, self_edge in enumerate(self_edges):
            region_edge = region_edges[(i+2)%4] # test left against right, etc
            if edges_overlap(self_edge, region_edge):
                return i
        return None
    
    def connect_to_adjacent_regions(self, regions):
        connect_sets = [
            self.down_regions,
            self.right_regions,
            self.up_regions,
            self.left_regions
        ]
        for region in regions:
            if self.area_overlap(region) > 0:
                self.overlapping_regions.add(region)
                continue
            edge_connect = self.check_shared_edges(region)
            if edge_connect is not None:
                connect_sets[edge_connect].add(region)


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
print('regions', len(floor_regions), floor_regions)

empty_regions = list(chain(*[region.subregions_from_ceilings(ceil_segments) for region in floor_regions]))

# connect regions

# debugging
floor_regions = empty_regions
        
# visualize
cam_bound = 200
cam_pad = 10
x_coords = [x for x in [*[seg[0] for seg in segments], *[seg[2] for seg in segments]] if x > -cam_bound and x < cam_bound]
y_coords = [y for y in [*[seg[1] for seg in segments], *[seg[3] for seg in segments]] if y > -cam_bound and y < cam_bound]
cam_x_min = min(x_coords) - cam_pad
cam_x_max = max(x_coords) + cam_pad
cam_y_min = min(y_coords) - cam_pad
cam_y_max = max(y_coords) + cam_pad
print(cam_x_min, cam_x_max, cam_y_min, cam_y_max)
cam_width, cam_height = (cam_x_max - cam_x_min), (cam_y_max - cam_y_min)
cam_x, cam_y = (cam_x_min + cam_width / 2), (cam_y_min + cam_height / 2)
img_width, img_height = 1280, 960
transform = ScreenTransform(img_width, img_height-2, cam_x, cam_y, cam_width, cam_height)
img = np.zeros((img_height, img_width, 3))

for idx, region in enumerate(floor_regions):
    p0 = transform.transform_xy(region.x_min, region.y_min)
    p1 = transform.transform_xy(region.x_max, region.y_max)
    img2 = img.copy()
    color = tuple([int(x) for x in (np.random.random(3) * 250)])
    cv2.rectangle(img2, p0, p1, color=color, thickness=-1)
    alpha = 0.2
    img = cv2.addWeighted(img, alpha, img2, 1-alpha, 0)

# for idx, region in enumerate(floor_regions):
#     p0 = transform.transform_xy(region.x_min, region.y_min)
#     p1 = transform.transform_xy(region.x_max, region.y_max)
#     cv2.rectangle(img, p0, p1, color=(0,0,0), thickness=3)

for segment in floor_segments:
    p0 = transform.transform_xy(segment[0], segment[1])
    p1 = transform.transform_xy(segment[2], segment[3])
    cv2.line(img, p0, p1, (0, 255, 255))

for segment in ceil_segments:
    p0 = transform.transform_xy(segment[0], segment[1])
    p1 = transform.transform_xy(segment[2], segment[3])
    cv2.line(img, p0, p1, (255, 0, 0))

for segment in vert_segments:
    p0 = transform.transform_xy(segment[0], segment[1])
    p1 = transform.transform_xy(segment[2], segment[3])
    cv2.line(img, p0, p1, (255, 255, 0))


cv2.imwrite('terrain.png', img)


# build kd-tree