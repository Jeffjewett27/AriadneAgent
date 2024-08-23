from collections import deque


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
        self.edges = [
            ((x_min, y_min), (x_max, y_min)),  # bottom edge
            ((x_max, y_min), (x_max, y_max)),  # right edge
            ((x_min, y_max), (x_max, y_max)),  # top edge
            ((x_min, y_min), (x_min, y_max))   # left edge
        ]

        self.region_type = region_type
        self.over_floor = None

        self.bordering_regions = [
            set(),
            set(),
            set(),
            set()
        ]
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

        def edges_overlap(edge1, edge2):
            (x1, y1), (x2, y2) = edge1
            (x3, y3), (x4, y4) = edge2

            if x1 == x2 == x3 == x4:  # vertical edges
                oy1 = max(y1, y3)
                oy2 = min(y2, y4)
                return oy1 < oy2, (oy1, oy2)
            elif y1 == y2 == y3 == y4:  # horizontal edges
                ox1 = max(x1, x3)
                ox2 = min(x2, x4)
                return ox1 < ox2, (ox1, ox2)
            return False, None

        self_edges = self.edges
        region_edges = region.edges
        for i, self_edge in enumerate(self_edges):
            region_edge = region_edges[(i+2)%4] # test left against right, etc
            is_overlap, bounds = edges_overlap(self_edge, region_edge)
            if is_overlap:
                return i, bounds
        return None, None
    
    def ray_to_edge(self, px, py, dx, dy):
        vert_idx = 1 if dx > 0 else 3
        vert_edge = self.edges[vert_idx]
        hor_idx = 2 if dy > 0 else 0
        hor_edge = self.edges[hor_idx]
        border_x = vert_edge[0][0]
        border_y = hor_edge[0][1]
        if dx == 0:
            # vertical ray intersects hor
            return (hor_idx, px, border_y)
        if dy == 0:
            # horizontal ray intersects vert
            return (vert_idx, border_x, py)
        slope = dy / dx
        vert_intercept = (border_x - px) * slope + py
        if self.y_min <= vert_intercept <= self.y_max:
            # intercepts vertical
            return (vert_idx, border_x, vert_intercept)
        hor_intercept = (border_y - py) / slope + px
        return (hor_idx, hor_intercept, border_y)
    
    def connect_to_adjacent_regions(self, regions):
        for region in regions:
            if self.area_overlap(region) > 0:
                self.overlapping_regions.add(region)
                continue
            connecting_edge, bounds = self.check_shared_edges(region)
            if connecting_edge is not None:
                self.bordering_regions[connecting_edge].add((region, *bounds))

    def check_line_safe(self, point1, point2):
        if not self.contains(*point1):
            return None, self, point1
        if self.contains(*point2):
            return True, self, point2
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        edge, ex, ey = self.ray_to_edge(*point1, dx, dy)
        bound_coord = ex if (edge % 2) == 0 else ey
        for border_region, min_bound, max_bound in self.bordering_regions[edge]:
            if min_bound <= bound_coord <= max_bound and border_region.region_type == Region.EMPTY:
                is_safe, final_region, final_point = border_region.check_line_safe((ex, ey), point2)
                if is_safe is not None:
                    return is_safe, final_region, final_point
        return False, self, (ex, ey)
    
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