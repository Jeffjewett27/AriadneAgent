import heapq

from collections import deque
from constants import KNIGHT_HALF_WIDTH
from region import Region

class Room:

    def __init__(self, regions, raw_segments, floors, ceil_segments, wall_segments):
        # larger boxes that are between floors, ignoring walls
        self.coarse_regions = set()
        # areas the knight can be. each is a subset of a coarse region
        self.air_regions = set()
        for region in regions:
            if region.region_type == Region.ALL_ABOVE:
                self.coarse_regions.add(region)
            elif region.region_type == Region.EMPTY:
                self.air_regions.add(region)

        self.raw_segments = raw_segments
        self.floors = floors
        self.ceil_segments = ceil_segments
        self.wall_segments = wall_segments
        
    def locate_coarse(self, x, y) -> Region:
        for region in self.coarse_regions:
            if region.contains(x, y):
                return region
        return None
            
    def locate_subregion(self, region, x, y) -> Region:
        for subregion in region.overlapping_regions:
            if subregion.contains(x, y):
                return subregion
        return None
            
    def locate_air(self, x, y) -> Region:
        coarse = self.locate_coarse(x, y)
        if coarse is None:
            return None
        return self.locate_subregion(coarse, x, y)
    
    def is_spot_safe(self, x, y):
        air = self.locate_air(x, y)
        return air != None
    
    def get_over_floor(self, x, y):
        region = self.locate_coarse(x, y)
        if region is None:
            return None
        return region.over_floor
    
    def is_grounded(self, x, y):
        def check_ground(x, y):
            floor = self.get_over_floor(x, y)
            if floor is None:
                return False
            return floor.x_min <= x <= floor.x_max and abs(y - floor.y_level) < 0.03
        return check_ground(x-KNIGHT_HALF_WIDTH, y) or check_ground(x+KNIGHT_HALF_WIDTH, y)
    
    def get_grounded_floor(self, x, y):
        def check_ground(x, y):
            floor = self.get_over_floor(x, y)
            if floor is None:
                return None
            if floor.x_min <= x <= floor.x_max and abs(y - floor.y_level) < 0.03:
                return floor
        left_floor = check_ground(x-KNIGHT_HALF_WIDTH, y) 
        right_floor = check_ground(x+KNIGHT_HALF_WIDTH, y)
        if left_floor is not None and right_floor is not None:
            if left_floor.y_level > right_floor.y_level:
                return left_floor
            else:
                return right_floor
        return left_floor or right_floor
    
    def region_raycast_generator(self, x1, y1, x2, y2):
        region = self.locate_air(x=x1, y=y1)
        if region is None:
            yield "failed", x1, y1, None
            return
        target = (x2, y2)
        curr_x, curr_y = x1, y1
        while True:
            neighbor, new_x, new_y, _ = region.ray_collide_neighbor((curr_x, curr_y), target)
            if (new_x, new_y) == (target):
                # reached end of ray segment
                yield "reached", new_x, new_y, region
                return
            elif neighbor is None:
                # reached wall
                yield "failed", new_x, new_y, region
                return
            yield "neighbor", new_x, new_y, neighbor
            curr_x, curr_y = new_x, new_y
            region = neighbor
        
    def region_raycast(self, x1, y1, x2, y2):
        raygen = self.region_raycast_generator(x1, y1, x2, y2)
        for cast_status, final_x, final_y, final_region in raygen:
            pass
        return cast_status, final_x, final_y, final_region
    
    def region_pathcast_generator(self, path):
        assert len(path) >= 2
        for p1, p2 in zip(path[:-1], path[1:]):
            raygen = self.region_raycast_generator(*p1, *p2)
            for cast_status, final_x, final_y, final_region in raygen:
                yield cast_status, final_x, final_y, final_region
                if cast_status == "failed":
                    return
                
    def pathfind_floors(self, from_pos, to_pos):
        from_coarse = self.locate_coarse(*from_pos)
        if from_coarse is None:
            return []
        from_floor = from_coarse.over_floor
        to_coarse = self.locate_coarse(*to_pos)
        if to_coarse is None:
            return []
        to_floor = to_coarse.over_floor
        if from_floor is None or to_floor is None:
            return []
        
        if to_floor == from_floor:
            return [to_floor]
        
        visited_floors = set()
        visited_edges = {}
        # to_visit = deque([from_floor])
        to_visit = [(0, from_floor)]

        while len(to_visit) > 0:
            distance, cur_floor = heapq.heappop(to_visit)
            if cur_floor in visited_floors:
                continue
            # cur_floor = to_visit.popleft()
            visited_floors.add(cur_floor)
            for neighbor in cur_floor.floor_neighbors:
                if neighbor == to_floor:
                    floor = neighbor
                    prev_floor = cur_floor
                    path = [neighbor, prev_floor]
                    while prev_floor != from_floor:
                        floor, prev_floor = prev_floor, visited_edges[prev_floor]
                        path.append(prev_floor)
                    path.reverse()
                    return path
                if neighbor in visited_floors:
                    continue
                # to_visit.append(neighbor)
                heapq.heappush(to_visit, (distance+1, neighbor))
                if neighbor not in visited_edges:
                    visited_edges[neighbor] = cur_floor
        return []
        
                
def generator_last(generator):
    for out in generator:
        pass
    return out