from region import Region


class Room:

    def __init__(self, regions, raw_segments, floor_segments, ceil_segments, wall_segments):
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
        self.floor_segments = floor_segments
        self.ceil_segments = ceil_segments
        self.wall_segments = wall_segments
        
    def locate_coarse(self, x, y):
        for region in self.coarse_regions:
            if region.contains(x, y):
                return region
        return None
            
    def locate_subregion(self, region, x, y):
        for subregion in region.overlapping_regions:
            if subregion.contains(x, y):
                return subregion
        return None
            
    def locate_air(self, x, y):
        coarse = self.locate_coarse(x, y)
        if coarse is None:
            return None
        return self.locate_subregion(coarse, x, y)
    
    def is_spot_safe(self, x, y):
        air = self.locate_air(x, y)
        return air != None