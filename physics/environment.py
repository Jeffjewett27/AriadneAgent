from dataclasses import dataclass
import numpy as np
from shapely import GeometryCollection, MultiLineString, MultiPoint
from shapely.geometry import LineString, box, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.strtree import STRtree


@dataclass
class Segment:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    normal: np.ndarray
    line_string: LineString
    type: str
    polygon: Polygon

    def __hash__(self):
        return hash((self.x_min, self.x_max, self.y_min, self.y_max, self.type))


class Terrain:
    def __init__(self, polygons: list[tuple[np.ndarray, bool]], epsilon: float = 1e-6):
        self.polygons = []
        self.segments: list[Segment] = []

        for p_idx, (coords, isHole) in enumerate(polygons):
            # Ensure closure: last point == first point
            if not np.allclose(coords[0], coords[-1], atol=epsilon):
                coords = np.vstack([coords, coords[0]])

            poly = Polygon(coords)
            self.polygons.append(poly)
            poly_coords = np.array(poly.exterior.coords)

            # Number of segments = number of points - 1 (closed loop)
            for i in range(len(poly_coords) - 1):
                x1, y1 = poly_coords[i]
                x2, y2 = poly_coords[i + 1]

                dx = x2 - x1
                dy = y2 - y1

                if isHole:
                    # Inward normal (CCW rotation)
                    nx, ny = -dy, dx
                else:
                    # Outward normal (CW rotation)
                    nx, ny = dy, -dx

                # Normalize
                length = np.linalg.norm([nx, ny])
                if length > epsilon:
                    nx /= length
                    ny /= length

                # Classify segment by normal direction
                if ny < -epsilon:
                    seg_type = "ceiling"
                elif abs(nx) < epsilon and ny > epsilon:
                    seg_type = "floor"
                elif nx > epsilon:
                    seg_type = "left_wall"
                else:
                    seg_type = "right_wall"

                segment_info = Segment(
                    x1,
                    y1,
                    x2,
                    y2,
                    np.array([nx, ny]),
                    LineString(((x1, y1), (x2, y2))),
                    seg_type,
                    poly,
                )

                self.segments.append(segment_info)

        line_strings = [seg.line_string for seg in self.segments]
        self.str_tree = STRtree(line_strings)

    def raycast(
        self,
        from_x: float,
        from_y: float,
        theta: float,
        max_dist: float,
        epsilon: float = 1e-6,
    ):
        """
        Casts a ray from (from_x, from_y) at angle `theta` (radians) up to `max_dist`.
        Returns a tuple (x, y, info_list) for the closest collision point, collecting
        metadata for all segments colliding at that point. Returns None if no hit.
        """
        end_x = from_x + np.cos(theta) * max_dist
        end_y = from_y + np.sin(theta) * max_dist
        self.linecast(from_x, from_y, end_x, end_y, epsilon)

    def linecast(
        self,
        from_x: float,
        from_y: float,
        to_x: float,
        to_y: float,
        epsilon: float = 1e-6,
    ):
        """
        Casts a ray from (from_x, from_y) to (to_x, to_y).
        Returns a tuple (x, y, info_list) for the closest collision point, collecting
        metadata for all segments colliding at that point. Returns None if no hit.
        """
        dx = to_x - from_x
        dy = to_y - from_y
        ray = LineString([(from_x, from_y), (to_x, to_y)])

        candidate_idxs = self.str_tree.query(ray)

        nearest_dist = float("inf")
        hit_point = None
        hit_infos = []

        for idx in candidate_idxs:
            seg = self.segments[idx].line_string
            if np.dot([dx, dy], self.segments[idx].normal) >= 0:
                # one-way collision. its only a hit if you oppose the normal
                continue
            inter = ray.intersection(seg)
            if inter.is_empty:
                continue

            pts = []
            if isinstance(inter, Point):
                pts = [inter]
            else:
                pts = [Point(x, y) for x, y in inter.coords]

            for pt in pts:
                dx = pt.x - from_x
                dy = pt.y - from_y
                dist = np.linalg.norm([dx, dy])
                if dist + epsilon < nearest_dist:
                    nearest_dist = dist
                    hit_point = pt
                    hit_infos = [self.segments[idx]]
                elif (
                    hit_point
                    and abs(dist - nearest_dist) <= epsilon
                    and pt.equals(hit_point)
                ):
                    hit_infos.append(self.segments[idx])

        if hit_point is not None:
            return True, hit_point.x, hit_point.y, hit_infos

        return False, None, None, None

    def integrate_motion(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        dt: float,
        epsilon: float = 1e-6,
    ):
        """
        Integrate a point mass starting at pos with velocity vel for duration dt.

        Simulate friction-free inelastic collisions.

        Returns a tuple (hit, new_pos, new_vel, hit_infos)
        - hit (bool): whether a collision was made
        - new_pos (np.ndarray): new xy coordinates
        - new_vel (np.ndarray): new velocity after collisions
        - hit_infos (list[Segment]): details of the final contacted segments
        """
        speed = np.linalg.norm(vel)
        if dt <= epsilon or speed <= epsilon:
            return False, pos, vel, []

        pos_next = pos + vel * dt

        # Broad phase AABB collision
        ray = LineString([pos, pos_next])
        candidate_ids = self.str_tree.query(ray)

        nearest_dist = float("inf")
        hit_point = None
        hit_segments: list[Segment] = []

        # Refine the collision to the closest point
        for idx in candidate_ids:
            seg_line = self.segments[idx].line_string
            if np.dot(vel, self.segments[idx].normal) >= 0:
                # one-way collision. its only a hit if you oppose the normal
                continue
            inter = ray.intersection(seg_line)
            if inter.is_empty:
                continue

            pts = []
            if isinstance(inter, Point):
                pts = [inter]
            elif isinstance(inter, (MultiPoint, MultiLineString, GeometryCollection)):
                # iterate each sub‐geometry
                for geom in inter.geoms:
                    if isinstance(geom, Point):
                        pts.append(geom)
                    else:
                        # e.g. LineString
                        for x, y in geom.coords:
                            pts.append(Point(x, y))
            else:
                pts = [Point(x, y) for x, y in inter.coords]

            for pt in pts:
                dx = pt.x - pos[0]
                dy = pt.y - pos[1]
                dist = np.linalg.norm([dx, dy])
                if dist + epsilon < nearest_dist:
                    nearest_dist = dist
                    hit_point = pt
                    hit_segments = [self.segments[idx]]
                elif (
                    hit_point
                    and abs(dist - nearest_dist) <= epsilon
                    and pt.equals(hit_point)
                ):
                    hit_segments.append(self.segments[idx])

        orig_vel_mag = np.linalg.norm(vel)
        # Augment the motion to slide along colliding surfaces
        if hit_point is not None:
            hit_point = np.array([hit_point.x, hit_point.y])
            t_frac = max(0, min(nearest_dist / (np.linalg.norm(vel) * dt), 1))
            t_left = dt * (1 - t_frac)

            for hit_info in hit_segments:
                v_dot = np.dot(vel, hit_info.normal)
                vel = vel - hit_info.normal * v_dot

            if (
                epsilon < t_left
                and np.linalg.norm(vel) > epsilon
                and (t_left < dt or np.linalg.norm(vel) < orig_vel_mag)
            ):
                # progress needs to be made either on distance or velocity
                return self.integrate_motion(hit_point, vel, t_left)

            return True, hit_point, vel, hit_segments

        return False, pos_next, vel, []

    def find_touching_segments(
        self, pos: np.ndarray, epsilon: float = 1e-6
    ) -> list[Segment]:
        """
        Returns a list of all segments whose distance
        to the point `pos` is <= epsilon.
        """
        pt = Point(pos[0], pos[1])
        candidate_idxs = self.str_tree.query(pt, predicate="dwithin", distance=epsilon)

        touching = []
        for idx in candidate_idxs:
            seg = self.segments[idx]
            line = seg.line_string

            if line.distance(pt) <= epsilon:
                touching.append(seg)

        return touching
