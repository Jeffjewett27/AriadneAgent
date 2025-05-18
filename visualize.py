from typing import Tuple
import cv2
import numpy as np
import pygame

from physics.environment import Terrain
from room import Room


class ScreenTransform:
    """
    Transforms from world space into camera space.
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        cam_x: float,
        cam_y: float,
        cam_width: float,
        cam_height: float,
    ):
        self.screen_width: int = screen_width
        self.screen_height: int = screen_height
        self.cam_x: float = cam_x
        self.cam_y: float = cam_y
        self.cam_height: float = cam_height
        self.cam_width: float = cam_width

        self.w_factor: float = screen_width / cam_width
        self.h_factor: float = screen_height / cam_height

    def transform_xy(self, world_X: float, world_y: float) -> Tuple[int, int]:
        """Transform world coordinates to screen space"""
        return int((world_X - self.cam_x) * self.w_factor + self.screen_width / 2), int(
            (self.cam_y - world_y) * self.h_factor + self.screen_height / 2
        )

    def inv_transform_xy(self, cam_x: int, cam_y: int) -> Tuple[float, float]:
        return (
            (cam_x - self.screen_width / 2) / self.w_factor + self.cam_x,
            (cam_y - self.screen_height / 2) / self.h_factor + self.cam_y,
        )


def get_screen_info(
    room: Room, screen_size, cam_pad=10, min_x=-200, min_y=-200, max_x=200, max_y=200
):
    cam_pad = 10
    x_coords = [
        x
        for x in [
            *[seg[0] for seg in room.raw_segments],
            *[seg[2] for seg in room.raw_segments],
        ]
        if x >= min_x and x <= max_x
    ]
    y_coords = [
        y
        for y in [
            *[seg[1] for seg in room.raw_segments],
            *[seg[3] for seg in room.raw_segments],
        ]
        if y >= min_y and y <= max_y
    ]
    cam_x_min = min(x_coords) - cam_pad
    cam_x_max = max(x_coords) + cam_pad
    cam_y_min = min(y_coords) - cam_pad
    cam_y_max = max(y_coords) + cam_pad
    cam_width, cam_height = (cam_x_max - cam_x_min), (cam_y_max - cam_y_min)
    cam_x, cam_y = (cam_x_min + cam_width / 2), (cam_y_min + cam_height / 2)
    img_width, img_height = screen_size
    transform = ScreenTransform(
        img_width, img_height - 2, cam_x, cam_y, cam_width, cam_height
    )
    img = np.zeros((img_height, img_width, 3))
    return transform, img

def get_screen_info2(
    terrain, screen_size, cam_pad=10, min_x=-200, min_y=-200, max_x=200, max_y=200
):
    cam_pad = 10
    x_coords = [
        x
        for x in [
            *[seg.x_min for seg in terrain.segments],
            *[seg.x_max for seg in terrain.segments],
        ]
        if x >= min_x and x <= max_x
    ]
    y_coords = [
        y
        for y in [
            *[seg.y_min for seg in terrain.segments],
            *[seg.y_max for seg in terrain.segments],
        ]
        if y >= min_y and y <= max_y
    ]
    cam_x_min = min(x_coords) - cam_pad
    cam_x_max = max(x_coords) + cam_pad
    cam_y_min = min(y_coords) - cam_pad
    cam_y_max = max(y_coords) + cam_pad
    cam_width, cam_height = (cam_x_max - cam_x_min), (cam_y_max - cam_y_min)
    cam_x, cam_y = (cam_x_min + cam_width / 2), (cam_y_min + cam_height / 2)
    img_width, img_height = screen_size
    transform = ScreenTransform(
        img_width, img_height - 2, cam_x, cam_y, cam_width, cam_height
    )
    img = np.zeros((img_height, img_width, 3))
    return transform, img


def clear_screen(img):
    img[:, :, :] = 0


def draw_room_segmentations(img, room, transform):
    for segment in room.floors:
        p0 = transform.transform_xy(segment.x_min, segment.y_level)
        p1 = transform.transform_xy(segment.x_max, segment.y_level)
        cv2.line(img, p0, p1, (0, 255, 255))

    for segment in room.ceil_segments:
        p0 = transform.transform_xy(segment[0], segment[1])
        p1 = transform.transform_xy(segment[2], segment[3])
        cv2.line(img, p0, p1, (255, 0, 0))

    for segment in room.wall_segments:
        p0 = transform.transform_xy(segment[0], segment[1])
        p1 = transform.transform_xy(segment[2], segment[3])
        cv2.line(img, p0, p1, (255, 255, 0))


def draw_terrain(img, terrain: Terrain, transform):
    for segment in terrain.segments:
        p0 = transform.transform_xy(segment.x0, segment.y0)
        p1 = transform.transform_xy(segment.x1, segment.y1)
        match segment.type:
            case "floor":
                color = (0, 255, 255)
            case "ceiling":
                color = (255, 0, 0)
            case "left_wall":
                color = (255, 255, 0)
            case "right_wall":
                color = (255, 0, 255)
            case _:
                print('invalid type', segment.type)
                color = (20, 20, 20)
        cv2.line(img, p0, p1, color)


def draw_knight(img, transform, x, y):
    # x, y, w, h = knight["x"], knight["y"], knight["w"], knight["h"]
    w, h = 0.25, 0.641
    low_corner = (x - w, y - h)
    high_corner = (x + w, y + h)
    p0 = transform.transform_xy(*low_corner)
    p1 = transform.transform_xy(*high_corner)
    p2 = transform.transform_xy(x, y)
    cv2.rectangle(img, p0, p1, color=(255, 255, 255), thickness=2)
    cv2.circle(img, p2, 5, (255, 255, 255), -1)


def draw_region(img, transform, region):
    p0 = transform.transform_xy(region.x_min, region.y_min)
    p1 = transform.transform_xy(region.x_max, region.y_max)
    cv2.rectangle(img, p0, p1, color=(255, 255, 255), thickness=2)


def highlight_floor(img, transform, floor, color=(255, 255, 0)):
    p0 = transform.transform_xy(floor.x_min, floor.y_level)
    p1 = transform.transform_xy(floor.x_max, floor.y_level)
    cv2.line(img, p0, p1, color, 5)


def draw_path(img, transform, room, path):
    path_generator = room.region_pathcast_generator(path)
    prev_x, prev_y = path[0]
    for status, x, y, region in path_generator:
        p0 = transform.transform_xy(prev_x, prev_y)
        p1 = transform.transform_xy(x, y)
        color = (0, 0, 255) if status == "failed" else (0, 255, 0)
        cv2.line(img, p0, p1, color, 3)
        if status == "failed":
            break
        prev_x, prev_y = x, y

def draw_path2(img, transform, path, color):
    for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
        p0 = transform.transform_xy(x1, y1)
        p1 = transform.transform_xy(x2, y2)
        cv2.line(img, p0, p1, color, 3)

def draw_object_box(img, transform, obj, color):
    x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
    low_corner = (x - w, y - h)
    high_corner = (x + w, y + h)
    p0 = transform.transform_xy(*low_corner)
    p1 = transform.transform_xy(*high_corner)
    cv2.rectangle(img, p0, p1, color=color, thickness=2)


def print_debug_text(img, text, posx, posy):
    cv2.putText(
        img,
        text,
        (posx, posy),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(255, 255, 255),
        thickness=1,
    )


def interactive_room(room: Room):
    # visualize
    cam_bound = 200
    cam_pad = 10
    x_coords = [
        x
        for x in [
            *[seg[0] for seg in room.raw_segments],
            *[seg[2] for seg in room.raw_segments],
        ]
        if x > -cam_bound and x < cam_bound
    ]
    y_coords = [
        y
        for y in [
            *[seg[1] for seg in room.raw_segments],
            *[seg[3] for seg in room.raw_segments],
        ]
        if y > -cam_bound and y < cam_bound
    ]
    cam_x_min = min(x_coords) - cam_pad
    cam_x_max = max(x_coords) + cam_pad
    cam_y_min = min(y_coords) - cam_pad
    cam_y_max = max(y_coords) + cam_pad
    # print(cam_x_min, cam_x_max, cam_y_min, cam_y_max)
    cam_width, cam_height = (cam_x_max - cam_x_min), (cam_y_max - cam_y_min)
    cam_x, cam_y = (cam_x_min + cam_width / 2), (cam_y_min + cam_height / 2)
    img_width, img_height = window_size = 640, 480
    # img_width, img_height = window_size = 1280, 960
    transform = ScreenTransform(
        img_width, img_height - 2, cam_x, cam_y, cam_width, cam_height
    )
    img = np.zeros((img_height, img_width, 3))

    # Initialize Pygame
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("HK Bot")
    surf = pygame.surfarray.make_surface(img)

    selected_region = None
    air_region = None
    point1 = None
    point2 = None
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                sx, sy = pygame.mouse.get_pos()
                point1 = point2
                point2 = transform.inv_transform_xy(sx, img_height - sy)
                selected_region = room.locate_coarse(*point2)
                if air_region is None:
                    air_region = room.locate_subregion(selected_region, *point2)
                else:
                    is_safe, final_region, final_point = air_region.check_line_safe(
                        point1, point2
                    )
                    print("checked", is_safe, point1, final_point, point2)
                    air_region = final_region
                    point2 = final_point
                # print('clicked', (sx, sy), point2, selected_region)
        if running == False:
            break
        # for idx, region in enumerate(room.floor_regions):
        #     p0 = transform.transform_xy(region.x_min, region.y_min)
        #     p1 = transform.transform_xy(region.x_max, region.y_max)
        #     img2 = img.copy()
        #     color = tuple([int(x) for x in (np.random.random(3) * 250)])
        #     cv2.rectangle(img2, p0, p1, color=color, thickness=-1)
        #     alpha = 0.2
        #     img = cv2.addWeighted(img, alpha, img2, 1-alpha, 0)
        img = np.zeros((img_height, img_width, 3))
        # for idx, region in enumerate(floor_regions):

        for segment in room.floors:
            p0 = transform.transform_xy(segment.x_min, segment.y_min)
            p1 = transform.transform_xy(segment.x_max, segment.y_max)
            cv2.line(img, p0, p1, (0, 255, 255))

        for segment in room.ceil_segments:
            p0 = transform.transform_xy(segment[0], segment[1])
            p1 = transform.transform_xy(segment[2], segment[3])
            cv2.line(img, p0, p1, (255, 0, 0))

        for segment in room.wall_segments:
            p0 = transform.transform_xy(segment[0], segment[1])
            p1 = transform.transform_xy(segment[2], segment[3])
            cv2.line(img, p0, p1, (255, 255, 0))

        if selected_region is not None:
            p0 = transform.transform_xy(selected_region.x_min, selected_region.y_min)
            p1 = transform.transform_xy(selected_region.x_max, selected_region.y_max)
            cv2.rectangle(img, p0, p1, color=(255, 255, 255), thickness=3)

            p0 = transform.transform_xy(air_region.x_min, air_region.y_min)
            p1 = transform.transform_xy(air_region.x_max, air_region.y_max)
            cv2.rectangle(img, p0, p1, color=(200, 128, 128), thickness=2)

            # for subregion in selected_region.overlapping_regions:
            #     p0 = transform.transform_xy(subregion.x_min, subregion.y_min)
            #     p1 = transform.transform_xy(subregion.x_max, subregion.y_max)
            #     cv2.rectangle(img, p0, p1, color=(80, 128, 128), thickness=2)

        if point1 is not None and point2 is not None:
            p0 = transform.transform_xy(*point1)
            p1 = transform.transform_xy(*point2)
            cv2.line(img, p0, p1, (80, 255, 200))
            cv2.circle(img, p0, 5, (80, 255, 200), -1)
            cv2.circle(img, p1, 5, (80, 255, 200), -1)

        cimg = np.transpose(img, (1, 0, 2))
        surf = pygame.surfarray.make_surface(cimg)

        # Draw the image at coordinates (x, y)
        screen.blit(surf, (0, 0))  # Change coordinates as needed

        # Update the display
        pygame.display.flip()
    pygame.quit()


def interactive_jumps(jumps):
    # visualize jumps.txt
    img_width, img_height = window_size = 640, 480
    # img_width, img_height = window_size = 1280, 960
    transform = ScreenTransform(img_width, img_height - 2, 0, 0, 20, 16)
    img = np.zeros((img_height, img_width, 3))

    # Initialize Pygame
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("HK Jumps")
    surf = pygame.surfarray.make_surface(img)

    num_jumps = len(jumps)
    prev_selected_idx = -1
    selected_idx = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                if event.key == pygame.K_LEFT:
                    selected_idx = ((selected_idx - 1) + num_jumps) % num_jumps
                elif event.key == pygame.K_RIGHT:
                    selected_idx = ((selected_idx + 1) + num_jumps) % num_jumps

        if selected_idx != prev_selected_idx:
            print("selected", selected_idx, jumps[selected_idx])
            prev_selected_idx = selected_idx
            img = np.zeros((img_height, img_width, 3))

            for idx, jump in enumerate(jumps):
                for s0, s1 in zip(jump[:-1], jump[1:]):
                    t0, x0, y0 = s0
                    t1, x1, y1 = s1
                    p0 = transform.transform_xy(x0, y0)
                    p1 = transform.transform_xy(x1, y1)
                    color = (255, 0, 0) if idx == selected_idx else (255, 255, 255)
                    cv2.line(img, p0, p1, color)

            selected = jumps[selected_idx]
            for s0, s1 in zip(selected[:-1], selected[1:]):
                t0, x0, y0 = s0
                t1, x1, y1 = s1
                p0 = transform.transform_xy(x0, y0)
                p1 = transform.transform_xy(x1, y1)
                color = (255, 0, 0)
                cv2.line(img, p0, p1, color, thickness=2)
            print("Selected max height", max(map(lambda p: p[2], selected)))

            cimg = np.transpose(img, (1, 0, 2))
            surf = pygame.surfarray.make_surface(cimg)

            # Draw the image at coordinates (x, y)
            screen.blit(surf, (0, 0))  # Change coordinates as needed

            # Update the display
            pygame.display.flip()
    pygame.quit()
