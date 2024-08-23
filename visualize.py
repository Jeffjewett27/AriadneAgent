from typing import Tuple
import cv2
import numpy as np
import pygame

from room import Room

class ScreenTransform:
    """
    Transforms from world space into camera space.
    """

    def __init__(self, screen_width: int, screen_height: int, cam_x: float, cam_y: float, 
            cam_width: float, cam_height: float):

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
        return int((world_X - self.cam_x) * self.w_factor + self.screen_width / 2), \
            int((self.cam_y - world_y) * self.h_factor + self.screen_height / 2)
    
    def inv_transform_xy(self, cam_x: int, cam_y: int) -> Tuple[float, float]:
        return (cam_x - self.screen_width / 2) / self.w_factor + self.cam_x, \
            (cam_y - self.screen_height / 2) / self.h_factor + self.cam_y,
        
def get_screen_info(room: Room, screen_size, cam_pad=10, min_x=-200, min_y=-200, max_x=200, max_y=200):
    cam_pad = 10
    x_coords = [x for x in [*[seg[0] for seg in room.raw_segments], *[seg[2] for seg in room.raw_segments]] if x >= min_x and x <= max_x]
    y_coords = [y for y in [*[seg[1] for seg in room.raw_segments], *[seg[3] for seg in room.raw_segments]] if y >= min_y and y <= max_y]
    cam_x_min = min(x_coords) - cam_pad
    cam_x_max = max(x_coords) + cam_pad
    cam_y_min = min(y_coords) - cam_pad
    cam_y_max = max(y_coords) + cam_pad
    cam_width, cam_height = (cam_x_max - cam_x_min), (cam_y_max - cam_y_min)
    cam_x, cam_y = (cam_x_min + cam_width / 2), (cam_y_min + cam_height / 2)
    img_width, img_height = screen_size
    transform = ScreenTransform(img_width, img_height-2, cam_x, cam_y, cam_width, cam_height)
    img = np.zeros((img_height, img_width, 3))
    return transform, img

def clear_screen(img):
    img[:,:,:] = 0

def draw_room_segmentations(img, room, transform):
    for segment in room.floor_segments:
        p0 = transform.transform_xy(segment[0], segment[1])
        p1 = transform.transform_xy(segment[2], segment[3])
        cv2.line(img, p0, p1, (0, 255, 255))

    for segment in room.ceil_segments:
        p0 = transform.transform_xy(segment[0], segment[1])
        p1 = transform.transform_xy(segment[2], segment[3])
        cv2.line(img, p0, p1, (255, 0, 0))

    for segment in room.wall_segments:
        p0 = transform.transform_xy(segment[0], segment[1])
        p1 = transform.transform_xy(segment[2], segment[3])
        cv2.line(img, p0, p1, (255, 255, 0))

def draw_knight(img, transform, knight):
    x, y, w, h = knight['x'], knight['y'], knight['w'], knight['h']
    low_corner = (x - w, y - h)
    high_corner = (x + w, y + h)
    p0 = transform.transform_xy(*low_corner)
    p1 = transform.transform_xy(*high_corner)
    p2 = transform.transform_xy(x, y)
    cv2.rectangle(img, p0, p1, color=(255,255,255), thickness=2)
    cv2.circle(img, p2, 5, (255, 255, 255), -1)

def interactive_room(room: Room):
    # visualize
    cam_bound = 200
    cam_pad = 10
    x_coords = [x for x in [*[seg[0] for seg in room.raw_segments], *[seg[2] for seg in room.raw_segments]] if x > -cam_bound and x < cam_bound]
    y_coords = [y for y in [*[seg[1] for seg in room.raw_segments], *[seg[3] for seg in room.raw_segments]] if y > -cam_bound and y < cam_bound]
    cam_x_min = min(x_coords) - cam_pad
    cam_x_max = max(x_coords) + cam_pad
    cam_y_min = min(y_coords) - cam_pad
    cam_y_max = max(y_coords) + cam_pad
    # print(cam_x_min, cam_x_max, cam_y_min, cam_y_max)
    cam_width, cam_height = (cam_x_max - cam_x_min), (cam_y_max - cam_y_min)
    cam_x, cam_y = (cam_x_min + cam_width / 2), (cam_y_min + cam_height / 2)
    img_width, img_height = window_size = 640, 480
    # img_width, img_height = window_size = 1280, 960
    transform = ScreenTransform(img_width, img_height-2, cam_x, cam_y, cam_width, cam_height)
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
                point2 = transform.inv_transform_xy(sx, img_height-sy)
                selected_region = room.locate_coarse(*point2)
                if air_region is None:
                    air_region = room.locate_subregion(selected_region, *point2)
                else:
                    is_safe, final_region, final_point = air_region.check_line_safe(point1, point2)
                    print('checked', is_safe, point1, final_point, point2)
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

        for segment in room.floor_segments:
            p0 = transform.transform_xy(segment[0], segment[1])
            p1 = transform.transform_xy(segment[2], segment[3])
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
            cv2.rectangle(img, p0, p1, color=(255,255,255), thickness=3)

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
