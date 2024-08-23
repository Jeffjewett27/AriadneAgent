import numpy as np
from hksocket import run_websocket_server, event_queue, connected_event, stop_websocket_server, is_server_running
from process_segmentations import process_segmentations
import pygame

from visualize import draw_knight, get_screen_info, clear_screen, draw_room_segmentations

def main_loop(window_size):
    # Start listening for data
    run_websocket_server()
    print('Waiting for websocket to connect')
    connected_event.wait()
    if not is_server_running():
        print('Server is not running')
        return

    # Init pygame
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("HK Bot")
    screen_transform = None

    # Game data
    room = None
    knight = None

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
                if screen_transform is not None:
                    sx, sy = pygame.mouse.get_pos()
                    coord = screen_transform.inv_transform_xy(sx, window_size[1]-sy)
                    print('clicked', coord)
        if not event_queue.empty():
            data = event_queue.get()
            # print(data)
            if 'terrainSegmentation' in data:
                segmentation = data['terrainSegmentation']
                room = process_segmentations(segmentation)
            # if 'playerSnapshot' in data:
            #     player = data['playerSnapshot']
            #     print(player)
            if 'worldObjects' in data:
                for key, wobj in data['worldObjects'].items():
                    name = wobj['name']
                    if name == 'Knight':
                        print(wobj)
                        knight = wobj

        if room is not None:
            screen_transform, screen_array = get_screen_info(room, window_size, min_y=-100)
            # clear_screen(screen_array)
            draw_room_segmentations(screen_array, room, screen_transform)
            if knight is not None:
                draw_knight(screen_array, screen_transform, knight)

            # Update the screen
            cimg = np.transpose(screen_array, (1, 0, 2)) # transpose to pygame dimensions
            surf = pygame.surfarray.make_surface(cimg) # np array to pygame surface
            screen.blit(surf, (0, 0)) # draw surface on screen
            pygame.display.flip() # update display

    stop_websocket_server()

if __name__ == "__main__":
    main_loop(window_size=(640,480))