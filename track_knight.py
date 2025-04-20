import numpy as np
from game_data import GameData
from hksocket import run_websocket_server, event_queue, connected_event, stop_websocket_server, is_server_running
from jumps import get_selected_jumps, log_knight_state, do_jump
from process_segmentations import process_segmentations
import pygame

import control
from visualize import draw_knight, draw_object_box, draw_path, draw_region, get_screen_info, clear_screen, draw_room_segmentations, highlight_floor, print_debug_text

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

    selected_jumps = get_selected_jumps()
    game = GameData()
    floor_path = []
    clicked_pos = None

    control.start_control_process()

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
                if event.key == pygame.K_r:
                    game.reset_room(use_cache=False)
                elif event.key == pygame.K_UP:
                    control.press_up()
                elif event.key == pygame.K_DOWN:
                    control.press_down()
                elif event.key == pygame.K_LEFT:
                    control.press_left()
                elif event.key == pygame.K_RIGHT:
                    control.press_right()
                elif event.key == pygame.K_z:
                    control.press_jump()
                elif event.key == pygame.K_x:
                    control.press_attack()
                elif event.key == pygame.K_j:
                    # game.begin_jump(do_jump(game, 5.5, kx + 1, 4))
                    from_floor = game.last_grounded_floor
                    if from_floor is not None and from_floor in floor_path:
                        idx = floor_path.index(from_floor)
                        if idx + 1 < len(floor_path):
                            game.begin_navigate_to_floor(from_floor, floor_path[idx+1])

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    control.release_up()
                elif event.key == pygame.K_DOWN:
                    control.release_down()
                elif event.key == pygame.K_LEFT:
                    control.release_left()
                elif event.key == pygame.K_RIGHT:
                    control.release_right()
                elif event.key == pygame.K_z:
                    control.release_jump()
                elif event.key == pygame.K_x:
                    control.release_attack()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if screen_transform is not None:
                    sx, sy = pygame.mouse.get_pos()
                    clicked_pos = screen_transform.inv_transform_xy(sx, window_size[1]-sy)
                    clicked_pos = tuple([round(float(x), 2) for x in clicked_pos])
                    if game.is_room_initialized:
                        floor_path = game.room.pathfind_floors(game.knight_position, clicked_pos)
        if not event_queue.empty():
            data = event_queue.get()
            game.new_snapshot(data)

            if game.is_room_initialized:
                kx, ky = game.knight['x'], game.knight['y']
                # grounded = room.is_grounded(kx, ky)
                # floor = room.get_over_floor(kx, ky)
                region = game.room.locate_coarse(kx, ky)
                # print('knight', kx, ky, grounded, [float(x) for x in floor[:3]], region)
                # draw_region(screen_array, screen_transform, region)
                log_knight_state(game.room, game.knight, None, game.time)

        if game.is_room_initialized:
            screen_transform, screen_array = get_screen_info(game.room, window_size, min_y=-100)
            print_debug_text(screen_array, f'Scene: {game.scene_name}', window_size[1]-50, 20)
            
            kx, ky = game.knight['x'], game.knight['y']
            print_debug_text(screen_array, f'Knight: {kx}, {ky}', 10, 20)
            print_debug_text(screen_array, f'Target: {clicked_pos}', 10, 40)
            grounded = game.room.is_grounded(kx, ky)
            floor = game.room.get_over_floor(kx, ky)
            region = game.room.locate_coarse(kx, ky)
            # print('knight', kx, ky, grounded, [float(x) for x in floor[:3]], region)
            if region is None:
                print('no region at ', kx, ky)
            else:
                draw_region(screen_array, screen_transform, region)
                # if region.over_floor is not None:
                #     highlight_floor(screen_array, screen_transform, region.over_floor)
                #     for floor in region.over_floor.floor_neighbors:
                #         highlight_floor(screen_array, screen_transform, floor, color=(128, 128, 0))
                
            is_next_floor = False
            for neighbor_floor in floor_path:
                color = (128, 128, 0)
                if neighbor_floor == game.last_grounded_floor:
                    color = (128, 128, 200)
                    is_next_floor = True
                elif is_next_floor:
                    color = (200, 128, 128)
                    is_next_floor = False
                highlight_floor(screen_array, screen_transform, neighbor_floor, color=color)
            # jump_path = np.array([[0.0, 0.0, 0.0], [0.0999999999994543, 0.8279999999999959, 1.3599999999999994], [0.1999999999998181, 1.6579999999999941, 2.9299999999999997], [0.29799999999977445, 2.479999999999997, 4.268999999999998], [0.39800000000013824, 3.308, 5.148], [0.49799999999959255, 4.142999999999994, 5.555]])
            # jump_path = jump_path[:,1:]
            # knight_offset = np.array([kx, ky])
            # for jump in selected_jumps:
            #     jump_path = jump + knight_offset
            #     draw_path(screen_array, screen_transform, game.room, list(jump_path))
            #     log_knight_state(game.room, game.knight, None, game.time)
            # clear_screen(screen_array)
            draw_room_segmentations(screen_array, game.room, screen_transform)
            draw_knight(screen_array, screen_transform, game.knight)

            for transition in game.get_transitions():
                draw_object_box(screen_array, screen_transform, transition, (0, 0, 200))

            # Update the screen
            cimg = np.transpose(screen_array, (1, 0, 2)) # transpose to pygame dimensions
            surf = pygame.surfarray.make_surface(cimg) # np array to pygame surface
            screen.blit(surf, (0, 0)) # draw surface on screen
            pygame.display.flip() # update display

            game.do_movement()

    stop_websocket_server()
    control.stop_control_process()

if __name__ == "__main__":
    main_loop(window_size=(640,480))