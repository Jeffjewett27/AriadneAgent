import copy
import os
import pickle
import numpy as np
from game_data2 import GameData2
from hksocket import (
    run_websocket_server,
    event_queue,
    connected_event,
    stop_websocket_server,
    is_server_running,
)
from jumps import get_selected_jumps, log_knight_state, do_jump
from physics.player_input import PlayerInput
from process_segmentations import process_segmentations
import pygame
from argparse import ArgumentParser

import control
from visualize import (
    draw_knight,
    draw_object_box,
    get_screen_info,
    draw_terrain,
    get_screen_info2,
    print_debug_text,
    draw_path2,
)
from physics.hero_controller import forward_dyanmics


def main_loop(window_size, scene, connect=True):
    # Start listening for data

    if connect:
        run_websocket_server()
        print("Waiting for websocket to connect")
        connected_event.wait()
        if not is_server_running():
            print("Server is not running")
            return

    # Init pygame
    pygame.init()
    clock = pygame.time.Clock()

    # Set up the display
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("HK Bot")
    screen_transform = None

    real_game = GameData2(use_cache=False)
    real_game.scene_name = scene
    simulator = real_game
    game = real_game
    clicked_pos = None

    simulated_physics = False

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
                    real_game.reset_room(use_cache=False)
                elif event.key == pygame.K_s:
                    room_cache_path = os.path.join("rooms", f"{real_game.scene_name}.pkl")
                    with open(room_cache_path, "wb") as file:
                        if simulated_physics:
                            pickle.dump(simulator, file, pickle.HIGHEST_PROTOCOL)
                        else:
                            pickle.dump(real_game, file, pickle.HIGHEST_PROTOCOL)
                    print(f"Saved game state to {room_cache_path}")
                elif event.key == pygame.K_l and real_game.scene_name:
                    room_cache_path = os.path.join("rooms", f"{real_game.scene_name}.pkl")
                    with open(room_cache_path, "rb") as file:
                        if simulated_physics:
                            simulator = pickle.load(file)
                        else:
                            real_game = pickle.load(file)
                    print(f"Loaded game state {room_cache_path}")
                elif event.key == pygame.K_p:
                    simulated_physics = not simulated_physics
                    if simulated_physics:
                        print("Simulating physics now")
                        simulator = copy.deepcopy(real_game)
                        game = simulator
                    else:
                        print("Tracking game state")
                        game = real_game
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
                elif event.key == pygame.K_c:
                    control.press_dash()

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
                elif event.key == pygame.K_c:
                    control.release_dash()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if screen_transform is not None:
                    sx, sy = pygame.mouse.get_pos()
                    clicked_pos = screen_transform.inv_transform_xy(
                        sx, window_size[1] - sy
                    )
                    clicked_pos = tuple([round(float(x), 2) for x in clicked_pos])

        first_data = True
        while not event_queue.empty():
            data = event_queue.get()
            real_game.new_snapshot(data)
            if not first_data:
                print("frame dropped")
            first_data = False
        if simulated_physics:
            framerate = 50
            if simulator.is_room_initialized:

                simulator.knight_state = forward_dyanmics(
                    simulator.knight_state,
                    simulator.terrain,
                    PlayerInput.from_keys(control.controls_pressed),
                    1 / framerate,
                )
                clock.tick(framerate)

        if game.is_room_initialized:
            screen_transform, screen_array = get_screen_info2(
                game.terrain, window_size, min_y=-100
            )
            print_debug_text(
                screen_array, f"Scene: {game.scene_name}", window_size[1] - 50, 20
            )

            draw_terrain(screen_array, game.terrain, screen_transform)
            draw_knight(screen_array, screen_transform, game.knight_state.x_pos, game.knight_state.y_pos)

            real_x = real_game.knight['x']
            real_y = real_game.knight['y']
            sim_x = simulator.knight_state.x_pos
            sim_y = simulator.knight_state.y_pos
            print(f'knight {real_x:.2f}, {sim_x:.2f}, {sim_x - real_x:.2f}')

            for transition in game.get_transitions():
                draw_object_box(screen_array, screen_transform, transition, (0, 0, 200))

            # Update the screen
            cimg = np.transpose(
                screen_array, (1, 0, 2)
            )  # transpose to pygame dimensions
            surf = pygame.surfarray.make_surface(cimg)  # np array to pygame surface
            screen.blit(surf, (0, 0))  # draw surface on screen
            pygame.display.flip()  # update display

    stop_websocket_server()
    control.stop_control_process()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--scene", "-s", default=None, type=str, help="The scene identifier"
    )
    parser.add_argument(
        "--no-connect", "-n", action="store_true", help="Don't connect to server"
    )
    config = parser.parse_args()
    main_loop(window_size=(640, 480), scene=config.scene, connect=not config.no_connect)
