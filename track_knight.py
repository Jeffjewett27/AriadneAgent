from collections import deque
import copy
import os
import pickle
import time
import numpy as np
from game_data import GameData
from hksocket import (
    run_websocket_server,
    event_queue,
    connected_event,
    stop_websocket_server,
    is_server_running,
)
from physics.hero_controller_state import HeroControllerStates
from physics.inverse_kinematics import get_apex_height_range, inverse_kinematics
from physics.player_input import PlayerInput
from planning import sample_connections, visualize_floor_connections, visualize_graph
import pygame
from argparse import ArgumentParser

import control
from visualize import (
    draw_knight,
    draw_object_box,
    draw_terrain,
    get_screen_info2,
    print_debug_text,
    draw_path2,
)
from physics.hero_controller import forward_dynamics, forward_dynamics_basic
from motion import CompositePrimitive, Jump, MotionPrimitive, TerminateOnGroundWrapper


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

    real_game = GameData(use_cache=False)
    real_game.scene_name = scene
    simulator = real_game
    game = real_game
    clicked_pos = None

    simulated_physics = False
    framerate = 50

    control.start_control_process()

    # jump_motion = Jump.sample(1)
    jump_motion = Jump(jump_time=0.5, is_right=False, x_wait=0.5, x_hold_time=0, x_pause=0)
    action_iterator = None
    execution_prim = None
    execution_queue = deque()
    execution_reset_prim = True
    last_ground = None
    trajectory = []
    target_segments = []
    target_position = None

    running = True
    executing = False
    last_plan_time = 0
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
                    room_cache_path = os.path.join(
                        "rooms", f"{real_game.scene_name}.pkl"
                    )
                    with open(room_cache_path, "wb") as file:
                        if simulated_physics:
                            pickle.dump(simulator, file, pickle.HIGHEST_PROTOCOL)
                        else:
                            pickle.dump(real_game, file, pickle.HIGHEST_PROTOCOL)
                    print(f"Saved game state to {room_cache_path}")
                elif event.key == pygame.K_l and real_game.scene_name:
                    room_cache_path = os.path.join(
                        "rooms", f"{real_game.scene_name}.pkl"
                    )
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
                        simulator = copy.copy(real_game)
                        game = simulator
                    else:
                        print("Tracking game state")
                        game = real_game
                elif event.key == pygame.K_e:
                    # execute the control action
                    # action_iterator = PlayerInput.get_action_iterator(
                    #     jump_motion.control_sequence(), framerate
                    # )
                    executing = not executing
                    print(f'Trajectory exectution toggled {"on" if executing else "off"}')
                elif event.key == pygame.K_d:
                    # sample action
                    # jump_motion = Jump(0.04, True, 0, 0, 0.5)
                    # jump_motion = MotionPrimitive.sample()
                    # from_state = game.knight_state
                    # to_state = game.knight_state.copy()
                    # to_state.x_pos += 5
                    # success, motion = inverse_kinematics(
                    #     from_state, to_state, game.terrain
                    # )
                    # if success:
                    #     jump_motion = motion
                    # else:
                    #     print("Could not inverse_kinematics")
                    # invkin_target = HeroControllerStates(
                    #     x_pos=game.knight_state.x_pos+3, y_pos=game.knight_state.y_pos+3, y_velocity=-5, onGround=False
                    # )
                    # success, inv_motion = inverse_kinematics(
                    #     game.knight_state, invkin_target, game.terrain
                    # )
                    # if success:
                    #     jump_motion = inv_motion
                    # visualize_floor_connections(rrt_graph, game.terrain)
                    # visualize_graph(game.planner, game.terrain)
                    # new_prim = MotionPrimitive.sample(2)
                    # execution_reset_prim = True
                    # print('Executing prim', new_prim)
                    # new_prim = TerminateOnGroundWrapper(new_prim)
                    # execution_queue.append(new_prim)
                    print('Executing:')
                    print(execution_prim)
                    print('Queue:')
                    print('\n'.join([str(p) for p in execution_queue]))
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

                    # invkin_target = HeroControllerStates(
                    #     x_pos=clicked_pos[0], y_pos=clicked_pos[1], y_velocity=None
                    # )
                    # success, inv_motion = inverse_kinematics(
                    #     game.knight_state, invkin_target, game.terrain
                    # )
                    # if success:
                    #     jump_motion = inv_motion
                    target_position = clicked_pos
                    trajectory, target_segments = game.planner.find_path(game.knight_state, clicked_pos)
            

        cur_time = time.time()
        if cur_time - last_plan_time > 1.2 and game.knight_state.onGround and (execution_prim is None or execution_prim.is_terminated):
            last_plan_time = cur_time
            if clicked_pos is not None:
                trajectory, target_segments = game.planner.find_path(game.knight_state, clicked_pos)
                
            if trajectory is not None:
                execution_queue.clear()
                prims = [c for _, c in trajectory if c is not None]
                prims = CompositePrimitive.merge_primitives(prims)
                for new_prim in prims:
                    execution_queue.append(new_prim)
        # if action_iterator is not None:
        #     try:
        #         next_action = next(action_iterator)
        #         control.set_controls_pressed(next_action.get_keys_pressed())
        #     except StopIteration:
        #         action_iterator = None
        #         control.set_controls_pressed([False] * 7)
        if executing:
            if execution_prim is None and len(execution_queue) > 0:
                execution_prim = execution_queue.pop()
                execution_prim.reset_state(game.knight_state)
                print('Executing primitive', execution_prim)
            if execution_prim is not None:
                if execution_prim.is_terminated:
                    execution_prim = None
                    control.set_controls_pressed([False] * 7)
                else:
                    next_action, terminated = execution_prim.get_action(game.knight_state)
                    control.set_controls_pressed(next_action.get_keys_pressed())
        else:
            execution_prim = None
        control.tick_controls()

        # Process real game data
        first_data = True
        while not event_queue.empty():
            data = event_queue.get()
            real_game.new_snapshot(data)
            if not first_data:
                print("frame dropped")
            first_data = False
        # Process simulation
        if simulated_physics:
            if simulator.is_room_initialized:
                before_y = simulator.knight_state.y_pos
                knight_state = forward_dynamics_basic(
                    simulator.knight_state,
                    simulator.terrain,
                    PlayerInput.from_keys(control.controls_pressed),
                    1 / framerate,
                )
                simulator.update_knight_state(knight_state)
                # print('y change', before_y, simulator.knight_state.y_pos)

        if game.is_room_initialized:
            screen_transform, screen_array = get_screen_info2(
                game.terrain, window_size, min_y=-100
            )
            print_debug_text(
                screen_array, f"Scene: {game.scene_name}", window_size[1] - 50, 20
            )
            print_debug_text(
                screen_array,
                f"<x={real_game.knight_state.x_pos:.2f}, y={real_game.knight_state.y_pos:.2f}, dy={real_game.knight_state.y_velocity:.2f}>",
                20,
                20,
            )
            if simulated_physics:
                print_debug_text(
                    screen_array,
                    f"<x={simulator.knight_state.x_pos:.2f}, y={simulator.knight_state.y_pos:.2f}, dy={simulator.knight_state.y_velocity:.2f}>",
                    20,
                    40,
                )

            # temp_debug_str = f"{get_apex_height_range(game.knight_state)}, {max(map(lambda s: s.y_pos, game.jump_trajectory), default=0)}"
            temp_debug_str = ','.join([f"{x.y_pos-y.y_pos:.1f}" for x, y in zip(game.jump_trajectory[1:], game.jump_trajectory[:-1])])
            print_debug_text(screen_array, temp_debug_str, 20, window_size[1] - 20)

            draw_terrain(screen_array, game.terrain, screen_transform)
            draw_knight(
                screen_array,
                screen_transform,
                game.knight_state.x_pos,
                game.knight_state.y_pos,
            )

            # Find ground segment
            # ground = [
            #     x
            #     for x in game.terrain.find_touching_segments(
            #         np.array((game.knight_state.x_pos, game.knight_state.y_pos)), 0.05
            #     )
            #     if x.type == "floor"
            # ]
            # if len(ground) > 0:
            #     last_ground = ground[0]
            # if game.last_grounded_floor:
            #     draw_path2(
            #         screen_array,
            #         screen_transform,
            #         [
            #             (
            #                 game.last_grounded_floor.x_min,
            #                 game.last_grounded_floor.y_min,
            #             ),
            #             (
            #                 game.last_grounded_floor.x_max,
            #                 game.last_grounded_floor.y_max,
            #             ),
            #         ],
            #         (200, 200, 200),
            #     )

            draw_path2(screen_array, screen_transform, [(state.x_pos, state.y_pos) for state, _ in trajectory], (255, 255, 255))
            if target_position is not None:
                draw_object_box(screen_array, screen_transform, {"x": target_position[0], "y": target_position[1], "w": 1, "h": 1}, (255, 0, 0))

            # Project actions
            # projection_state = simulator.knight_state
            # projection_coords = [(projection_state.x_pos, projection_state.y_pos)]
            # for action in PlayerInput.get_action_iterator(
            #     jump_motion.control_sequence(), framerate
            # ):
            #     projection_state = forward_dynamics_basic(
            #         projection_state, simulator.terrain, action, 1 / framerate
            #     )
            #     projection_coords.append(
            #         (projection_state.x_pos, projection_state.y_pos)
            #     )
            # draw_path2(
            #     screen_array, screen_transform, projection_coords, (255, 255, 255)
            # )
            projection_state = simulator.knight_state
            projection_coords = [(projection_state.x_pos, projection_state.y_pos)]
            hit, new_pos, new_vel, segs = simulator.terrain.integrate_motion(np.array((projection_state.x_pos, projection_state.y_pos)), np.array([0, 15]), 0.5)
            projection_coords.append(tuple(new_pos))
            # print(hit, new_pos, new_vel, segs)
            draw_path2(
                screen_array, screen_transform, projection_coords, (255, 255, 255)
            )

            for transition in game.get_transitions():
                draw_object_box(screen_array, screen_transform, transition, (0, 0, 200))

            # Update the screen
            cimg = np.transpose(
                screen_array, (1, 0, 2)
            )  # transpose to pygame dimensions
            surf = pygame.surfarray.make_surface(cimg)  # np array to pygame surface
            screen.blit(surf, (0, 0))  # draw surface on screen
            pygame.display.flip()  # update display

        clock.tick(framerate)

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
