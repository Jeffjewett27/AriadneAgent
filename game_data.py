

from constants import KNIGHT_HALF_WIDTH
import control
import jumps
from process_segmentations import process_segmentations

class GameData:

    def __init__(self):
        self.clear_scene()

    def new_snapshot(self, snapshot):
        new_scene_name= snapshot.get('sceneName', None)
        if self.scene_name != new_scene_name:
            self.clear_scene()
            self.scene_name = new_scene_name
        terrain_segmentation = snapshot.get('terrainSegmentation', None)
        hazard_segmentation = snapshot.get('staticHazardSegmentation', None)
        get_new_room = False
        if terrain_segmentation is not None:
            if self.terrain_segmentation != terrain_segmentation:
                get_new_room = True
            self.terrain_segmentation = terrain_segmentation
        if hazard_segmentation is not None:
            if self.hazard_segmentation != hazard_segmentation:
                get_new_room = True
            self.hazard_segmentation = hazard_segmentation
        if self.room is None or get_new_room:
            self.reset_room()

        if 'currentTime' in snapshot:
            self.time = snapshot['currentTime']

        if 'worldObjects' in snapshot:
            for key, wobj in snapshot['worldObjects'].items():
                name = wobj['name']
                if name == 'Knight':
                    # print(wobj)
                    self.knight = wobj
                wobj['eid'] = key
                self.objects[key] = wobj

        if self.room is not None:
            grounded_floor = self.room.get_grounded_floor(*self.knight_position)
            if grounded_floor is not None:
                self.last_grounded_floor = grounded_floor

    def clear_scene(self):
        self.scene_name = None
        self.knight = None
        self.knight_state = None
        self.objects = {}
        self.terrain_segmentation = None
        self.hazard_segmentation = None
        self.room = None
        self.time = None
        self.last_grounded_floor = None

        self.current_jump = None
        self.queued_jump = None
        self.move_target_x = None
        self.frames_til_tap = 0
        self.from_floor = None

    def reset_room(self, use_cache=True):
        self.room = process_segmentations(self, use_cache=use_cache)

    @property
    def is_room_initialized(self):
        return self.room is not None and self.knight is not None
    
    @property
    def knight_position(self):
        if self.knight is None:
            return None
        return self.knight['x'], self.knight['y']

    def get_transitions(self):
        return list(filter(lambda obj: obj.get('type', None) == 'Transition', self.objects.values()))
    
    def is_grounded(self, x, y):
        def check_ground(x, y):
            floor = self.room.get_over_floor(x, y)
            if floor is None:
                return False
            return floor.x_min <= x <= floor.x_max and abs(y - floor.y_level) < 0.03
        return check_ground(x-KNIGHT_HALF_WIDTH, y) or check_ground(x+KNIGHT_HALF_WIDTH, y)
    
    def begin_jump(self, jump_doer):
        self.current_jump = jump_doer

    def do_movement(self):
        cancel_movement = False

        if self.current_jump is not None:
            is_jumped_finished = next(self.current_jump)
            if is_jumped_finished:
                self.current_jump = None
        kx, ky = self.knight_position
        hold_threshold = 2
        tap_threshold = 0.3
        if self.move_target_x is not None:
            delta_x = self.move_target_x - kx
            print(delta_x, self.move_target_x)
            if delta_x > hold_threshold:
                cancel_movement = cancel_movement or self.from_floor != self.last_grounded_floor
                # hold right
                control.press_right()
            elif delta_x < -hold_threshold:
                cancel_movement = cancel_movement or self.from_floor != self.last_grounded_floor
                # hold left
                control.press_left()
            elif abs(delta_x) >= tap_threshold:
                cancel_movement = cancel_movement or self.from_floor != self.last_grounded_floor
                control.release_right()
                control.release_left()
                self.frames_til_tap += 1
                if self.frames_til_tap > 100:
                    if delta_x > 0:
                        control.press_right()
                        control.release_right()
                    else:
                        control.press_left()
                        control.release_left()
                    self.frames_til_tap = 0
            else:
                cancel_movement = True

        if cancel_movement:
            control.release_right()
            control.release_left()
            self.move_target_x = None
            self.current_jump = self.queued_jump
            self.queued_jump = None
            self.frames_til_tap = 0
            cancel_movement = False

    def begin_navigate_to_floor(self, from_floor, to_floor):
        # assumes knight is on from_floor

        self.from_floor = from_floor
        padding = 0.4
        # kx, ky = self.knight_position
        if from_floor.x_max < to_floor.x_min:
            # to_floor is to the right
            self.move_target_x = from_floor.x_max - padding # add a bit of padding
            delta_x = self.move_target_x - (to_floor.x_min + padding)
            delta_y = to_floor.y_level - from_floor.y_level
            min_y_before_x = 3 if abs(delta_x) < 3 else 0
            self.queued_jump = jumps.do_jump(self, 5.5, to_floor.x_min + padding, min_y_before_x)
        elif from_floor.x_min > to_floor.x_max:
            # to_floor is to the left
            self.move_target_x = from_floor.x_min + padding # add a bit of padding
            delta_x = self.move_target_x - (to_floor.x_max - padding)
            delta_y = to_floor.y_level - from_floor.y_level
            min_y_before_x = 3 if abs(delta_x) < 3 else 0
            self.queued_jump = jumps.do_jump(self, 5.5, to_floor.x_max - padding, min_y_before_x)
        elif from_floor.y_level < to_floor.y_level:
            if from_floor.x_min <= to_floor.x_min <= from_floor.x_max:
                # to_floor is above. come from left side
                self.move_target_x = to_floor.x_min - padding
                self.queued_jump = jumps.do_jump(self, 5.5, to_floor.x_min + padding, 4)
            elif from_floor.x_min <= to_floor.x_max <= from_floor.x_max:
                # to_floor is above. come from left side
                self.move_target_x = to_floor.x_max + padding
                self.queued_jump = jumps.do_jump(self, 5.5, to_floor.x_max - padding, 4)
        else:
            # floor is below
            if to_floor.x_min < from_floor.x_min:
                # below and to left
                self.move_target_x = from_floor.x_min - 0.4
            elif from_floor.x_max < to_floor.x_max:
                # below and to right
                self.move_target_x = from_floor.x_max + 0.4
            else:
                # no idea
                pass