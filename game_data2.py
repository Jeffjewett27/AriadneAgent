from constants import KNIGHT_HALF_WIDTH
import control
import jumps
from physics.hero_controller_state import HeroControllerStates
from process import process_terrain


class GameData2:
    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self.clear_scene()

    def new_snapshot(self, snapshot):
        new_scene_name = snapshot.get("sceneName", None)
        if self.scene_name != new_scene_name:
            self.clear_scene()
            self.scene_name = new_scene_name
        terrain_segmentation = snapshot.get("terrainSegmentation", None)
        hazard_segmentation = snapshot.get("staticHazardSegmentation", None)
        get_new_room = False
        if terrain_segmentation is not None:
            if self.terrain_segmentation != terrain_segmentation:
                get_new_room = True
            self.terrain_segmentation = terrain_segmentation
        if hazard_segmentation is not None:
            if self.hazard_segmentation != hazard_segmentation:
                get_new_room = True
            self.hazard_segmentation = hazard_segmentation
        if self.terrain is None or get_new_room:
            self.reset_room(self.use_cache)

        if "currentTime" in snapshot:
            self.time = snapshot["currentTime"]

        if "worldObjects" in snapshot:
            for key, wobj in snapshot["worldObjects"].items():
                name = wobj["name"]
                if name == "Knight":
                    # print(wobj)
                    self.knight = wobj
                    self.knight_state.x_pos = wobj['x']
                    self.knight_state.y_pos = wobj['y']

                wobj["eid"] = key
                self.objects[key] = wobj

        # if self.terrain is not None:
        #     grounded_floor = self.terrain.get_grounded_floor(*self.knight_position)
        #     if grounded_floor is not None:
        #         self.last_grounded_floor = grounded_floor

    def clear_scene(self):
        self.scene_name = None
        self.knight = None
        self.player_state = None
        self.objects = {}
        self.terrain_segmentation = None
        self.hazard_segmentation = None
        self.terrain = None
        self.time = None
        self.last_grounded_floor = None
        self.knight_state = HeroControllerStates()

        self.current_jump = None
        self.queued_jump = None
        self.move_target_x = None
        self.frames_til_tap = 0
        self.from_floor = None

    def reset_room(self, use_cache=True):
        self.terrain = process_terrain(self, use_cache=use_cache)
        print('terrain processed', self.terrain)

    @property
    def is_room_initialized(self):
        return self.terrain is not None and self.knight is not None and self.knight_state is not None

    @property
    def knight_position(self):
        if self.knight is None:
            return None
        return self.knight["x"], self.knight["y"]

    def get_transitions(self):
        return list(
            filter(
                lambda obj: obj.get("type", None) == "Transition", self.objects.values()
            )
        )
