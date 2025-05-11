from dataclasses import dataclass


@dataclass
class PlayerInput:
    vertical: float = 0
    horizontal: float = 0
    jump: bool = False
    attack: bool = False
    dash: bool = False

    @property
    def move_direction(self):
        if self.horizontal < 0:
            return -1
        elif self.horizontal > 0:
            return 1
        return 0

    @property
    def down(self):
        return self.vertical < 0

    @property
    def up(self):
        return self.vertical > 0

    @property
    def left(self):
        return self.horizontal < 0

    @property
    def right(self):
        return self.horizontal > 0

    def get_keys_pressed(self):
        keys_pressed = [False] * 7
        if self.vertical > 0:
            keys_pressed[0] = True
        elif self.vertical < 0:
            keys_pressed[2] = True
        if self.horizontal > 0:
            keys_pressed[3] = True
        elif self.horizontal < 0:
            keys_pressed[1] = True
        keys_pressed[4] = self.jump
        keys_pressed[5] = self.attack
        keys_pressed[6] = self.dash
        return keys_pressed

    @staticmethod
    def from_keys(keys_pressed):
        # controls_pressed = [
        #     False, # up=0
        #     False, # left=1
        #     False, # down=2
        #     False, # right=3
        #     False, # jump=4
        #     False # attack=5
        # ]
        vertical = int(keys_pressed[0]) - int(keys_pressed[2])
        horizontal = int(keys_pressed[3]) - int(keys_pressed[1])
        return PlayerInput(
            vertical, horizontal, keys_pressed[4], keys_pressed[5], keys_pressed[6]
        )

    @staticmethod
    def get_action_iterator(actions: list[tuple["PlayerInput", float]], framerate):
        dt = 1 / framerate
        for action, duration in actions:
            t = duration
            while t > dt / 2:
                yield action
                t -= dt
