from dataclasses import dataclass
from enum import Enum

class ActorStates(Enum):
    GROUNDED = "grounded"
    IDLE = "idle"
    RUNNING = "running"
    AIRBORNE = "airborne"
    WALL_SLIDING = "wall_sliding"
    HARD_LANDING = "hard_landing"
    DASH_LANDING = "dash_landing"
    NO_INPUT = "no_input"
    PREVIOUS = "previous"

@dataclass
class HeroControllerStates:
    x_pos: float = 0
    y_pos: float = 0
    y_velocity: float = 0
    x_velocity: float = 0
    facingRight: bool = False
    
    onGround: bool = True
    touchingWall: bool = False
    
    wallJumping: bool = False
    wallSliding: bool = False
    attackDir: int = -1 # 0=left, 1=right, 2=up, 3=down
    shroomBouncing: bool = False
    willHardLand: bool = False

    jump_time: float = 0
    double_jump_time: float = 0
    dash_timer: float = 0
    attack_time: float = 0
    bounceTimer: float = 0
    hardLandingTimer: float = 0
    recoil_dx: float = 0
    recoil_dy: float = 0
    airDashed: bool = False
    doubleJumped: bool = False
    currentWalljumpSpeed: float = 0
    walljumpTimer: float = 0
    fallTimer: float = 0
    coyoteTime: float = 0
    recoilTimer: float = 0
    dashing: bool = False
    superDashOnWall: bool = False

    @property
    def facingDir(self):
        if self.facingRight:
            return 1
        else:
            return -1
        
    @property
    def recoilHorizontal(self):
        return self.recoil_dy == 0