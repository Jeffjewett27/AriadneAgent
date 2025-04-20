import copy
from physics.hero_controller_state import HeroControllerStates
from physics.environment import Terrain
from physics.player_input import PlayerInput
import numpy as np

FIXED_UPDATE_FRAMERATE = 50
ATTACK_COOLDOWN_TIME = 0.41
ATTACK_DURATION = 0.35
BIG_FALL_TIME = 1.1
BOUNCE_VELOCITY = 12
DASH_COOLDOWN = 0.6
DASH_SPEED = 20
DASH_TIME = 0.25
DOUBLE_JUMP_STEPS = 9
DOUBLE_JUMP_TIME = DOUBLE_JUMP_STEPS / FIXED_UPDATE_FRAMERATE
DOUBLE_JUMP_DELAY_STEPS = 3
DOUBLE_JUMP_DELAY = DOUBLE_JUMP_DELAY_STEPS / FIXED_UPDATE_FRAMERATE
GRAVITY = 0.79
JUMP_SPEED = 16.65
JUMP_STEPS = 9
JUMP_TIME = JUMP_STEPS / FIXED_UPDATE_FRAMERATE
JUMP_STEPS_MIN = 4
JUMP_TIME_MIN = JUMP_STEPS_MIN / FIXED_UPDATE_FRAMERATE
MAX_FALL_VELOCITY = 20
RECOIL_HOR_VELOCITY = 3.75
RUN_SPEED = 8.35
WALLSLIDE_DECEL = 0
WALLSLIDE_SPEED = -8
WJ_KICKOFF_SPEED = 16
WJLOCK_STEPS_LONG = 10
WJLOCK_TIME = WJLOCK_STEPS_LONG / FIXED_UPDATE_FRAMERATE
WJ_DECEL = (WJ_KICKOFF_SPEED - RUN_SPEED) / WJLOCK_TIME


def forward_dyanmics(
    old_state: HeroControllerStates,
    environment: Terrain,
    controls: PlayerInput,
    duration: float,
):
    new_state: HeroControllerStates = copy.copy(old_state)
    gravity = GRAVITY
    # -------------------------
    # FixedUpdate:
    # check recoil cancel
    # new_state.onGround = False
    # check crash landing

    # check if recoiling --- damage recoil, no gravity, use recoilVector
    if new_state.recoilTimer > 0:
        if not new_state.recoilHorizontal:
            # damage recoil has no gravity
            gravity = 0
        new_state.x_velocity = new_state.recoil_dx
        new_state.y_velocity = new_state.recoil_dy
    # else if not dashing:
    #   Move(move_direction) --- horizontal motion dependent on player state
    #   horizontal recoil
    elif not new_state.dashing:
        if not new_state.wallSliding:
            new_state.x_velocity = RUN_SPEED * controls.move_direction
        if new_state.recoilHorizontal:  # horizontal
            if new_state.recoil_dx > 0:  # right
                if new_state.x_velocity < RECOIL_HOR_VELOCITY:
                    new_state.x_velocity = RECOIL_HOR_VELOCITY
                else:
                    new_state.x_velocity += RECOIL_HOR_VELOCITY
            elif new_state.recoil_dx < 0:  # left
                if new_state.x_velocity > -RECOIL_HOR_VELOCITY:
                    new_state.x_velocity = -RECOIL_HOR_VELOCITY
                else:
                    new_state.x_velocity -= RECOIL_HOR_VELOCITY

    # Jump
    if new_state.jump_time > 0:
        new_state.jump_time -= duration
        new_state.y_velocity = JUMP_SPEED
        new_state.coyoteTime = 0

    # doublejump
    if new_state.double_jump_time > 0:
        if new_state.double_jump_time <= DOUBLE_JUMP_TIME:
            new_state.y_velocity = JUMP_SPEED
        new_state.jump_time -= duration

    # dash --- pause gravity, get vector in direction
    if new_state.dash_timer > 0:
        if new_state.dash_timer > (DASH_COOLDOWN - DASH_TIME):
            gravity = 0
            new_state.hardLandingTimer = 0
            new_state.x_velocity = DASH_SPEED * new_state.facingDir
        new_state.dash_timer -= duration
    # TODO: spell recoil

    # bouncing vertical velocity
    if new_state.bounceTimer > 0:
        new_state.y_velocity = BOUNCE_VELOCITY
        new_state.bounceTimer -= duration
    # walljump horizontal force
    if new_state.walljumpTimer > 0:
        new_state.x_velocity += new_state.currentWalljumpSpeed
        new_state.currentWalljumpSpeed -= (
            WJ_DECEL * duration * np.sign(new_state.currentWalljumpSpeed)
        )

    # unstick from wall

    # cap fall velocity
    if new_state.y_velocity < -MAX_FALL_VELOCITY:
        new_state.y_velocity = -MAX_FALL_VELOCITY

    # handle input queue
    # wall slide slowing
    if new_state.wallSliding:
        # WALLSLIDE_DECEL is 0, so this can be simplified
        # if np.abs(new_state.y_velocity) > WALLSLIDE_SPEED:
        #     new_state.y_velocity -= WALLSLIDE_DECEL * np.sign(new_state.y_velocity)
        # if new_state.y_velocity < WALLSLIDE_SPEED:
        #     new_state.y_velocity = WALLSLIDE_SPEED
        new_state.y_velocity = max(new_state.y_velocity, WALLSLIDE_SPEED)

    # superdash on wall pause
    if new_state.superDashOnWall:
        new_state.x_velocity = 0
        new_state.y_velocity = 0

    # ------------------------
    # Update:

    # update fall timer
    if not new_state.onGround and new_state.y_velocity < -0.000001:
        new_state.wallJumping = False
        if new_state.wallSliding:
            new_state.fallTimer = 0
        else:
            new_state.fallTimer += duration
        if new_state.fallTimer > BIG_FALL_TIME:
            new_state.willHardLand = True
    else:
        new_state.fallTimer = 0
        new_state.willHardLand = False
        new_state.airDashed = False

    # hard landing pause
    if new_state.hardLandingTimer > 0:
        new_state.hardLandingTimer -= duration
    elif new_state.recoilTimer > 0:
        # if recoiling, update timer or cancel

        new_state.recoilTimer -= duration
        # if new_state.recoilTimer <= 0:
        #     reset_motion(new_state)
    else:
        # Input:
        #    if pressing against wall and has wall jump, start slide

        #    cancel slide with down
        if new_state.wallSliding and controls.down:
            new_state.wallSliding = False
        #    cancel wallLocked if holding away direction
        #    check if jump released --- cancel positive velocity and jump (assume jumpReleaseQueueingEnabled==false)
        if not controls.jump:
            if new_state.jump_time > 0:
                print('release', new_state.jump_time, new_state.y_velocity, JUMP_TIME)
            if new_state.y_velocity > 0 and new_state.jump_time < (JUMP_TIME - JUMP_TIME_MIN):
                new_state.y_velocity = 0
                new_state.jump_time = 0

        #    if dash/attack released, stop buffered input

        # run attack timer, stop attack after attackDuration, enable attack again after attackCooldown
        if new_state.attack_time > 0:
            new_state.attack_time -= duration
            if new_state.attack_time < (ATTACK_COOLDOWN_TIME - ATTACK_DURATION):
                new_state.attackDir = -1
        # run bounce timer, stop after bounceDuration
        if new_state.bounceTimer > 0:
            new_state.bounceTimer -= duration
            if new_state.bounceTimer <= 0:
                new_state.y_velocity = 0
                new_state.bounceTimer = 0
        # stop shroomBouncing when y velocity is nonpositive
        if new_state.shroomBouncing and new_state.y_velocity < 0:
            new_state.shroomBouncing = False

        # QueueInput:
        #    if jump pressed:
        if controls.jump:
            #        wall jump
            #        else jump
            if new_state.onGround:
                if new_state.jump_time <= 0:
                    new_state.jump_time = JUMP_TIME
                    new_state.onGround = False
            #        else doublejump
            # elif not new_state.doubleJumped:
            #     if new_state.jump_time <= 0:
            #         new_state.jump_time = DOUBLE_JUMP_TIME + DOUBLE_JUMP_DELAY
            #         new_state.doubleJumped = True
        else:
            new_state.jump_time = 0
        #    if dash pressed
        if controls.dash:
            #        dash
            if new_state.dash_timer <= 0:
                if not new_state.onGround:
                    new_state.airDashed = True
                new_state.dash_timer = DASH_COOLDOWN
        #    if attack pressed or queued
        if controls.attack:
            #        DoAttack:
            #           start cooldown timer
            if new_state.attack_time <= 0:
                new_state.attack_time = ATTACK_COOLDOWN_TIME
                if controls.up:
                    new_state.attackDir = 2
                elif controls.down:
                    new_state.attackDir = 3
                elif new_state.facingRight:
                    new_state.attackDir = 1
                else:
                    new_state.attackDir = 0
    #           check for terrain thunk --- recoil in reverse direction

    # cancel wall slide if grounded or not touching wall
    if new_state.onGround:
        new_state.wallSliding = False
    # run attack/dash cooldown timers
    # charge nail attack while attack pressed

    # ------------------------
    # run physics

    # move player by velocity * deltaTime
    # new_state.x_velocity = -2
    # new_x = new_state.x_pos + new_state.x_velocity * duration
    # new_y = new_state.y_pos + (
    #     new_state.y_velocity * duration - 0.5 * gravity * duration * duration
    # )

    new_state.y_velocity -= gravity
    hit, new_pos, new_vel, hit_info = environment.integrate_motion(
        np.array([old_state.x_pos, old_state.y_pos]), np.array([new_state.x_velocity, new_state.y_velocity]), duration
    )
    new_state.x_pos = new_pos[0]
    new_state.y_pos = new_pos[1]
    new_state.x_velocity = new_vel[0]
    new_state.y_velocity = new_vel[1]

    # handle collisions:
    touching = environment.find_touching_segments(new_pos)
    touching_types = [seg.type for seg in touching]
    new_state.onGround = "floor" in touching_types
    # new_state.wall
    
    #   if top collision:
    #       cancel jump/bounce/etc, kill vertical velocity
    #   if bottom collision:
    #       cancel down attack
    #       maybe start hard landing
    #       set grounded
    #   else:
    #       set ungrounded, start coyote time
    #   if slidable wall collision:
    #       if not dashing, not grounded, has wall jump, start sliding

    return new_state


def reset_motion(state: HeroControllerStates):
    state.jump_time = 0
    state.double_jump_time = 0
    state.dash_timer = 0
    state.bounceTimer = 0
    state.recoil_dx = 0
    state.recoil_dy = 0
    state.recoilTimer = 0
    state.wallSliding = 0
    state.x_velocity = 0
    state.y_velocity = 0
    state.currentWalljumpSpeed = 0
