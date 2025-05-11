using System;
using System.Collections;
using System.Collections.Generic;
using GlobalEnums;
using HutongGames.PlayMaker;
using Modding;
using UnityEngine;

public class HeroController : MonoBehaviour
{

	public float RUN_SPEED;

	public float RUN_SPEED_CH;

	public float RUN_SPEED_CH_COMBO;

	public float WALK_SPEED;

	public float UNDERWATER_SPEED;

	public float JUMP_SPEED;

	public float JUMP_SPEED_UNDERWATER;

	public float MIN_JUMP_SPEED;

	public int JUMP_STEPS;

	public int JUMP_STEPS_MIN;

	public int JUMP_TIME;

	public int DOUBLE_JUMP_STEPS;

	public int WJLOCK_STEPS_SHORT;

	public int WJLOCK_STEPS_LONG;

	public float WJ_KICKOFF_SPEED;

	public int WALL_STICKY_STEPS;

	public float DASH_SPEED;

	public float DASH_SPEED_SHARP;

	public float DASH_TIME;

	public int DASH_QUEUE_STEPS;

	public float BACK_DASH_SPEED;

	public float BACK_DASH_TIME;

	public float SHADOW_DASH_SPEED;

	public float SHADOW_DASH_TIME;

	public float SHADOW_DASH_COOLDOWN;

	public float SUPER_DASH_SPEED;

	public float DASH_COOLDOWN;

	public float DASH_COOLDOWN_CH;

	public float BACKDASH_COOLDOWN;

	public float WALLSLIDE_SPEED;

	public float WALLSLIDE_DECEL;

	public float NAIL_CHARGE_TIME_DEFAULT;

	public float NAIL_CHARGE_TIME_CHARM;

	public float CYCLONE_HORIZONTAL_SPEED;

	public float SWIM_ACCEL;

	public float SWIM_MAX_SPEED;

	public float TIME_TO_ENTER_SCENE_BOT;

	public float TIME_TO_ENTER_SCENE_HOR;

	public float SPEED_TO_ENTER_SCENE_HOR;

	public float SPEED_TO_ENTER_SCENE_UP;

	public float SPEED_TO_ENTER_SCENE_DOWN;

	public float DEFAULT_GRAVITY;

	public float UNDERWATER_GRAVITY;

	public float ATTACK_DURATION;

	public float ATTACK_DURATION_CH;

	public float ALT_ATTACK_RESET;

	public float ATTACK_RECOVERY_TIME;

	public float ATTACK_COOLDOWN_TIME;

	public float ATTACK_COOLDOWN_TIME_CH;

	public float BOUNCE_TIME;

	public float BOUNCE_SHROOM_TIME;

	public float BOUNCE_VELOCITY;

	public float SHROOM_BOUNCE_VELOCITY;

	public float RECOIL_HOR_TIME;

	public float RECOIL_HOR_VELOCITY;

	public float RECOIL_HOR_VELOCITY_LONG;

	public float RECOIL_HOR_STEPS;

	public float RECOIL_DOWN_VELOCITY;

	public float RUN_PUFF_TIME;

	public float BIG_FALL_TIME;

	public float HARD_LANDING_TIME;

	public float DOWN_DASH_TIME;

	public float MAX_FALL_VELOCITY;

	public float MAX_FALL_VELOCITY_UNDERWATER;

	public float RECOIL_DURATION;

	public float RECOIL_DURATION_STAL;

	public float RECOIL_VELOCITY;

	public float DAMAGE_FREEZE_DOWN;

	public float DAMAGE_FREEZE_WAIT;

	public float DAMAGE_FREEZE_UP;

	public float INVUL_TIME;

	public float INVUL_TIME_STAL;

	public float INVUL_TIME_PARRY;

	public float INVUL_TIME_QUAKE;

	public float INVUL_TIME_CYCLONE;

	public float CAST_TIME;

	public float CAST_RECOIL_TIME;

	public float CAST_RECOIL_VELOCITY;

	public float WALLSLIDE_CLIP_DELAY;

	public int GRUB_SOUL_MP;

	public int GRUB_SOUL_MP_COMBO;

	private int JUMP_QUEUE_STEPS = 2;

	private int JUMP_RELEASE_QUEUE_STEPS = 2;

	private int DOUBLE_JUMP_QUEUE_STEPS = 10;

	private int ATTACK_QUEUE_STEPS = 5;

	private float DELAY_BEFORE_ENTER = 0.1f;

	private float LOOK_DELAY = 0.85f;

	private float LOOK_ANIM_DELAY = 0.25f;

	private float DEATH_WAIT = 2.85f;

	private float HAZARD_DEATH_CHECK_TIME = 3f;

	private float FLOATING_CHECK_TIME = 0.18f;

	private float NAIL_TERRAIN_CHECK_TIME = 0.12f;

	private float BUMP_VELOCITY = 4f;

	private float BUMP_VELOCITY_DASH = 5f;

	private int LANDING_BUFFER_STEPS = 5;

	private int LEDGE_BUFFER_STEPS = 2;

	private int HEAD_BUMP_STEPS = 3;

	private float MANTIS_CHARM_SCALE = 1.35f;

	private float FIND_GROUND_POINT_DISTANCE = 10f;

	private float FIND_GROUND_POINT_DISTANCE_EXT = 50f;

	// public ActorStates hero_state;

	// public float move_input;

	// public float vertical_input;

	// public Vector2 current_velocity;

	// private int jumped_steps;

	// private float dash_timer;

	// private float attack_time;

	// private float attack_cooldown;

	// private float bounceTimer;

	// private float hardLandingTimer;

	// private int cState.recoilSteps;

	// private float dashCooldownTimer;

	// private float nailChargeTimer;

	// private int wallLockSteps;

	// private float attackDuration;

	// ???
	// private Vector2 recoilVector;

	private GameManager gm;

	private Rigidbody2D rb2d;

	private Collider2D col2d;

	private new Transform transform;

	public HeroControllerStates cState;

	public PlayerData playerData;

	private InputHandler inputHandler;

	// ???
	// private bool hardLanded;

	private int jumpQueueSteps;

	private bool jumpQueuing;

	private int doubleJumpQueueSteps;

	private bool doubleJumpQueuing;

	private int jumpReleaseQueueSteps;

	private bool jumpReleaseQueuing;

	private int attackQueueSteps;

	private bool attackQueuing;

	// public bool touchingWallL;

	// public bool touchingWallR;

	// public bool wallSlidingL;

	// public bool wallSlidingR;

	// private bool airDashed;

	// private bool nailArt_cyclone;

	// private bool doubleJumped;

	// true when currentWalljumpSpeed != 0
	// public bool wallLocked;

	// private int ledgeBufferSteps;
	// private float currentWalljumpSpeed;

	// public float fallTimer { get; private set; }

	// constant
	// private float walljumpSpeedDecel;
	// private float nailChargeTime;

	// ignored
	// private bool recoilLarge;
	// public float conveyorSpeed;
	// public float conveyorSpeedV;

	private bool jumpReleaseQueueingEnabled;

	private void FixedUpdate()
	{
		if (cState.recoilingLeft || cState.recoilingRight)
		{
			if ((float)cState.recoilSteps <= RECOIL_HOR_STEPS)
			{
				cState.recoilSteps++;
			}
			else
			{
				CancelRecoilHorizontal();
			}
		}
		if ((hero_state == ActorStates.hard_landing) || hero_state == ActorStates.dash_landing)
		{
			ResetMotion();
		}
		else if (hero_state == ActorStates.no_input)
		{
			if (cState.recoiling)
			{
				AffectedByGravity(gravityApplies: false);
				rb2d.velocity = recoilVector;
			}
		}
		else if (hero_state != ActorStates.no_input)
		{
			if (hero_state == ActorStates.running)
			{
				if (move_input > 0f)
				{
					if (CheckForBump(CollisionSide.right))
					{
						rb2d.velocity = new Vector2(rb2d.velocity.x, BUMP_VELOCITY);
					}
				}
				else if (move_input < 0f && CheckForBump(CollisionSide.left))
				{
					rb2d.velocity = new Vector2(rb2d.velocity.x, BUMP_VELOCITY);
				}
			}
			if (!cState.backDashing && !cState.dashing)
			{
				Move(move_input);
				if ((!cState.attacking || !(attack_time < ATTACK_RECOVERY_TIME)) && !cState.wallSliding && !wallLocked)
				{
					if (move_input > 0f && !cState.facingRight)
					{
						FlipSprite();
						CancelAttack();
					}
					else if (move_input < 0f && cState.facingRight)
					{
						FlipSprite();
						CancelAttack();
					}
				}
				if (cState.recoilingLeft)
				{
					float num = ((!recoilLarge) ? RECOIL_HOR_VELOCITY : RECOIL_HOR_VELOCITY_LONG);
					if (rb2d.velocity.x > 0f - num)
					{
						rb2d.velocity = new Vector2(0f - num, rb2d.velocity.y);
					}
					else
					{
						rb2d.velocity = new Vector2(rb2d.velocity.x - num, rb2d.velocity.y);
					}
				}
				if (cState.recoilingRight)
				{
					float num2 = ((!recoilLarge) ? RECOIL_HOR_VELOCITY : RECOIL_HOR_VELOCITY_LONG);
					if (rb2d.velocity.x < num2)
					{
						rb2d.velocity = new Vector2(num2, rb2d.velocity.y);
					}
					else
					{
						rb2d.velocity = new Vector2(rb2d.velocity.x + num2, rb2d.velocity.y);
					}
				}
			}
			if (cState.jumping)
			{
				Jump();
			}
			if (cState.doubleJumping)
			{
				DoubleJump();
			}
			if (cState.dashing)
			{
				Dash();
			}
			if (cState.casting)
			{
				if (cState.castRecoiling)
				{
					if (cState.facingRight)
					{
						rb2d.velocity = new Vector2(0f - CAST_RECOIL_VELOCITY, 0f);
					}
					else
					{
						rb2d.velocity = new Vector2(CAST_RECOIL_VELOCITY, 0f);
					}
				}
				else
				{
					rb2d.velocity = Vector2.zero;
				}
			}
			if (cState.bouncing)
			{
				rb2d.velocity = new Vector2(rb2d.velocity.x, BOUNCE_VELOCITY);
			}
			_ = cState.shroomBouncing;
			if (wallLocked)
			{
				if (wallJumpedR)
				{
					rb2d.velocity = new Vector2(currentWalljumpSpeed, rb2d.velocity.y);
				}
				else if (wallJumpedL)
				{
					rb2d.velocity = new Vector2(0f - currentWalljumpSpeed, rb2d.velocity.y);
				}
				wallLockSteps++;
				if (wallLockSteps > WJLOCK_STEPS_LONG)
				{
					wallLocked = false;
				}
				currentWalljumpSpeed -= walljumpSpeedDecel;
			}
			if (cState.wallSliding)
			{
				if (wallSlidingL && inputHandler.inputActions.right.IsPressed)
				{
					wallUnstickSteps++;
				}
				else if (wallSlidingR && inputHandler.inputActions.left.IsPressed)
				{
					wallUnstickSteps++;
				}
				else
				{
					wallUnstickSteps = 0;
				}
				if (wallUnstickSteps >= WALL_STICKY_STEPS)
				{
					CancelWallsliding();
				}
				if (wallSlidingL)
				{
					if (!CheckStillTouchingWall(CollisionSide.left))
					{
						FlipSprite();
						CancelWallsliding();
					}
				}
				else if (wallSlidingR && !CheckStillTouchingWall(CollisionSide.right))
				{
					FlipSprite();
					CancelWallsliding();
				}
			}
		}
		if (rb2d.velocity.y < 0f - MAX_FALL_VELOCITY && !inAcid && !controlReqlinquished && !cState.shadowDashing && !cState.spellQuake)
		{
			rb2d.velocity = new Vector2(rb2d.velocity.x, 0f - MAX_FALL_VELOCITY);
		}
		if (jumpQueuing)
		{
			jumpQueueSteps++;
		}
		if (doubleJumpQueuing)
		{
			doubleJumpQueueSteps++;
		}
		if (dashQueuing)
		{
			dashQueueSteps++;
		}
		if (attackQueuing)
		{
			attackQueueSteps++;
		}
		if (cState.wallSliding)
		{
			if (rb2d.velocity.y > WALLSLIDE_SPEED)
			{
				rb2d.velocity = new Vector3(rb2d.velocity.x, rb2d.velocity.y - WALLSLIDE_DECEL);
				if (rb2d.velocity.y < WALLSLIDE_SPEED)
				{
					rb2d.velocity = new Vector3(rb2d.velocity.x, WALLSLIDE_SPEED);
				}
			}
			if (rb2d.velocity.y < WALLSLIDE_SPEED)
			{
				rb2d.velocity = new Vector3(rb2d.velocity.x, rb2d.velocity.y + WALLSLIDE_DECEL);
				if (rb2d.velocity.y < WALLSLIDE_SPEED)
				{
					rb2d.velocity = new Vector3(rb2d.velocity.x, WALLSLIDE_SPEED);
				}
			}
		}
		if (nailArt_cyclone)
		{
			if (inputHandler.inputActions.right.IsPressed && !inputHandler.inputActions.left.IsPressed)
			{
				rb2d.velocity = new Vector3(CYCLONE_HORIZONTAL_SPEED, rb2d.velocity.y);
			}
			else if (inputHandler.inputActions.left.IsPressed && !inputHandler.inputActions.right.IsPressed)
			{
				rb2d.velocity = new Vector3(0f - CYCLONE_HORIZONTAL_SPEED, rb2d.velocity.y);
			}
			else
			{
				rb2d.velocity = new Vector3(0f, rb2d.velocity.y);
			}
		}
		if (cState.swimming)
		{
			rb2d.velocity = new Vector3(rb2d.velocity.x, rb2d.velocity.y + SWIM_ACCEL);
			if (rb2d.velocity.y > SWIM_MAX_SPEED)
			{
				rb2d.velocity = new Vector3(rb2d.velocity.x, SWIM_MAX_SPEED);
			}
		}
		if (cState.superDashOnWall)
		{
			rb2d.velocity = new Vector3(0f, 0f);
		}
		if (landingBufferSteps > 0)
		{
			landingBufferSteps--;
		}
		if (ledgeBufferSteps > 0)
		{
			ledgeBufferSteps--;
		}
		if (jumpReleaseQueueSteps > 0)
		{
			jumpReleaseQueueSteps--;
		}
		cState.wasOnGround = cState.onGround;
	}

	private void Move(float move_direction)
	{
		if (cState.onGround)
		{
			SetState(ActorStates.grounded);
		}
		if (!cState.wallSliding)
		{
			if (inAcid)
			{
				rb2d.velocity = new Vector2(move_direction * UNDERWATER_SPEED, rb2d.velocity.y);
			}
			else if (playerData.GetBool("equippedCharm_37") && cState.onGround && playerData.GetBool("equippedCharm_31"))
			{
				rb2d.velocity = new Vector2(move_direction * RUN_SPEED_CH_COMBO, rb2d.velocity.y);
			}
			else if (playerData.GetBool("equippedCharm_37") && cState.onGround)
			{
				rb2d.velocity = new Vector2(move_direction * RUN_SPEED_CH, rb2d.velocity.y);
			}
			else
			{
				rb2d.velocity = new Vector2(move_direction * RUN_SPEED, rb2d.velocity.y);
			}
		}
	}

	private void Jump()
	{
		if (jumped_steps <= JUMP_STEPS)
		{
			if (inAcid)
			{
				rb2d.velocity = new Vector2(rb2d.velocity.x, JUMP_SPEED_UNDERWATER);
			}
			else
			{
				rb2d.velocity = new Vector2(rb2d.velocity.x, JUMP_SPEED);
			}
			jumped_steps++;
			ledgeBufferSteps = 0;
		}
		else
		{
			CancelJump();
		}
	}

	private void DoubleJump()
	{
		if (jumped_steps <= DOUBLE_JUMP_STEPS)
		{
			if (jumped_steps > 3)
			{
				rb2d.velocity = new Vector2(rb2d.velocity.x, JUMP_SPEED * 1.1f);
			}
			jumped_steps++;
		}
		else
		{
			CancelDoubleJump();
		}
		if (cState.onGround)
		{
			CancelDoubleJump();
		}
	}

	public void Attack(AttackDirection attackDir)
	{
		cState.attacking = true;
		if (playerData.GetBool("equippedCharm_32"))
		{
			attackDuration = ATTACK_DURATION_CH;
		}
		else
		{
			attackDuration = ATTACK_DURATION;
		}
		if (cState.wallSliding)
		{
			if (cState.facingRight)
			{
				slashFsm.FsmVariables.GetFsmFloat("direction").Value = 180f;
			}
			else
			{
				slashFsm.FsmVariables.GetFsmFloat("direction").Value = 0f;
			}
		}
		else if (attackDir == AttackDirection.normal && cState.facingRight)
		{
			slashFsm.FsmVariables.GetFsmFloat("direction").Value = 0f;
		}
		else if (attackDir == AttackDirection.normal && !cState.facingRight)
		{
			slashFsm.FsmVariables.GetFsmFloat("direction").Value = 180f;
		}
		else
		{
			switch (attackDir)
			{
				case AttackDirection.upward:
					slashFsm.FsmVariables.GetFsmFloat("direction").Value = 90f;
					break;
				case AttackDirection.downward:
					slashFsm.FsmVariables.GetFsmFloat("direction").Value = 270f;
					break;
			}
		}
	}

	private void Dash()
	{
		AffectedByGravity(gravityApplies: false);
		ResetHardLandingTimer();
		if (dash_timer > DASH_TIME)
		{
			FinishedDashing();
			return;
		}
		Vector2 change = OrigDashVector();
		change = ModHooks.DashVelocityChange(change);
		rb2d.velocity = change;
		dash_timer += Time.deltaTime;
	}

	public void FaceRight()
	{
		cState.facingRight = true;
	}

	public void FaceLeft()
	{
		cState.facingRight = false;
	}

	public void SetBackOnGround()
	{
		cState.onGround = true;
	}

	public void IsSwimming()
	{
		cState.swimming = true;
	}

	public void NotSwimming()
	{
		cState.swimming = false;
	}

	public void ResetAirMoves()
	{
		doubleJumped = false;
		airDashed = false;
	}

	public void CancelHeroJump()
	{
		if (cState.jumping)
		{
			CancelJump();
			CancelDoubleJump();
			if (rb2d.velocity.y > 0f)
			{
				rb2d.velocity = new Vector2(rb2d.velocity.x, 0f);
			}
		}
	}

	public void FlipSprite()
	{
		cState.facingRight = !cState.facingRight;
	}

	public void TakeDamage(GameObject go, CollisionSide damageSide, int damageAmount, int hazardType)
	{
		if ((damageMode == DamageMode.HAZARD_ONLY && hazardType == 1) || (cState.shadowDashing && hazardType == 1) || (parryInvulnTimer > 0f && hazardType == 1))
		{
			return;
		}
		CancelAttack();
		if (cState.wallSliding)
		{
			cState.wallSliding = false;
		}
		if (cState.touchingWall)
		{
			cState.touchingWall = false;
		}
		if (cState.recoilingLeft || cState.recoilingRight)
		{
			CancelRecoilHorizontal();
		}
		if (cState.bouncing)
		{
			CancelBounce();
			rb2d.velocity = new Vector2(rb2d.velocity.x, 0f);
		}
		if (cState.shroomBouncing)
		{
			CancelBounce();
			rb2d.velocity = new Vector2(rb2d.velocity.x, 0f);
		}
		if (cState.nailCharging || nailChargeTimer != 0f)
		{
			cState.nailCharging = false;
			nailChargeTimer = 0f;
		}
		switch (hazardType)
		{
			// case 2, 3, 4, 5: diefromhazard
			default:
				StartCoroutine(StartRecoil(damageSide, spawnDamageEffect, damageAmount));
				break;
		}
	}

	public void Bounce()
	{
		if (!cState.bouncing && !cState.shroomBouncing && !controlReqlinquished)
		{
			doubleJumped = false;
			airDashed = false;
			cState.bouncing = true;
		}
	}

	public void BounceHigh()
	{
		if (!cState.bouncing && !controlReqlinquished)
		{
			doubleJumped = false;
			airDashed = false;
			cState.bouncing = true;
			bounceTimer = -0.03f;
			rb2d.velocity = new Vector2(rb2d.velocity.x, BOUNCE_VELOCITY);
		}
	}

	public void ShroomBounce()
	{
		doubleJumped = false;
		airDashed = false;
		cState.bouncing = false;
		cState.shroomBouncing = true;
		rb2d.velocity = new Vector2(rb2d.velocity.x, SHROOM_BOUNCE_VELOCITY);
	}

	public void RecoilLeft()
	{
		if (!cState.recoilingLeft && !cState.recoilingRight && !playerData.GetBool("equippedCharm_14"))
		{
			CancelDash();
			cState.recoilSteps = 0;
			cState.recoilingLeft = true;
			cState.recoilingRight = false;
			recoilLarge = false;
			rb2d.velocity = new Vector2(0f - RECOIL_HOR_VELOCITY, rb2d.velocity.y);
		}
	}

	public void RecoilRight()
	{
		if (!cState.recoilingLeft && !cState.recoilingRight && !playerData.GetBool("equippedCharm_14"))
		{
			CancelDash();
			cState.recoilSteps = 0;
			cState.recoilingRight = true;
			cState.recoilingLeft = false;
			recoilLarge = false;
			rb2d.velocity = new Vector2(RECOIL_HOR_VELOCITY, rb2d.velocity.y);
		}
	}

	public void RecoilDown()
	{
		CancelJump();
		if (rb2d.velocity.y > RECOIL_DOWN_VELOCITY && !controlReqlinquished)
		{
			rb2d.velocity = new Vector2(rb2d.velocity.x, RECOIL_DOWN_VELOCITY);
		}
	}

	public void StartCyclone()
	{
		nailArt_cyclone = true;
	}

	public void EndCyclone()
	{
		nailArt_cyclone = false;
	}

	public void ResetHardLandingTimer()
	{
		cState.willHardLand = false;
		hardLandingTimer = 0f;
		fallTimer = 0f;
		hardLanded = false;
	}

	public bool CanCast()
	{
		if (!gm.isPaused && !cState.dashing && hero_state != ActorStates.no_input && !cState.backDashing && (!cState.attacking || !(attack_time < ATTACK_RECOVERY_TIME)) && !cState.recoiling && !cState.recoilFrozen && !cState.transitioning && !cState.hazardDeath && !cState.hazardRespawning && CanInput() && preventCastByDialogueEndTimer <= 0f)
		{
			return true;
		}
		return false;
	}

	public bool CanFocus()
	{
		if (!gm.isPaused && hero_state != ActorStates.no_input && !cState.dashing && !cState.backDashing && (!cState.attacking || !(attack_time < ATTACK_RECOVERY_TIME)) && !cState.recoiling && cState.onGround && !cState.transitioning && !cState.recoilFrozen && !cState.hazardDeath && !cState.hazardRespawning && CanInput())
		{
			return true;
		}
		return false;
	}

	public bool CanNailArt()
	{
		if (!cState.transitioning && hero_state != ActorStates.no_input && !cState.attacking && !cState.hazardDeath && !cState.hazardRespawning && nailChargeTimer >= nailChargeTime)
		{
			nailChargeTimer = 0f;
			return true;
		}
		nailChargeTimer = 0f;
		return false;
	}

	public bool CanQuickMap()
	{
		if (!gm.isPaused && !controlReqlinquished && hero_state != ActorStates.no_input && !cState.onConveyor && !cState.dashing && !cState.backDashing && (!cState.attacking || !(attack_time < ATTACK_RECOVERY_TIME)) && !cState.recoiling && !cState.transitioning && !cState.hazardDeath && !cState.hazardRespawning && !cState.recoilFrozen && cState.onGround && CanInput())
		{
			return true;
		}
		return false;
	}

	public bool CanInspect()
	{
		if (!gm.isPaused && !cState.dashing && hero_state != ActorStates.no_input && !cState.backDashing && (!cState.attacking || !(attack_time < ATTACK_RECOVERY_TIME)) && !cState.recoiling && !cState.transitioning && !cState.hazardDeath && !cState.hazardRespawning && !cState.recoilFrozen && cState.onGround && CanInput())
		{
			return true;
		}
		return false;
	}

	public bool CanBackDash()
	{
		if (!gm.isPaused && !cState.dashing && hero_state != ActorStates.no_input && !cState.backDashing && (!cState.attacking || !(attack_time < ATTACK_RECOVERY_TIME)) && !cState.preventBackDash && !cState.backDashCooldown && !controlReqlinquished && !cState.recoilFrozen && !cState.recoiling && !cState.transitioning && cState.onGround && playerData.GetBool("canBackDash"))
		{
			return true;
		}
		return false;
	}

	public bool CanSuperDash()
	{
		if (!gm.isPaused && hero_state != ActorStates.no_input && !cState.dashing && !cState.hazardDeath && !cState.hazardRespawning && !cState.backDashing && (!cState.attacking || !(attack_time < ATTACK_RECOVERY_TIME)) && !cState.slidingLeft && !cState.slidingRight && !controlReqlinquished && !cState.recoilFrozen && !cState.recoiling && !cState.transitioning && playerData.GetBool("hasSuperDash") && (cState.onGround || cState.wallSliding))
		{
			return true;
		}
		return false;
	}

	public bool CanDreamNail()
	{
		if (!gm.isPaused && hero_state != ActorStates.no_input && !cState.dashing && !cState.backDashing && (!cState.attacking || !(attack_time < ATTACK_RECOVERY_TIME)) && !controlReqlinquished && !cState.hazardDeath && rb2d.velocity.y > -0.1f && !cState.hazardRespawning && !cState.recoilFrozen && !cState.recoiling && !cState.transitioning && playerData.GetBool("hasDreamNail") && cState.onGround)
		{
			return true;
		}
		return false;
	}

	public void AffectedByGravity(bool gravityApplies)
	{
		_ = rb2d.gravityScale;
		if (rb2d.gravityScale > Mathf.Epsilon && !gravityApplies)
		{
			prevGravityScale = rb2d.gravityScale;
			rb2d.gravityScale = 0f;
		}
		else if (rb2d.gravityScale <= Mathf.Epsilon && gravityApplies)
		{
			rb2d.gravityScale = prevGravityScale;
			prevGravityScale = 0f;
		}
	}

	private void LookForInput()
	{
		if (!acceptingInput || gm.isPaused || !isGameplayScene)
		{
			return;
		}
		move_input = inputHandler.inputActions.moveVector.Vector.x;
		vertical_input = inputHandler.inputActions.moveVector.Vector.y;
		FilterInput();
		if (playerData.GetBool("hasWalljump") && CanWallSlide() && !cState.attacking)
		{
			if (touchingWallL && inputHandler.inputActions.left.IsPressed && !cState.wallSliding)
			{
				airDashed = false;
				doubleJumped = false;
				cState.wallSliding = true;
				cState.willHardLand = false;
				wallSlidingL = true;
				wallSlidingR = false;
				FaceLeft();
			}
			if (touchingWallR && inputHandler.inputActions.right.IsPressed && !cState.wallSliding)
			{
				airDashed = false;
				doubleJumped = false;
				cState.wallSliding = true;
				cState.willHardLand = false;
				wallSlidingL = false;
				wallSlidingR = true;
				FaceRight();
			}
		}
		if (cState.wallSliding && inputHandler.inputActions.down.WasPressed)
		{
			CancelWallsliding();
			FlipSprite();
		}
		if (wallLocked && wallJumpedL && inputHandler.inputActions.right.IsPressed && wallLockSteps >= WJLOCK_STEPS_SHORT)
		{
			wallLocked = false;
		}
		if (wallLocked && wallJumpedR && inputHandler.inputActions.left.IsPressed && wallLockSteps >= WJLOCK_STEPS_SHORT)
		{
			wallLocked = false;
		}
		if (inputHandler.inputActions.jump.WasReleased && jumpReleaseQueueingEnabled)
		{
			jumpReleaseQueueSteps = JUMP_RELEASE_QUEUE_STEPS;
			jumpReleaseQueuing = true;
		}
		if (!inputHandler.inputActions.jump.IsPressed)
		{
			JumpReleased();
		}
		if (!inputHandler.inputActions.dash.IsPressed)
		{
			if (cState.preventDash && !cState.dashCooldown)
			{
				cState.preventDash = false;
			}
			dashQueuing = false;
		}
		if (!inputHandler.inputActions.attack.IsPressed)
		{
			attackQueuing = false;
		}
	}

	private void LookForQueueInput()
	{
		if (!acceptingInput || gm.isPaused || !isGameplayScene)
		{
			return;
		}
		if (inputHandler.inputActions.jump.WasPressed)
		{
			if (CanWallJump())
			{
				DoWallJump();
			}
			else if (CanJump())
			{
				HeroJump();
			}
			else if (CanDoubleJump())
			{
				DoDoubleJump();
			}
			else if (CanInfiniteAirJump())
			{
				CancelJump();
				audioCtrl.PlaySound(HeroSounds.JUMP);
				ResetLook();
				cState.jumping = true;
			}
			else
			{
				jumpQueueSteps = 0;
				jumpQueuing = true;
				doubleJumpQueueSteps = 0;
				doubleJumpQueuing = true;
			}
		}
		if (inputHandler.inputActions.dash.WasPressed && !ModHooks.OnDashPressed())
		{
			if (CanDash())
			{
				HeroDash();
			}
			else
			{
				dashQueueSteps = 0;
				dashQueuing = true;
			}
		}
		if (inputHandler.inputActions.attack.WasPressed)
		{
			if (CanAttack())
			{
				DoAttack();
			}
			else
			{
				attackQueueSteps = 0;
				attackQueuing = true;
			}
		}
		if (inputHandler.inputActions.jump.IsPressed)
		{
			if (jumpQueueSteps <= JUMP_QUEUE_STEPS && CanJump() && jumpQueuing)
			{
				HeroJump();
			}
			else if (doubleJumpQueueSteps <= DOUBLE_JUMP_QUEUE_STEPS && CanDoubleJump() && doubleJumpQueuing)
			{
				if (cState.onGround)
				{
					HeroJump();
				}
				else
				{
					DoDoubleJump();
				}
			}
			if (CanSwim())
			{
				if (hero_state != ActorStates.airborne)
				{
					SetState(ActorStates.airborne);
				}
				cState.swimming = true;
			}
		}
		if (inputHandler.inputActions.dash.IsPressed && dashQueueSteps <= DASH_QUEUE_STEPS && CanDash() && dashQueuing && !ModHooks.OnDashPressed() && CanDash())
		{
			HeroDash();
		}
		if (inputHandler.inputActions.attack.IsPressed && attackQueueSteps <= ATTACK_QUEUE_STEPS && CanAttack() && attackQueuing)
		{
			DoAttack();
		}
	}

	private void HeroJump()
	{
		cState.recoiling = false;
		cState.jumping = true;
		jumpQueueSteps = 0;
		jumped_steps = 0;
		doubleJumpQueuing = false;
	}

	private void DoWallJump()
	{
		if (touchingWallL)
		{
			FaceRight();
			wallJumpedR = true;
			wallJumpedL = false;
		}
		else if (touchingWallR)
		{
			FaceLeft();
			wallJumpedR = false;
			wallJumpedL = true;
		}
		CancelWallsliding();
		cState.touchingWall = false;
		touchingWallL = false;
		touchingWallR = false;
		airDashed = false;
		doubleJumped = false;
		currentWalljumpSpeed = WJ_KICKOFF_SPEED;
		walljumpSpeedDecel = (WJ_KICKOFF_SPEED - RUN_SPEED) / (float)WJLOCK_STEPS_LONG;
		dashBurst.SendEvent("CANCEL");
		cState.jumping = true;
		wallLockSteps = 0;
		wallLocked = true;
		jumpQueueSteps = 0;
		jumped_steps = 0;
	}

	private void DoDoubleJump()
	{
		cState.jumping = false;
		cState.doubleJumping = true;
		jumped_steps = 0;
		doubleJumped = true;
	}

	private void DoHardLanding()
	{
		AffectedByGravity(gravityApplies: true);
		ResetInput();
		SetState(ActorStates.hard_landing);
		CancelAttack();
		hardLanded = true;
		audioCtrl.PlaySound(HeroSounds.HARD_LANDING);
		hardLandingEffectPrefab.Spawn(transform.position);
	}

	private void DoAttack()
	{
		ModHooks.OnDoAttack();
		orig_DoAttack();
	}

	private void HeroDash()
	{
		if (!cState.onGround && !inAcid)
		{
			airDashed = true;
		}
		ResetAttacksDash();
		CancelBounce();
		audioCtrl.StopSound(HeroSounds.FOOTSTEPS_RUN);
		audioCtrl.StopSound(HeroSounds.FOOTSTEPS_WALK);
		audioCtrl.PlaySound(HeroSounds.DASH);
		ResetLook();
		cState.recoiling = false;
		if (cState.wallSliding)
		{
			FlipSprite();
		}
		else if (inputHandler.inputActions.right.IsPressed)
		{
			FaceRight();
		}
		else if (inputHandler.inputActions.left.IsPressed)
		{
			FaceLeft();
		}
		cState.dashing = true;
		dashQueueSteps = 0;
		HeroActions inputActions = inputHandler.inputActions;
		if (inputActions.down.IsPressed && !cState.onGround && playerData.GetBool("equippedCharm_31") && !inputActions.left.IsPressed && !inputActions.right.IsPressed)
		{
			dashBurst.transform.localPosition = new Vector3(-0.07f, 3.74f, 0.01f);
			dashBurst.transform.localEulerAngles = new Vector3(0f, 0f, 90f);
			dashingDown = true;
		}
		else
		{
			dashBurst.transform.localPosition = new Vector3(4.11f, -0.55f, 0.001f);
			dashBurst.transform.localEulerAngles = new Vector3(0f, 0f, 0f);
			dashingDown = false;
		}
		if (playerData.GetBool("equippedCharm_31"))
		{
			dashCooldownTimer = DASH_COOLDOWN_CH;
		}
		else
		{
			dashCooldownTimer = DASH_COOLDOWN;
		}
		if (playerData.GetBool("hasShadowDash") && shadowDashTimer <= 0f)
		{
			shadowDashTimer = SHADOW_DASH_COOLDOWN;
			cState.shadowDashing = true;
			if (playerData.GetBool("equippedCharm_16"))
			{
				audioSource.PlayOneShot(sharpShadowClip, 1f);
				sharpShadowPrefab.SetActive(value: true);
			}
			else
			{
				audioSource.PlayOneShot(shadowDashClip, 1f);
			}
		}
		if (cState.shadowDashing)
		{
			if (dashingDown)
			{
				dashEffect = shadowdashDownBurstPrefab.Spawn(new Vector3(transform.position.x, transform.position.y + 3.5f, transform.position.z + 0.00101f));
				dashEffect.transform.localEulerAngles = new Vector3(0f, 0f, 90f);
			}
			else if (transform.localScale.x > 0f)
			{
				dashEffect = shadowdashBurstPrefab.Spawn(new Vector3(transform.position.x + 5.21f, transform.position.y - 0.58f, transform.position.z + 0.00101f));
				dashEffect.transform.localScale = new Vector3(1.919591f, dashEffect.transform.localScale.y, dashEffect.transform.localScale.z);
			}
			else
			{
				dashEffect = shadowdashBurstPrefab.Spawn(new Vector3(transform.position.x - 5.21f, transform.position.y - 0.58f, transform.position.z + 0.00101f));
				dashEffect.transform.localScale = new Vector3(-1.919591f, dashEffect.transform.localScale.y, dashEffect.transform.localScale.z);
			}
			shadowRechargePrefab.SetActive(value: true);
			FSMUtility.LocateFSM(shadowRechargePrefab, "Recharge Effect").SendEvent("RESET");
			shadowdashParticlesPrefab.GetComponent<ParticleSystem>().enableEmission = true;
			VibrationManager.PlayVibrationClipOneShot(shadowDashVibration);
			shadowRingPrefab.Spawn(transform.position);
		}
		else
		{
			dashBurst.SendEvent("PLAY");
			dashParticlesPrefab.GetComponent<ParticleSystem>().enableEmission = true;
			VibrationManager.PlayVibrationClipOneShot(dashVibration);
		}
		if (cState.onGround && !cState.shadowDashing)
		{
			dashEffect = backDashPrefab.Spawn(transform.position);
			dashEffect.transform.localScale = new Vector3(transform.localScale.x * -1f, transform.localScale.y, transform.localScale.z);
		}
	}

	private IEnumerator StartRecoil(CollisionSide impactSide, bool spawnDamageEffect, int damageAmount)
	{
		if (cState.recoiling)
		{
			yield break;
		}
		playerData.SetBoolSwappedArgs(value: true, "disablePause");
		ResetMotion();
		AffectedByGravity(gravityApplies: false);
		switch (impactSide)
		{
			case CollisionSide.left:
				recoilVector = new Vector2(RECOIL_VELOCITY, RECOIL_VELOCITY * 0.5f);
				if (cState.facingRight)
				{
					FlipSprite();
				}
				break;
			case CollisionSide.right:
				recoilVector = new Vector2(0f - RECOIL_VELOCITY, RECOIL_VELOCITY * 0.5f);
				if (!cState.facingRight)
				{
					FlipSprite();
				}
				break;
			default:
				recoilVector = Vector2.zero;
				break;
		}
		SetState(ActorStates.no_input);
		cState.recoilFrozen = true;
		if (spawnDamageEffect)
		{
			damageEffectFSM.SendEvent("DAMAGE");
			if (damageAmount > 1)
			{
				UnityEngine.Object.Instantiate(takeHitDoublePrefab, transform.position, transform.rotation);
			}
		}
		if (playerData.GetBool("equippedCharm_4"))
		{
			StartCoroutine(Invulnerable(INVUL_TIME_STAL));
		}
		else
		{
			StartCoroutine(Invulnerable(INVUL_TIME));
		}
		yield return takeDamageCoroutine = StartCoroutine(gm.FreezeMoment(DAMAGE_FREEZE_DOWN, DAMAGE_FREEZE_WAIT, DAMAGE_FREEZE_UP, 0.0001f));
		cState.recoilFrozen = false;
		cState.recoiling = true;
		playerData.SetBoolSwappedArgs(value: false, "disablePause");
	}

	private void FallCheck()
	{
		if (rb2d.velocity.y <= -1E-06f)
		{
			if (CheckTouchingGround())
			{
				return;
			}
			cState.falling = true;
			cState.onGround = false;
			cState.wallJumping = false;
			if (hero_state != ActorStates.no_input)
			{
				SetState(ActorStates.airborne);
			}
			if (cState.wallSliding)
			{
				fallTimer = 0f;
			}
			else
			{
				fallTimer += Time.deltaTime;
			}
			if (fallTimer > BIG_FALL_TIME)
			{
				if (!cState.willHardLand)
				{
					cState.willHardLand = true;
				}
				if (!fallRumble)
				{
					StartFallRumble();
				}
			}
			if (fallCheckFlagged)
			{
				fallCheckFlagged = false;
			}
		}
		else
		{
			cState.falling = false;
			fallTimer = 0f;
		}
	}

	private void CancelJump()
	{
		cState.jumping = false;
		jumpReleaseQueuing = false;
		jumped_steps = 0;
	}

	private void CancelDoubleJump()
	{
		cState.doubleJumping = false;
		jumped_steps = 0;
	}

	private void CancelDash()
	{
		if (cState.shadowDashing)
		{
			cState.shadowDashing = false;
		}
		cState.dashing = false;
		dash_timer = 0f;
		AffectedByGravity(gravityApplies: true);
	}

	private void CancelWallsliding()
	{
		cState.wallSliding = false;
		wallSlidingL = false;
		wallSlidingR = false;
		touchingWallL = false;
		touchingWallR = false;
	}

	private void CancelBackDash()
	{
		cState.backDashing = false;
		back_dash_timer = 0f;
	}

	private void CancelDownAttack()
	{
		if (cState.downAttacking)
		{
			slashComponent.CancelAttack();
			ResetAttacks();
		}
	}

	private void CancelAttack()
	{
		if (cState.attacking)
		{
			slashComponent.CancelAttack();
			ResetAttacks();
		}
	}

	private void CancelBounce()
	{
		cState.bouncing = false;
		cState.shroomBouncing = false;
		bounceTimer = 0f;
	}

	private void CancelRecoilHorizontal()
	{
		cState.recoilingLeft = false;
		cState.recoilingRight = false;
		cState.recoilSteps = 0;
	}

	private void CancelDamageRecoil()
	{
		cState.recoiling = false;
		recoilTimer = 0f;
		ResetMotion();
		AffectedByGravity(gravityApplies: true);
	}

	private void ResetAttacks()
	{
		cState.nailCharging = false;
		nailChargeTimer = 0f;
		cState.attacking = false;
		cState.upAttacking = false;
		cState.downAttacking = false;
		attack_time = 0f;
	}

	private void ResetAttacksDash()
	{
		cState.attacking = false;
		cState.upAttacking = false;
		cState.downAttacking = false;
		attack_time = 0f;
	}

	private void ResetMotion()
	{
		CancelJump();
		CancelDoubleJump();
		CancelDash();
		CancelBackDash();
		CancelBounce();
		CancelRecoilHorizontal();
		CancelWallsliding();
		rb2d.velocity = Vector2.zero;
		transition_vel = Vector2.zero;
		wallLocked = false;
		nailChargeTimer = 0f;
	}

	private void BackOnGround()
	{
		if (landingBufferSteps <= 0)
		{
			landingBufferSteps = LANDING_BUFFER_STEPS;
		}
		cState.falling = false;
		fallTimer = 0f;
		dashLandingTimer = 0f;
		cState.willHardLand = false;
		hardLandingTimer = 0f;
		hardLanded = false;
		jumped_steps = 0;
		if (cState.doubleJumping)
		{
			HeroJump();
		}
		SetState(ActorStates.grounded);
		cState.onGround = true;
		airDashed = false;
		doubleJumped = false;
	}

	private void JumpReleased()
	{
		if (rb2d.velocity.y > 0f && jumped_steps >= JUMP_STEPS_MIN && !inAcid && !cState.shroomBouncing)
		{
			if (jumpReleaseQueueingEnabled)
			{
				if (jumpReleaseQueuing && jumpReleaseQueueSteps <= 0)
				{
					rb2d.velocity = new Vector2(rb2d.velocity.x, 0f);
					CancelJump();
				}
			}
			else
			{
				rb2d.velocity = new Vector2(rb2d.velocity.x, 0f);
				CancelJump();
			}
		}
		jumpQueuing = false;
		doubleJumpQueuing = false;
		if (cState.swimming)
		{
			cState.swimming = false;
		}
	}

	private void FinishedDashing()
	{
		CancelDash();
		AffectedByGravity(gravityApplies: true);
		if (cState.touchingWall && !cState.onGround && (playerData.GetBool("hasWalljump") & (touchingWallL || touchingWallR)))
		{
			cState.wallSliding = true;
			cState.willHardLand = false;
			if (touchingWallL)
			{
				wallSlidingL = true;
			}
			if (touchingWallR)
			{
				wallSlidingR = true;
			}
			if (dashingDown)
			{
				FlipSprite();
			}
		}
	}

	private bool CheckStillTouchingWall(CollisionSide side, bool checkTop = false)
	{
		// check wall collision
		return false;
	}

	public bool CheckForBump(CollisionSide side)
	{
		// check for ground elevation
		return false;
	}

	public bool CheckTouchingGround()
	{
		// check ground under feet
		return false;
	}

	private bool CanJump()
	{
		if (hero_state != ActorStates.no_input && hero_state != ActorStates.hard_landing && hero_state != ActorStates.dash_landing && !cState.wallSliding && !cState.dashing && !cState.backDashing && !cState.jumping && !cState.bouncing && !cState.shroomBouncing)
		{
			if (cState.onGround)
			{
				return true;
			}
			if (ledgeBufferSteps > 0)
			{
				ledgeBufferSteps = 0;
				return true;
			}
			return false;
		}
		return false;
	}

	private bool CanDoubleJump()
	{
		if (playerData.GetBool("hasDoubleJump") && !controlReqlinquished && !doubleJumped && !inAcid && hero_state != ActorStates.no_input && hero_state != ActorStates.hard_landing && hero_state != ActorStates.dash_landing && !cState.dashing && !cState.wallSliding && !cState.backDashing && !cState.attacking && !cState.bouncing && !cState.shroomBouncing && !cState.onGround)
		{
			return true;
		}
		return false;
	}

	private bool CanInfiniteAirJump()
	{
		if (playerData.GetBool("infiniteAirJump") && hero_state != ActorStates.hard_landing && !cState.onGround)
		{
			return true;
		}
		return false;
	}

	private bool CanSwim()
	{
		if (hero_state != ActorStates.no_input && hero_state != ActorStates.hard_landing && hero_state != ActorStates.dash_landing && !cState.attacking && !cState.dashing && !cState.jumping && !cState.bouncing && !cState.shroomBouncing && !cState.onGround && inAcid)
		{
			return true;
		}
		return false;
	}

	private bool CanDash()
	{
		if (hero_state != ActorStates.no_input && hero_state != ActorStates.hard_landing && hero_state != ActorStates.dash_landing && dashCooldownTimer <= 0f && !cState.dashing && !cState.backDashing && (!cState.attacking || !(attack_time < ATTACK_RECOVERY_TIME)) && !cState.preventDash && (cState.onGround || !airDashed || cState.wallSliding) && !cState.hazardDeath && playerData.GetBool("canDash"))
		{
			return true;
		}
		return false;
	}

	private bool CanAttack()
	{
		if (attack_cooldown <= 0f && !cState.attacking && !cState.dashing && !cState.dead && !cState.hazardDeath && !cState.hazardRespawning && !controlReqlinquished && hero_state != ActorStates.no_input && hero_state != ActorStates.hard_landing && hero_state != ActorStates.dash_landing)
		{
			return true;
		}
		return false;
	}

	private bool CanNailCharge()
	{
		if (!cState.attacking && !controlReqlinquished && !cState.recoiling && !cState.recoilingLeft && !cState.recoilingRight && playerData.GetBool("hasNailArt"))
		{
			return true;
		}
		return false;
	}

	private bool CanWallSlide()
	{
		if (cState.wallSliding && gm.isPaused)
		{
			return true;
		}
		if (!cState.touchingNonSlider && !inAcid && !cState.dashing && playerData.GetBool("hasWalljump") && !cState.onGround && !cState.recoiling && !gm.isPaused && !controlReqlinquished && !cState.transitioning && (cState.falling || cState.wallSliding) && !cState.doubleJumping && CanInput())
		{
			return true;
		}
		return false;
	}

	private bool CanTakeDamage()
	{
		if (damageMode != DamageMode.NO_DAMAGE && transitionState == HeroTransitionState.WAITING_TO_TRANSITION && !cState.invulnerable && !cState.recoiling && !playerData.GetBool("isInvincible") && !cState.dead && !cState.hazardDeath && !BossSceneController.IsTransitioning)
		{
			return true;
		}
		return false;
	}

	private bool CanWallJump()
	{
		if (playerData.GetBool("hasWalljump"))
		{
			if (cState.touchingNonSlider)
			{
				return false;
			}
			if (cState.wallSliding)
			{
				return true;
			}
			if (cState.touchingWall && !cState.onGround)
			{
				return true;
			}
			return false;
		}
		return false;
	}

	private bool ShouldHardLand(Collision2D collision)
	{
		if (!collision.gameObject.GetComponent<NoHardLanding>() && cState.willHardLand && !inAcid && hero_state != ActorStates.hard_landing)
		{
			return true;
		}
		return false;
	}

	private void OnCollisionEnter2D(Collision2D collision)
	{
		if (hero_state != ActorStates.no_input)
		{
			CollisionSide collisionSide = FindCollisionDirection(collision);
			if (collision.gameObject.layer != 8 && !collision.gameObject.CompareTag("HeroWalkable"))
			{
				return;
			}
			fallTrailGenerated = false;
			if (collisionSide == CollisionSide.top)
			{
				headBumpSteps = HEAD_BUMP_STEPS;
				if (cState.jumping)
				{
					CancelJump();
					CancelDoubleJump();
				}
				if (cState.bouncing)
				{
					CancelBounce();
					rb2d.velocity = new Vector2(rb2d.velocity.x, 0f);
				}
				if (cState.shroomBouncing)
				{
					CancelBounce();
					rb2d.velocity = new Vector2(rb2d.velocity.x, 0f);
				}
			}
			if (collisionSide == CollisionSide.bottom)
			{
				if (cState.attacking)
				{
					CancelDownAttack();
				}
				if (ShouldHardLand(collision))
				{
					DoHardLanding();
				}
				else if (collision.gameObject.GetComponent<SteepSlope>() == null && hero_state != ActorStates.hard_landing)
				{
					BackOnGround();
				}
				if (cState.dashing && dashingDown)
				{
					AffectedByGravity(gravityApplies: true);
					SetState(ActorStates.dash_landing);
					hardLanded = true;
				}
			}
		}
		else if (hero_state == ActorStates.no_input && transitionState == HeroTransitionState.DROPPING_DOWN && (gatePosition == GatePosition.bottom || gatePosition == GatePosition.top))
		{
			FinishedEnteringScene();
		}
	}

	private void OnCollisionStay2D(Collision2D collision)
	{
		if (cState.superDashing && (CheckStillTouchingWall(CollisionSide.left) || CheckStillTouchingWall(CollisionSide.right)))
		{
			superDash.SendEvent("HIT WALL");
		}
		if (hero_state == ActorStates.no_input || collision.gameObject.layer != 8)
		{
			return;
		}
		if (collision.gameObject.GetComponent<NonSlider>() == null)
		{
			cState.touchingNonSlider = false;
			if (CheckStillTouchingWall(CollisionSide.left))
			{
				cState.touchingWall = true;
				touchingWallL = true;
				touchingWallR = false;
			}
			else if (CheckStillTouchingWall(CollisionSide.right))
			{
				cState.touchingWall = true;
				touchingWallL = false;
				touchingWallR = true;
			}
			else
			{
				cState.touchingWall = false;
				touchingWallL = false;
				touchingWallR = false;
			}
			if (CheckTouchingGround())
			{
				if (ShouldHardLand(collision))
				{
					DoHardLanding();
				}
				else if (hero_state != ActorStates.hard_landing && hero_state != ActorStates.dash_landing && cState.falling)
				{
					BackOnGround();
				}
			}
			else if (cState.jumping || cState.falling)
			{
				cState.onGround = false;
				proxyFSM.SendEvent("HeroCtrl-LeftGround");
				SetState(ActorStates.airborne);
			}
		}
		else
		{
			cState.touchingNonSlider = true;
		}
	}

	private void OnCollisionExit2D(Collision2D collision)
	{
		if (cState.recoilingLeft || cState.recoilingRight)
		{
			cState.touchingWall = false;
			touchingWallL = false;
			touchingWallR = false;
			cState.touchingNonSlider = false;
		}
		if (touchingWallL && !CheckStillTouchingWall(CollisionSide.left))
		{
			cState.touchingWall = false;
			touchingWallL = false;
		}
		if (touchingWallR && !CheckStillTouchingWall(CollisionSide.right))
		{
			cState.touchingWall = false;
			touchingWallR = false;
		}
		if (hero_state == ActorStates.no_input || cState.recoiling || collision.gameObject.layer != 8 || CheckTouchingGround())
		{
			return;
		}
		cState.onGround = false;
		SetState(ActorStates.airborne);
		if (cState.wasOnGround)
		{
			ledgeBufferSteps = LEDGE_BUFFER_STEPS;
		}
	}


	private void FilterInput()
	{
		if (move_input > 0.3f)
		{
			move_input = 1f;
		}
		else if (move_input < -0.3f)
		{
			move_input = -1f;
		}
		else
		{
			move_input = 0f;
		}
		if (vertical_input > 0.5f)
		{
			vertical_input = 1f;
		}
		else if (vertical_input < -0.5f)
		{
			vertical_input = -1f;
		}
		else
		{
			vertical_input = 0f;
		}
	}

	private void orig_Update()
	{
		current_velocity = rb2d.velocity;
		if (hero_state == ActorStates.dash_landing)
		{
			dashLandingTimer += Time.deltaTime;
			if (dashLandingTimer > DOWN_DASH_TIME)
			{
				BackOnGround();
			}
		}
		if (hero_state == ActorStates.hard_landing)
		{
			hardLandingTimer += Time.deltaTime;
			if (hardLandingTimer > HARD_LANDING_TIME)
			{
				SetState(ActorStates.grounded);
				BackOnGround();
			}
		}
		else if (hero_state == ActorStates.no_input)
		{
			if (cState.recoiling)
			{
				if ((!playerData.GetBool("equippedCharm_4") && recoilTimer < RECOIL_DURATION) || (playerData.GetBool("equippedCharm_4") && recoilTimer < RECOIL_DURATION_STAL))
				{
					recoilTimer += Time.deltaTime;
				}
				else
				{
					CancelDamageRecoil();
					if ((prev_hero_state == ActorStates.idle || prev_hero_state == ActorStates.running) && !CheckTouchingGround())
					{
						cState.onGround = false;
						SetState(ActorStates.airborne);
					}
					else
					{
						SetState(ActorStates.previous);
					}
				}
			}
		}
		else if (hero_state != ActorStates.no_input)
		{
			LookForInput();
			if (cState.recoiling)
			{
				cState.recoiling = false;
				AffectedByGravity(gravityApplies: true);
			}
			if (cState.attacking && !cState.dashing)
			{
				attack_time += Time.deltaTime;
				if (attack_time >= attackDuration)
				{
					ResetAttacks();
				}
			}
			if (cState.bouncing)
			{
				if (bounceTimer < BOUNCE_TIME)
				{
					bounceTimer += Time.deltaTime;
				}
				else
				{
					CancelBounce();
					rb2d.velocity = new Vector2(rb2d.velocity.x, 0f);
				}
			}
			if (cState.shroomBouncing && current_velocity.y <= 0f)
			{
				cState.shroomBouncing = false;
			}
		}
		LookForQueueInput();
		if (cState.wallSliding)
		{
			if (airDashed)
			{
				airDashed = false;
			}
			if (doubleJumped)
			{
				doubleJumped = false;
			}
			if (cState.onGround)
			{
				FlipSprite();
				CancelWallsliding();
			}
			if (!cState.touchingWall)
			{
				FlipSprite();
				CancelWallsliding();
			}
			if (!CanWallSlide())
			{
				CancelWallsliding();
			}
		}
		if (wallSlashing && !cState.wallSliding)
		{
			CancelAttack();
		}
		if (attack_cooldown > 0f)
		{
			attack_cooldown -= Time.deltaTime;
		}
		if (dashCooldownTimer > 0f)
		{
			dashCooldownTimer -= Time.deltaTime;
		}
		if (!gm.isPaused)
		{
			if (inputHandler.inputActions.attack.IsPressed && CanNailCharge())
			{
				cState.nailCharging = true;
				nailChargeTimer += Time.deltaTime;
			}
			else if (cState.nailCharging || nailChargeTimer != 0f)
			{
				cState.nailCharging = false;
			}
			if (!artChargedEffect.activeSelf && nailChargeTimer >= nailChargeTime)
			{
				cState.nailCharging = true;
			}
		}
		if (gm.isPaused && !inputHandler.inputActions.attack.IsPressed)
		{
			cState.nailCharging = false;
			nailChargeTimer = 0f;
		}
		if (cState.swimming && !CanSwim())
		{
			cState.swimming = false;
		}
	}

	private Vector2 OrigDashVector()
	{
		float num = ((!playerData.GetBool("equippedCharm_16") || !cState.shadowDashing) ? DASH_SPEED : DASH_SPEED_SHARP);
		Vector2 result = (dashingDown ? new Vector2(0f, 0f - num) : (cState.facingRight ? ((!CheckForBump(CollisionSide.right)) ? new Vector2(num, 0f) : new Vector2(num, (!cState.onGround) ? 5f : 4f)) : ((!CheckForBump(CollisionSide.left)) ? new Vector2(0f - num, 0f) : new Vector2(0f - num, (!cState.onGround) ? 5f : 4f))));
		return result;
	}

	private void orig_DoAttack()
	{
		cState.recoiling = false;
		if (playerData.GetBool("equippedCharm_32"))
		{
			attack_cooldown = ATTACK_COOLDOWN_TIME_CH;
		}
		else
		{
			attack_cooldown = ATTACK_COOLDOWN_TIME;
		}
		if (vertical_input > Mathf.Epsilon)
		{
			Attack(AttackDirection.upward);
			StartCoroutine(CheckForTerrainThunk(AttackDirection.upward));
		}
		else if (vertical_input < 0f - Mathf.Epsilon)
		{
			if (hero_state != ActorStates.idle && hero_state != ActorStates.running)
			{
				Attack(AttackDirection.downward);
				StartCoroutine(CheckForTerrainThunk(AttackDirection.downward));
			}
			else
			{
				Attack(AttackDirection.normal);
				StartCoroutine(CheckForTerrainThunk(AttackDirection.normal));
			}
		}
		else
		{
			Attack(AttackDirection.normal);
			StartCoroutine(CheckForTerrainThunk(AttackDirection.normal));
		}
	}
}
