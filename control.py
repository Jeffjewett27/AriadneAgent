import multiprocessing
import time
from pywinauto.application import Application
from time import sleep
from typing import List, Tuple

# -----------------------------------------------------------------------------
# Configuration: adjust if your key bindings differ
# -----------------------------------------------------------------------------
# Human‑readable mapping for each control index:
#  0=up, 1=left, 2=down, 3=right, 4=jump, 5=attack, 6=dash
KEY_STRINGS = [
    "{UP}",  # up
    "{LEFT}",  # left
    "{DOWN}",  # down
    "{RIGHT}",  # right
    "z",  # jump
    "x",  # attack
    "c",  # dash (example)
]
# Versions for "key down" events
# KEY_STRINGS_DOWN = [k if k.startswith("{") else k.upper() for k in KEY_STRINGS]
KEY_STRINGS_DOWN = [
    k.replace("}", " down}") if k.startswith("{") else f"{{{k} down}}"
    for k in KEY_STRINGS
]

# -----------------------------------------------------------------------------
# IPC Protocol
# -----------------------------------------------------------------------------
FRAME = "FRAME"  # tag for a full 7‑bool snapshot
SHUTDOWN = "EXIT"  # tag to cleanly shut down the control process

# -----------------------------------------------------------------------------
# Shared state and IPC queue
# -----------------------------------------------------------------------------
control_process: multiprocessing.Process | None = None
control_queue = multiprocessing.Queue()
# This list is the authoritative state for each frame.
controls_pressed: List[bool] = [False] * len(KEY_STRINGS)

hollow_knight_window = None  # will hold the pywinauto WindowSpecification


def tick_controls():
    frame_state = controls_pressed.copy()
    control_queue.put((FRAME, frame_state))


# -----------------------------------------------------------------------------
# Window discovery
# -----------------------------------------------------------------------------
def get_hollow_knight_window():
    global hollow_knight_window
    window_name = "Hollow Knight"
    try:
        app = Application().connect(title=window_name)
        hollow_knight_window = app.window(title=window_name)
    except Exception as e:
        hollow_knight_window = None
        print(f"Could not find window '{window_name}': {e}")
    return hollow_knight_window is not None


def require_window() -> bool:
    if hollow_knight_window is None:
        return get_hollow_knight_window()
    return True


# -----------------------------------------------------------------------------
# Control functions: ONLY mutate controls_pressed[], no queue.put() here
# -----------------------------------------------------------------------------
def press_up():
    controls_pressed[0] = True


def release_up():
    controls_pressed[0] = False


def press_left():
    controls_pressed[1] = True


def release_left():
    controls_pressed[1] = False


def press_down():
    controls_pressed[2] = True


def release_down():
    controls_pressed[2] = False


def press_right():
    controls_pressed[3] = True


def release_right():
    controls_pressed[3] = False


def press_jump():
    controls_pressed[4] = True


def release_jump():
    controls_pressed[4] = False


def press_attack():
    controls_pressed[5] = True


def release_attack():
    controls_pressed[5] = False


def press_dash():
    controls_pressed[6] = True


def release_dash():
    controls_pressed[6] = False


def set_controls_pressed(new_controls: list[bool]):
    assert len(new_controls) == len(controls_pressed)
    for i, val in enumerate(new_controls):
        controls_pressed[i] = val


# -----------------------------------------------------------------------------
# Control‐process entrypoint: consumes whole-frame state packets
# -----------------------------------------------------------------------------
def handle_controls(event_queue: multiprocessing.Queue):
    # Ensure we have a window handle before processing frames
    if not require_window():
        return

    prev_state = [False] * len(KEY_STRINGS)
    down_times = [0] * len(KEY_STRINGS)

    while True:
        tag, payload = event_queue.get()
        if tag == SHUTDOWN:
            break
        if tag == FRAME:
            new_state: List[bool] = payload
            # For each control, compare prev vs now and send key down/up
            for i, (was, now) in enumerate(zip(prev_state, new_state)):
                if was == now:
                    continue
                down_str = ""
                if now:
                    # key pressed this frame
                    ks = KEY_STRINGS_DOWN[i]
                    down_times[i] = time.time()
                else:
                    # key released this frame
                    ks = KEY_STRINGS[i]
                    down_str = str(down_times[i] - time.time())
                print("keystroke ", ks, down_str)
                try:
                    hollow_knight_window.send_keystrokes(ks)
                except Exception as e:
                    print(f"Error sending keystroke '{ks}': {e}")
            prev_state = new_state


# -----------------------------------------------------------------------------
# Lifecycle helpers
# -----------------------------------------------------------------------------
def start_control_process():
    global control_process
    """Spawn the control process to consume frame packets."""
    if control_process is not None:
        print("control process already started")
        return
    control_process = multiprocessing.Process(
        target=handle_controls, args=(control_queue,), daemon=True
    )
    control_process.start()


def stop_control_process(): 
    """Shutdown the control process cleanly."""
    control_queue.put((SHUTDOWN, None))
    control_process.join()
    control_process.close()
