import pywinauto
from pywinauto.application import Application
from time import sleep
import multiprocessing


def get_hollow_knight_window():
    window_name = "Hollow Knight"
    try:
        # Connect to an already running application window by its title
        app = Application().connect(title=window_name)

        # Get the window handle
        window = app.window(title=window_name)

        # Send the keystrokes directly to the window
        # print(f"Sent keystrokes '{keys}' to '{window_name}' without bringing it to focus.")
    except Exception as e:
        window = None
        print(f"Could not find window '{window_name}': {e}")
    return window


hollow_knight_window = None
control_queue = multiprocessing.Queue()
control_process = None
controls_pressed = [
    False,  # up=0
    False,  # left=1
    False,  # down=2
    False,  # right=3
    False,  # jump=4
    False,  # attack=5
    False,  # dash=6
]


def require_window():
    global hollow_knight_window
    if hollow_knight_window is None:
        hollow_knight_window = get_hollow_knight_window()
    return hollow_knight_window is not None


def press_up():
    if controls_pressed[0]:
        return
    controls_pressed[0] = True
    control_queue.put("press_up")


def release_up():
    if not controls_pressed[0]:
        return
    controls_pressed[0] = False
    control_queue.put("release_up")


def press_down():
    if controls_pressed[2]:
        return
    controls_pressed[2] = True
    control_queue.put("press_down")


def release_down():
    if not controls_pressed[2]:
        return
    controls_pressed[2] = False
    control_queue.put("release_down")


def press_left():
    if controls_pressed[1]:
        return
    controls_pressed[1] = True
    control_queue.put("press_left")


def release_left():
    if not controls_pressed[1]:
        return
    controls_pressed[1] = False
    control_queue.put("release_left")


def press_right():
    if controls_pressed[3]:
        return
    controls_pressed[3] = True
    control_queue.put("press_right")


def release_right():
    if not controls_pressed[3]:
        return
    controls_pressed[3] = False
    control_queue.put("release_right")


def press_jump():
    if controls_pressed[4]:
        return
    controls_pressed[4] = True
    control_queue.put("press_jump")


def release_jump():
    if not controls_pressed[4]:
        return
    controls_pressed[4] = False
    control_queue.put("release_jump")


def attack():
    control_queue.put("attack")


def press_attack():
    if controls_pressed[5]:
        return
    controls_pressed[5] = True
    control_queue.put("press_attack")


def release_attack():
    if not controls_pressed[5]:
        return
    controls_pressed[5] = False
    control_queue.put("release_attack")


def dash():
    control_queue.put("dash")


def press_dash():
    if controls_pressed[6]:
        return
    controls_pressed[6] = True
    control_queue.put("press_dash")


def release_dash():
    if not controls_pressed[6]:
        return
    controls_pressed[6] = False
    control_queue.put("release_dash")


def test_move():
    import time

    for _ in range(2):
        press_right()
        time.sleep(1)
        release_right()
        press_left()
        time.sleep(1)
        release_left()


def handle_controls(event_queue):
    success = require_window()
    if not success:
        return

    while True:
        task = event_queue.get()
        # Execute the task (a method in this case)
        match task:
            case "EXIT":
                break
            case "press_up":
                hollow_knight_window.send_keystrokes("{UP down}")
            case "release_up":
                hollow_knight_window.send_keystrokes("{UP}")
            case "press_down":
                hollow_knight_window.send_keystrokes("{DOWN down}")
            case "release_down":
                hollow_knight_window.send_keystrokes("{DOWN}")
            case "press_left":
                hollow_knight_window.send_keystrokes("{LEFT down}")
            case "release_left":
                hollow_knight_window.send_keystrokes("{LEFT}")
            case "press_right":
                hollow_knight_window.send_keystrokes("{RIGHT down}")
            case "release_right":
                hollow_knight_window.send_keystrokes("{RIGHT}")
            case "press_jump":
                hollow_knight_window.send_keystrokes("{z down}")
            case "release_jump":
                hollow_knight_window.send_keystrokes("z")
            case "press_attack":
                hollow_knight_window.send_keystrokes("{x down}")
            case "release_attack":
                hollow_knight_window.send_keystrokes("x")
            case "attack":
                hollow_knight_window.send_keystrokes("x")

            case _:
                pass


def start_control_process():
    global control_process
    control_process = multiprocessing.Process(
        target=handle_controls, args=(control_queue,), daemon=True
    )
    control_process.start()


def stop_control_process():
    control_queue.put("EXIT")
    control_process.join()
    control_process.close()


if __name__ == "__main__":
    import time

    press_left()
    time.sleep(1)
    release_left()
