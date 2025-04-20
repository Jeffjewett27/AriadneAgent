import pywinauto
from pywinauto.application import Application
from time import sleep

window_name = "Hollow Knight"
try:
    # Connect to an already running application window by its title
    app = Application().connect(title=window_name)

    # Get the window handle
    window = app.window(title=window_name)

    # Send the keystrokes directly to the window
    # print(f"Sent keystrokes '{keys}' to '{window_name}' without bringing it to focus.")
except Exception as e:
    print(f"Could not find window '{window_name}': {e}")
    quit()

for i in range(5):
    window.send_keystrokes("{z down}")
    sleep(0.02)
    window.send_keystrokes("z")
    sleep(1)
