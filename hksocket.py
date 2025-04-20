import asyncio
import os
from queue import Queue
import queue
from collections import deque
import threading
import websockets
import json


class OverwritingQueue:
    """Implements a thread-safe 'overwrite-on-full' queue"""

    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        # Condition uses the same lock internally
        self.not_empty = threading.Condition(self.lock)

    def put(self, item):
        with self.not_empty:
            # deque with maxlen handles overwriting oldest
            self.buffer.append(item)
            self.not_empty.notify()

    def get(self, timeout=None):
        with self.not_empty:
            # wait until buffer has something or timeout expires
            if not self.buffer:
                if not self.not_empty.wait(timeout):
                    raise queue.Empty
            # at this point buffer must be non‑empty
            return self.buffer.popleft()

    def __len__(self):
        with self.lock:
            return len(self.buffer)
        
    def qsize(self):
        with self.lock:
            return len(self.buffer)

    def empty(self):
        return self.qsize() == 0

    def full(self):
        return self.qsize() == self.buffer.maxlen


socket_task_status = "unused"
socket_thread = None
max_events = 1
# event_queue = OverwritingQueue(maxlen=max_events)
event_queue = Queue()
connected_event = threading.Event()


async def connect_to_server():
    global socket_task_status
    uri = "ws://localhost:8645"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to server")
            connected_event.set()
            try:
                while socket_task_status == "running":
                    message = await websocket.recv()
                    try:
                        data = json.loads(message)
                        event_queue.put(data)

                    except:
                        print(f"Received non-json: {message}")
                        continue
                print("The server loop has ended")

            except websockets.ConnectionClosed:
                print("Connection closed")
            except Exception as e:
                print("Unexpected exception", e)
                return
    except ConnectionRefusedError:
        print(f"The connection to {uri} was refused")
        socket_task_status = "failed"
        connected_event.set()
    except Exception as e:
        socket_task_status = "failed"
        connected_event.set()
        print("Unexpected exception", e)
        return


def websocket_server_thread():
    global socket_thread, socket_task_status
    socket_task_status = "running"
    asyncio.run(connect_to_server())
    if socket_task_status == "running":
        socket_task_status = "completed"
    print("Socket thread is terminating gracefully")
    socket_thread = None


def run_websocket_server():
    global socket_task_status, socket_thread
    if socket_task_status == "running" or socket_thread is not None:
        print("Cannot start server: websocket server is already running")
        return
    socket_thread = threading.Thread(target=websocket_server_thread, daemon=True)
    socket_thread.start()


def stop_websocket_server():
    global socket_task_status, socket_thread
    if socket_task_status != "running" and socket_thread is None:
        print("Cannot stop server: there is no server being run")
        return
    socket_task_status = "stopped"


def is_server_running():
    global socket_task_status, socket_thread
    return socket_task_status == "running" and socket_thread is not None


if __name__ == "__main__":
    run_websocket_server()
