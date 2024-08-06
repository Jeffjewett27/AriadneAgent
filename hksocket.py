import asyncio
import os
import websockets
import json

async def connect_to_server():
    uri = "ws://localhost:8645"
    async with websockets.connect(uri) as websocket:
        print("Connected to server")
        try:
            while True:
                message = await websocket.recv()
                try:
                    data = json.loads(message)
                    print(data.keys())
                    if 'terrainSegmentation' in data:
                        segmentation = data['terrainSegmentation']
                        print(f'Segmentation for {data["sceneName"]}', data['terrainSegmentation'])
                        segmentationFileContents = f'terrain = {segmentation}'
                        segmentationFilePath = f'segmentations/{data["sceneName"]}.py'
                        if not os.path.exists(segmentationFilePath):
                            try:
                                with open(segmentationFilePath, 'w') as file:
                                    file.write(segmentationFileContents)
                            except:
                                print(f'Could not open file, {segmentationFilePath}')
                except:
                    print(f"Received non-json: {message}")
                    continue

        except websockets.ConnectionClosed:
            print("Connection closed")

if __name__ == "__main__":
    asyncio.run(connect_to_server())
