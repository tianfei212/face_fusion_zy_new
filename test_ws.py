import asyncio
import websockets
import sys

async def test_connect(uri):
    print(f"Testing {uri}...", flush=True)
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Successfully connected to {uri}", flush=True)
    except Exception as e:
        print(f"Failed to connect to {uri}: {e}", flush=True)

async def main():
    print("Starting tests...", flush=True)
    await test_connect("ws://localhost:5555/")
    await test_connect("ws://localhost:5555/ws")
    await test_connect("ws://localhost:5555/api/ws")
    await test_connect("ws://localhost:5555/video_in")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Main error: {e}", flush=True)
