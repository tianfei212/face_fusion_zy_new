import asyncio
import websockets
import sys

async def test_ws(url):
    print(f"Testing {url}...")
    try:
        async with websockets.connect(url) as websocket:
            print(f"Successfully connected to {url}")
            await websocket.close()
            return True
    except Exception as e:
        print(f"Failed to connect to {url}: {e}")
        return False

async def main():
    r1 = await test_ws("ws://localhost:8001/video_in")
    r2 = await test_ws("ws://localhost:5555/video_in")
    # Gateway might not use /video_in path, maybe just root?
    r3 = await test_ws("ws://localhost:5555/")
    
    if not r1 and not r2 and not r3:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
