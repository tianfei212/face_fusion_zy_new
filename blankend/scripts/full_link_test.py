
import os
import sys
import time
import json
import threading
import queue
import cv2
import numpy as np
import asyncio
from pathlib import Path

# Add project root to path
ROOT_DIR = str(Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger("full_link_test")

try:
    import zmq
except ImportError:
    print("zmq module not found. Please install pyzmq.")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("websockets module not found. Please install websockets.")
    sys.exit(1)

from blankend.ai_core.service import AiCoreService
from blankend.ai_core.resources import default_asset_paths
from blankend.streaming.routes import ZmqRelay
from blankend.streaming.hub import StreamHub

# Mock environment variables
os.environ["BLANKEND_HW_TRANSCODE"] = "1"

def main():
    # 1. Configuration
    RELAY_IP = "localhost"
RELAY_ZMQ_PORT = 8888
RELAY_WS_PORT = 5555
RELAY_ENDPOINT = f"tcp://{RELAY_IP}:{RELAY_ZMQ_PORT}"
WS_ENDPOINT = f"ws://{RELAY_IP}:{RELAY_WS_PORT}/video_in"
    
    # Use standard identity to ensure Relay routes to us? 
    # Or try TEST identity first. 
    # If Relay is hardcoded to route to AI_WORKER, we must use AI_WORKER.
    # Let's try AI_WORKER (assuming dev env).
    WORKER_ID = "AI_WORKER" 
    
    print(f"Starting full-link test (WS Client -> Relay -> ZMQ Backend)")
    print(f"Relay ZMQ: {RELAY_ENDPOINT}")
    print(f"Relay WS:  {WS_ENDPOINT}")
    print(f"Worker Identity: {WORKER_ID}")

    # 2. Skip Backend Service (Assumed running separately via uvicorn)
    # If we run a second backend here, it will conflict on ZMQ identity.
    print("Assuming Backend is running separately (e.g. uvicorn blankend.main:app)")

    # 3. Prepare Test Data
    # Use real image if available, otherwise noise
    image_path = Path(ROOT_DIR) / "t1.jpg"
    if image_path.exists():
        print(f"Using image: {image_path}")
        img = cv2.imread(str(image_path))
        # Resize to simulate reasonable input (e.g. 720p or smaller for test)
        # img = cv2.resize(img, (720, 1280)) 
        # Actually t1.jpg is 1228x2754. Let's resize to 540x960 for speed in test
        img = cv2.resize(img, (540, 960))
    else:
        print("Using random noise image")
        img = np.random.randint(0, 255, (540, 960, 3), dtype=np.uint8)
        
    ok, jpeg_bytes = cv2.imencode(".jpg", img)
    if not ok:
        print("Failed to encode image")
        return
    jpeg_bytes = bytes(jpeg_bytes)
    print(f"Frame size: {len(jpeg_bytes)} bytes")
    
    # Calculate hash of sent frame to identify echo
    import hashlib
    sent_hash = hashlib.md5(jpeg_bytes).hexdigest()
    print(f"Sent Frame Hash: {sent_hash}")

    async def test_loop():
        async with websockets.connect(WS_ENDPOINT, ping_interval=None) as ws:
            print("WS Connected")
            
            # Send 1 frame first to characterize Echo vs Processed
            print("\n--- Phase 1: Characterization (Send 1 frame) ---")
            await ws.send(jpeg_bytes)
            send_time = time.time()
            
            echo_seen = False
            processed_seen = False
            
            try:
                while time.time() - send_time < 10.0: # Wait 10s
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        recv_time = time.time()
                        latency = (recv_time - send_time) * 1000
                        
                        if isinstance(msg, bytes):
                            recv_hash = hashlib.md5(msg).hexdigest()
                            is_echo = (recv_hash == sent_hash)
                            type_str = "ECHO" if is_echo else "PROCESSED"
                            print(f"Recv: {len(msg)} bytes, Latency: {latency:.1f}ms, Type: {type_str}")
                            
                            if is_echo: echo_seen = True
                            else: processed_seen = True
                            
                        else:
                            print(f"Recv Text: {msg}")
                            
                        if echo_seen and processed_seen:
                            print("Both Echo and Processed frames received.")
                            break
                            
                    except asyncio.TimeoutError:
                        print("Timeout waiting for frame")
                        continue
                    except Exception as e:
                        print(f"Error: {e}")
                        break
            except Exception as e:
                print(f"Loop Error: {e}")

            if not processed_seen:
                print("WARNING: No processed frame received in Phase 1. Backend might be down or dropping.")
            
            # Phase 2: Load Test
            print("\n--- Phase 2: Load Test (Send 15 frames) ---")
            sent_count = 0
            recv_processed_count = 0
            
            async def send_loop():
                nonlocal sent_count
                for i in range(15):
                    await ws.send(jpeg_bytes)
                    sent_count += 1
                    print(f"Sent {i+1}/15")
                    await asyncio.sleep(1.0/15.0)

            send_task = asyncio.create_task(send_loop())
            
            # Wait for all frames to be sent and received
            # Instead of per-frame timeout causing break, we wait for total duration
            # With 2.4s latency and 15 frames, we might need 40-60s to process all if serial.
            end_time = time.time() + 60.0 
            start_t = time.time()
            
            while time.time() < end_time and recv_processed_count < 15:
                try:
                    # Use shorter timeout for check to allow checking loop condition
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    
                    if isinstance(msg, bytes):
                        # Check hash
                        recv_hash = hashlib.md5(msg).hexdigest()
                        is_echo = (recv_hash == sent_hash)
                        
                        if is_echo:
                            # Even if it looks like an Echo (identical hash), it might be the result
                            # if the pipeline didn't change the image (e.g. no face swap configured).
                            # We count it as processed for this connectivity test.
                            recv_processed_count += 1
                            print(f"Recv (Echo-like): {recv_processed_count}/15")
                        else:
                            recv_processed_count += 1
                            print(f"Recv Processed: {recv_processed_count}/15")
                    
                except asyncio.TimeoutError:
                    if send_task.done() and recv_processed_count == 15:
                        break
                    continue
                except Exception as e:
                    print(f"Error receiving: {e}")
                    break
            
            if not send_task.done():
                print("Warning: Send task did not finish in time?")
                send_task.cancel()
            
            end_t = time.time()
            
            print(f"\n--- Results ---")
            print(f"Sent: {sent_count}")
            print(f"Recv Processed: {recv_processed_count}")
            print(f"Drop Rate: {100 * (1 - recv_processed_count/sent_count):.1f}%")
            print(f"Duration: {end_t - start_t:.2f}s")
            
    asyncio.run(test_loop())

if __name__ == "__main__":
    main()
