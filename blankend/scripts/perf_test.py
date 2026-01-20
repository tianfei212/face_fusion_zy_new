
import os
import sys
import time
import threading
import queue
import cv2
import numpy as np
from pathlib import Path

import logging

# Add project root to path
ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')


# Mock environment variables if needed
os.environ["BLANKEND_HW_TRANSCODE"] = "1"

from blankend.ai_core.service import AiCoreService, FrameResult
from blankend.ai_core.resources import default_asset_paths

def main():
    print("Starting perf test...")
    service = AiCoreService()
    
    # Configure service
    service.configure(
        enabled=True,
        similarity_threshold=0.6,
        depth_fusion=True,
        output_jpeg_quality=85,
        max_fps=30.0,
        min_fps=15.0
    )
    
    # Ensure assets are loaded
    assets = default_asset_paths()
    portrait_path = assets.portraits_dir / "portrait-1.jpg"
    # Create dummy portrait if not exists
    if not portrait_path.exists():
        print(f"Creating dummy portrait at {portrait_path}")
        dummy_p = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(portrait_path), dummy_p)
    
    service.set_portrait("portrait-1.jpg")
    
    # Start service
    service.start()
    
    # Prepare test image (1080p)
    img_path = Path(ROOT_DIR) / "frontend/public/t1.jpg"
    if img_path.exists():
        img = cv2.imread(str(img_path))
        print(f"Loaded test image from {img_path}")
    else:
        print(f"Test image not found at {img_path}, using random noise")
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    ok, jpeg_bytes = cv2.imencode(".jpg", img)
    if not ok:
        print("Failed to encode test image")
        return
    jpeg_bytes = bytes(jpeg_bytes)
    print(f"Test image size: {len(jpeg_bytes)} bytes")
    
    # Sender thread
    client_id = b"perf_test_client"
    stop_event = threading.Event()
    
    def sender_loop():
        seq = 0
        while not stop_event.is_set():
            ok = service.submit(client_id, jpeg_bytes)
            if ok:
                seq += 1
            else:
                # print("Drop")
                pass
            time.sleep(1.0 / 30.0) # Simulate 30 FPS input
            
    t_send = threading.Thread(target=sender_loop)
    t_send.start()
    
    # Receiver loop
    received = 0
    start_time = time.time()
    try:
        while True:
            try:
                res = service._output_q.get(timeout=1.0)
                received += 1
                if received % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = received / elapsed
                    print(f"Received {received} frames. FPS: {fps:.2f}")
            except queue.Empty:
                if received > 0:
                    print("Queue empty, waiting...")
                
            if time.time() - start_time > 10.0:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        t_send.join()
        print("Test finished")

if __name__ == "__main__":
    main()
