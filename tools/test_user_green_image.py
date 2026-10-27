import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from detector import IRDetector

im = cv2.imread(r'C:/Users/prana/.gemini/antigravity/brain/742360c1-85ca-4369-a3d9-0dea5a3a94ec/.user_uploaded/media_1793093665318_00f30c26.png')
# Extract the camera feed inside the testing workbench (border is at x=28..412, y=36..326)
cam_feed = im[36:326, 28:412]

# Initialize detector in Green LED mode
det = IRDetector(led_type="green")

# Run detection on user's exact camera feed
pt, annotated, thresh_mask, meta = det.detect_debug(cv2.cvtColor(cam_feed, cv2.COLOR_BGR2RGB))

print("=" * 60)
print("TESTING DETECTOR ON USER'S UPLOADED SCREENSHOT")
print("=" * 60)
print(f"Drawing Point Detected : {pt}")
print(f"Bulb State             : {meta['bulb_state']}")
print(f"Green Excess Meter     : +{meta['excess']:.1f}")
print(f"Status                 : {meta['status']}")
print("=" * 60)

assert pt is not None, "FAILED: The green spot on the paper in the user's image was not detected!"
assert meta['bulb_state'] == "EMITTING", f"FAILED: Expected EMITTING, got {meta['bulb_state']}"

print("SUCCESS! The scattered green light on the paper was 100% correctly detected as DRAWING ACTIVE!")
