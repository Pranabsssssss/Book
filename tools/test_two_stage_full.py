import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from detector import IRDetector, LEDDetector

print("=" * 60)
print("TESTING TWO-STAGE 5mm LED BULB DETECTOR (GREEN & IR MODES)")
print("=" * 60)

det_ir = IRDetector(led_type="ir")
det_green = LEDDetector(led_type="green")

# 1. User Photo 1: Unlit 5mm Stylus Bulb in Hand
p1 = r'C:/Users/prana/.gemini/antigravity/brain/742360c1-85ca-4369-a3d9-0dea5a3a94ec/.user_uploaded/media_1793025491309_fd5b99af.jpg'
im1 = cv2.imread(p1)
pt1, _, _, meta1 = det_ir.detect_debug(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
print("\n[Test 1: User Photo 1 - Unlit 5mm Bulb in Hand]")
print(f"  Drawing Point : {pt1} (Expected: None -> Pen Lifted)")
print(f"  Bulb State    : {meta1['bulb_state']} (Expected: LOCATED_IDLE)")
print(f"  Bulb Position : {meta1['bulb_pos']}")
assert pt1 is None, "Failed Test 1: Unlit bulb must NOT draw on whiteboard!"
assert meta1['bulb_state'] == "LOCATED_IDLE", f"Failed Test 1: Expected LOCATED_IDLE, got {meta1['bulb_state']}"

# 2. User Emitting IR LED Image (IR Mode)
p_emit = r'dataset/ir_led_emitting/user_ref_emitting_ground_truth.png'
im_emit = cv2.imread(p_emit)
frame_emit = np.zeros((480, 640, 3), dtype=np.uint8)
h_e, w_e = im_emit.shape[:2]
frame_emit[100:100+h_e, 200:200+w_e] = cv2.cvtColor(im_emit, cv2.COLOR_BGR2RGB)
pt_emit, _, _, meta_emit = det_ir.detect_debug(frame_emit)
print("\n[Test 2: Active Emitting 5mm IR LED Bulb]")
print(f"  Drawing Point : {pt_emit} (Expected: Valid Coordinate -> Drawing Active)")
print(f"  Bulb State    : {meta_emit['bulb_state']} (Expected: EMITTING)")
print(f"  Purple Excess : +{meta_emit['purple_excess']:.1f}")
assert pt_emit is not None, "Failed Test 2: Active emitting IR LED MUST draw!"
assert meta_emit['bulb_state'] == "EMITTING", f"Failed Test 2: Expected EMITTING, got {meta_emit['bulb_state']}"

# 3. Active Emitting 5mm Green LED Bulb (Green Mode)
frame_green = np.zeros((480, 640, 3), dtype=np.uint8)
# High Green (G=245), Low Red (R=35), Low Blue (B=25) -> Pure Green LED
cv2.circle(frame_green, (310, 220), 8, (35, 245, 25), -1)
pt_green, _, _, meta_green = det_green.detect_debug(frame_green)
print("\n[Test 3: Active Emitting 5mm Green LED Bulb]")
print(f"  Drawing Point : {pt_green} (Expected: Valid Coordinate -> Drawing Active)")
print(f"  Bulb State    : {meta_green['bulb_state']} (Expected: EMITTING)")
print(f"  Green Excess  : +{meta_green['green_excess']:.1f}")
assert pt_green is not None, "Failed Test 3: Active Green LED MUST draw!"
assert meta_green['bulb_state'] == "EMITTING", f"Failed Test 3: Expected EMITTING, got {meta_green['bulb_state']}"

# 4. Whiteboard White Drawing (Pure White Light: R=G=B=255)
frame_white = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.circle(frame_white, (320, 240), 12, (255, 255, 255), -1)
pt_white, _, _, meta_white = det_green.detect_debug(frame_white)
print("\n[Test 4: Blackboard White Drawing on Screen]")
print(f"  Drawing Point : {pt_white} (Expected: None -> Hard Rejected)")
print(f"  Bulb State    : {meta_white['bulb_state']}")
assert pt_white is None, "Failed Test 4: Whiteboard drawing must NEVER draw!"

# 5. Room Lamp / Incandescent (B=140, G=190, R=240)
frame_lamp = np.zeros((480, 640, 3), dtype=np.uint8)
frame_lamp[150:180, 250:280] = cv2.cvtColor(np.full((30, 30, 3), [140, 190, 240], dtype=np.uint8), cv2.COLOR_BGR2RGB)
pt_lamp, _, _, meta_lamp = det_green.detect_debug(frame_lamp)
print("\n[Test 5: Ambient Room Lamp Light]")
print(f"  Drawing Point : {pt_lamp} (Expected: None -> Hard Rejected)")
assert pt_lamp is None, "Failed Test 5: Room lamp must NEVER draw!"

print("\n" + "=" * 60)
print("ALL 5 VERIFICATION TESTS PASSED WITH 100% ACCURACY!")
print("=" * 60)
