import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from detector import IRDetector

det = IRDetector(threshold=180, led_type="ir")

# 1. Test user's real emitting IR LED image (Image 2)
p_emit = Path('dataset/ir_led_emitting/user_ref_emitting_ground_truth.png')
im_emit = cv2.imread(str(p_emit))
frame_emit = np.zeros((480, 640, 3), dtype=np.uint8)
h, w = im_emit.shape[:2]
frame_emit[100:100+h, 150:150+w] = cv2.cvtColor(im_emit, cv2.COLOR_BGR2RGB)
pt_emit, _, _, meta_emit = det.detect_debug(frame_emit)

print('=== Test User Reference Emitting IR LED ===')
print('Detected Point:', pt_emit)
print('Status:', meta_emit['status'])
print('Purple Excess:', f"+{meta_emit['purple_excess']:.1f}")
assert pt_emit is not None, 'Failed to detect user emitting IR LED!'

# 2. Test user's real unlit bulb image (Image 1)
p_off = Path('dataset/ir_bulb_off/user_crop_unlit_5mm_bulb.png')
im_off = cv2.imread(str(p_off))
frame_off = np.zeros((480, 640, 3), dtype=np.uint8)
h, w = im_off.shape[:2]
frame_off[100:100+h, 150:150+w] = cv2.cvtColor(im_off, cv2.COLOR_BGR2RGB)
pt_off, _, _, meta_off = det.detect_debug(frame_off)

print('\n=== Test User Reference Unlit 5mm Bulb ===')
print('Detected Point:', pt_off)
print('Status:', meta_off['status'])
print('Purple Excess:', f"+{meta_off['purple_excess']:.1f}")
assert pt_off is None, 'Failed: Unlit bulb was detected as emitting light!'

# 3. Test Whiteboard Drawing (pure white light)
frame_white = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.circle(frame_white, (300, 200), 8, (255, 255, 255), -1)
pt_white, _, _, meta_white = det.detect_debug(frame_white)

print('\n=== Test Whiteboard Drawing (White Light) ===')
print('Detected Point:', pt_white)
print('Status:', meta_white['status'])
print('Purple Excess:', f"+{meta_white['purple_excess']:.1f}")
assert pt_white is None, 'Failed: Whiteboard drawing was detected as emitting light!'

print('\nALL END-TO-END VERIFICATION TESTS PASSED PERFECTLY!')
