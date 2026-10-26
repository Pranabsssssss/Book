import cv2
import numpy as np
from pathlib import Path

tmpl_dir = Path('models/bulb_templates')
tmpl_dir.mkdir(parents=True, exist_ok=True)

# 1. Base template from user photo 1 (held upright)
im1 = cv2.imread(r'C:/Users/prana/.gemini/antigravity/brain/742360c1-85ca-4369-a3d9-0dea5a3a94ec/.user_uploaded/media_1793025491309_fd5b99af.jpg')
t1 = im1[205:240, 420:450]
t1 = cv2.resize(t1, (26, 30))
cv2.imwrite(str(tmpl_dir / 'user_tip_up.png'), t1)

# 2. Base template from user photo 4 (pointing downward)
im4 = cv2.imread(r'C:/Users/prana/.gemini/antigravity/brain/742360c1-85ca-4369-a3d9-0dea5a3a94ec/.user_uploaded/media_1793025491323_3fb73609.jpg')
t4 = im4[240:285, 570:610]
t4 = cv2.resize(t4, (26, 30))
cv2.imwrite(str(tmpl_dir / 'user_tip_down.png'), t4)

# 3. Generate 8-direction rotations
for base_name, t in [('up', t1), ('down', t4)]:
    h, w = t.shape[:2]
    center = (w // 2, h // 2)
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(t, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(str(tmpl_dir / f'bulb_rot_{base_name}_{angle}.png'), rot)

count = len(list(tmpl_dir.glob('*.png')))
print(f'Generated {count} orientation templates for 5mm LED bulb!')
