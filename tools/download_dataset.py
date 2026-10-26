import os
import sys
import json
import time
import shutil
import urllib.request
import urllib.parse
from pathlib import Path
import cv2
import numpy as np

# Ensure stdout flushes immediately
sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
IR_EMITTING_DIR = DATASET_DIR / "ir_led_emitting"
IR_OFF_DIR = DATASET_DIR / "ir_bulb_off"
OTHER_LIGHTS_DIR = DATASET_DIR / "other_lights"

USER_IMG_BULB = Path(r"C:\Users\prana\.gemini\antigravity\brain\742360c1-85ca-4369-a3d9-0dea5a3a94ec\.user_uploaded\media_1793022375427_0d7d756f.png")
USER_IMG_EMITTING = Path(r"C:\Users\prana\.gemini\antigravity\brain\742360c1-85ca-4369-a3d9-0dea5a3a94ec\.user_uploaded\media_1793022782230_17282914.png")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
}

QUERIES = {
    "ir_led_emitting": [
        "infrared LED",
        "remote control infrared",
        "Apple Remote Infrared",
        "infrarot LED",
        "infrared light diode"
    ],
    "ir_bulb_off": [
        "5mm LED",
        "through hole LED",
        "light emitting diode 5mm",
        "diode LED transparent",
        "LED component"
    ],
    "other_lights": [
        "electric light",
        "lamp",
        "chandelier",
        "fluorescent lamp",
        "bulb",
        "flashlight",
        "incandescent light bulb",
        "halogen lamp"
    ]
}


def fetch_wikimedia_urls(query: str, limit: int = 15) -> list:
    """Fetch image URLs from Wikimedia Commons API."""
    urls = []
    try:
        url = (
            "https://commons.wikimedia.org/w/api.php?"
            "action=query&generator=search&"
            f"gsrsearch={urllib.parse.quote(query)}&"
            f"gsrnamespace=6&gsrlimit={limit}&"
            "prop=imageinfo&iiprop=url&format=json"
        )
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode('utf-8', errors='ignore'))
            pages = data.get('query', {}).get('pages', {})
            for page in pages.values():
                info = page.get('imageinfo', [])
                if info and 'url' in info[0]:
                    img_url = info[0]['url']
                    path = urllib.parse.urlparse(img_url).path
                    if any(path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                        urls.append(img_url)
    except Exception as e:
        print(f"  [Error searching '{query}']: {e}")
    return urls


def download_file(url: str, dest_path: Path) -> bool:
    """Download image with timeout and size check."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=6) as resp:
            content = resp.read()
            if len(content) < 3000:
                return False
            arr = np.frombuffer(content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None or img.shape[0] < 30 or img.shape[1] < 30:
                return False
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, 'wb') as f:
                f.write(content)
            return True
    except Exception:
        return False


def copy_and_augment_user_references():
    """Copy user reference images and generate synthetic variations."""
    IR_EMITTING_DIR.mkdir(parents=True, exist_ok=True)
    IR_OFF_DIR.mkdir(parents=True, exist_ok=True)

    # User reference emitting image
    if USER_IMG_EMITTING.exists():
        shutil.copy(USER_IMG_EMITTING, IR_EMITTING_DIR / "user_ref_emitting_ground_truth.png")
        img_emit = cv2.imread(str(USER_IMG_EMITTING))
        if img_emit is not None:
            h, w = img_emit.shape[:2]
            aug_idx = 1
            for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
                for scale in [0.75, 1.0, 1.25]:
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, scale)
                    rotated = cv2.warpAffine(img_emit, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                    cv2.imwrite(str(IR_EMITTING_DIR / f"aug_emit_{aug_idx:03d}.png"), rotated)
                    aug_idx += 1

                    for bright_mult in [0.85, 1.15]:
                        brightened = np.clip(rotated.astype(np.float32) * bright_mult, 0, 255).astype(np.uint8)
                        cv2.imwrite(str(IR_EMITTING_DIR / f"aug_emit_{aug_idx:03d}.png"), brightened)
                        aug_idx += 1

    # User reference bulb off image
    if USER_IMG_BULB.exists():
        shutil.copy(USER_IMG_BULB, IR_OFF_DIR / "user_ref_bulb_off.png")
        img_bulb = cv2.imread(str(USER_IMG_BULB))
        if img_bulb is not None:
            h, w = img_bulb.shape[:2]
            crop_unlit = img_bulb[:int(h * 0.75), :int(w * 0.6)]
            cv2.imwrite(str(IR_OFF_DIR / "user_crop_unlit_5mm_bulb.png"), crop_unlit)
            crop_inset = img_bulb[int(h * 0.45):, int(w * 0.2):int(w * 0.55)]
            if crop_inset.size > 0:
                cv2.imwrite(str(IR_EMITTING_DIR / "user_crop_inset_bulb.png"), crop_inset)

            # Augment unlit 5mm bulb with rotations
            for ang in [45, 90, 135, 180, 225, 270]:
                M = cv2.getRotationMatrix2D((crop_unlit.shape[1]//2, crop_unlit.shape[0]//2), ang, 1.0)
                rot = cv2.warpAffine(crop_unlit, M, (crop_unlit.shape[1], crop_unlit.shape[0]), borderMode=cv2.BORDER_REFLECT)
                cv2.imwrite(str(IR_OFF_DIR / f"aug_unlit_{ang}.png"), rot)

    # 3. Create synthetic negative samples: Whiteboard drawings on dark background and screen glares
    OTHER_LIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(1, 16):
        synth = np.zeros((120, 120, 3), dtype=np.uint8)
        # Random white strokes (R=G=B=255, simulating whiteboard lines and dots)
        if i % 3 == 0:
            # White dot / circle
            cv2.circle(synth, (60, 60), 4 + i, (255, 255, 255), -1)
        elif i % 3 == 1:
            # White line / stroke
            cv2.line(synth, (20, 20 + i*4), (100, 100 - i*2), (255, 255, 255), 3 + (i % 4))
        else:
            # Warm incandescent / yellowish room reflection (R=255, G=240, B=180)
            cv2.circle(synth, (60, 60), 10 + i, (180, 240, 255), -1)
        cv2.imwrite(str(OTHER_LIGHTS_DIR / f"synth_whiteboard_white_{i:02d}.png"), synth)


def download_dataset():
    print("=" * 60)
    print("DOWNLOADING & PREPARING IR LED DATASET (~100+ IMAGES)")
    print("=" * 60)

    copy_and_augment_user_references()

    targets = {
        "ir_led_emitting": 60,
        "ir_bulb_off": 25,
        "other_lights": 35
    }

    for category, queries in QUERIES.items():
        dest_dir = DATASET_DIR / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        current = len(list(dest_dir.glob("*.jpg"))) + len(list(dest_dir.glob("*.png")))
        target = targets[category]
        print(f"\n[Category: {category}] Current: {current}, Target: {target}")

        if current >= target:
            print(f"  Target already met for {category} ({current} images).")
            continue

        seen = set()
        for q in queries:
            if current >= target:
                break
            print(f"  Searching '{q}'...")
            urls = fetch_wikimedia_urls(q, limit=12)
            for u in urls:
                if u in seen:
                    continue
                seen.add(u)
                ext = Path(urllib.parse.urlparse(u).path).suffix or ".jpg"
                dest_file = dest_dir / f"web_{category}_{current + 1:03d}{ext}"
                if download_file(u, dest_file):
                    current += 1
                    print(f"    [OK] Downloaded ({current}/{target}): {dest_file.name}")
                if current >= target:
                    break
                time.sleep(0.15)

    cnt_emit = len(list(IR_EMITTING_DIR.glob("*.*")))
    cnt_off = len(list(IR_OFF_DIR.glob("*.*")))
    cnt_other = len(list(OTHER_LIGHTS_DIR.glob("*.*")))
    total_imgs = cnt_emit + cnt_off + cnt_other

    print("\n" + "=" * 60)
    print("DATASET COLLECTION COMPLETE:")
    print(f"  - Active Emitting IR LEDs (Class 1)   : {cnt_emit} images")
    print(f"  - Unlit 5mm LED Bulbs                 : {cnt_off} images")
    print(f"  - Other Lights (White/Lamps/Torches)  : {cnt_other} images")
    print(f"  - Total Dataset Size                  : {total_imgs} images")
    print("=" * 60)


if __name__ == "__main__":
    download_dataset()
