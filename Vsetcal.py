import cv2
import numpy as np

img = cv2.imread(r'F:\Languages\DIP\est1\drishtiGS_010.png', cv2.IMREAD_GRAYSCALE)
mask_od = cv2.imread(r'F:\Languages\DIP\est1\drishtiGS_010_ODsegSoftmap.png', cv2.IMREAD_GRAYSCALE)
mask_oc = cv2.imread(r'F:\Languages\DIP\est1\drishtiGS_010_cupsegSoftmap.png', cv2.IMREAD_GRAYSCALE)
img=cv2.resize(img,dsize=(512,512))
mask_oc=cv2.resize(mask_oc,dsize=(512,512))
mask_od=cv2.resize(mask_od,dsize=(512,512))

if img is None or mask_od is None or mask_oc is None:
    print("Error: Could not load one or more images. Check your paths.")
else:
    od_px = img[mask_od >= 127]
    oc_px = img[mask_oc >= 127]

    if od_px.size > 0 and oc_px.size > 0:
        v_od_min = int(np.percentile(od_px, 5))
        v_od_max = int(np.percentile(od_px, 95))

        v_oc_min = int(np.percentile(oc_px, 5))
        v_oc_max = int(np.percentile(oc_px, 95))

        print(f"--- V-set Analysis (Percentile Method) ---")
        print("-" * 40)
        print(f"Optic Disc V-set: {v_od_min} to {v_od_max}")
        print(f"Optic Cup V-set:  {v_oc_min} to {v_oc_max}")
        print("-" * 40)
    else:
        print("Error: Masks are empty or do not contain enough pixels.")