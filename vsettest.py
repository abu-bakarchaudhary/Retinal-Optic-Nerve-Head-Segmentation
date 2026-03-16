import cv2
import numpy as np

V_OD_MIN, V_OD_MAX = 56, 185
V_OC_MIN, V_OC_MAX = 70, 190

img = cv2.imread(r'F:\Languages\DIP\est1\training_data\drishtiGS_010.png', cv2.IMREAD_GRAYSCALE)
img=cv2.resize(img, dsize=(512,512))

if img is None:
    print(f"Error: Could not load image ")
else:
    bin_od = np.zeros(img.shape, dtype=np.uint8)
    bin_od[(img >= V_OD_MIN) & (img <= V_OD_MAX)] = 255
    num_od, labels_od, stats_od, _ = cv2.connectedComponentsWithStats(bin_od, connectivity=8)
    
    mask_od_final = np.zeros_like(img)
    if num_od > 1:
        largest_od = 1 + np.argmax(stats_od[1:, cv2.CC_STAT_AREA])
        mask_od_final[labels_od == largest_od] = 255

    bin_oc = np.zeros(img.shape, dtype=np.uint8)
    bin_oc[(img >= V_OC_MIN) & (img <= V_OC_MAX)] = 255
    num_oc, labels_oc, stats_oc, _ = cv2.connectedComponentsWithStats(bin_oc, connectivity=8)
    
    mask_oc_final = np.zeros_like(img)
    if num_oc > 1:
        largest_oc = 1 + np.argmax(stats_oc[1:, cv2.CC_STAT_AREA])
        mask_oc_final[labels_oc == largest_oc] = 255

    cv2.imshow("OD mask",mask_od_final)
    cv2.imshow("OC mask", mask_oc_final)
    print("Press any key on the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()