import cv2
import numpy as np

V_OD_MIN, V_OD_MAX = 62, 169
V_OC_MIN, V_OC_MAX = 75, 169

def compute_metrics(res_img, gt_im, name):

    if res_img is None or gt_im is None:
        print(f"Error loading {name} images.")
        return

    res_bin = (res_img >= 50)
    gt_bin = (gt_im >= 50)

    true_pixels = np.sum(res_bin & gt_bin)    
    false_pixels = np.sum(res_bin != gt_bin)
    total_gt_pixels = np.sum(gt_bin)

    norm_true = true_pixels / total_gt_pixels if total_gt_pixels > 0 else 0
    norm_false = false_pixels / total_gt_pixels if total_gt_pixels > 0 else 0
    
    dice = (2.0 * true_pixels) / (np.sum(res_bin) + total_gt_pixels) if (np.sum(res_bin) + total_gt_pixels) > 0 else 1.0

    print(f"--- Metrics for {name} ---")
    print(f"True Pixels (Overlap):   {true_pixels}")
    print(f"False Pixels (Mismatch): {false_pixels}")
    print(f"Total Ground Truth Px:   {total_gt_pixels}")
    print(f"Normalized True:         {norm_true:.4f}")
    print(f"Normalized False:        {norm_false:.4f}")
    print(f"Dice Coefficient:        {dice:.4f}\n")


img = cv2.imread(r'F:\Languages\DIP\est1\test_data\Test\Test\Images\glaucoma\drishtiGS_025.png', cv2.IMREAD_GRAYSCALE)
gt_od=cv2.imread(r'F:\Languages\DIP\est1\test_data\Test\Test\Test_GT\drishtiGS_025\SoftMap\drishtiGS_025_ODsegSoftmap.png',cv2.IMREAD_GRAYSCALE)
gt_oc=cv2.imread(r'F:\Languages\DIP\est1\test_data\Test\Test\Test_GT\drishtiGS_025\SoftMap\drishtiGS_025_cupsegSoftmap.png',cv2.IMREAD_GRAYSCALE)
img=cv2.resize(img, dsize=(512,512))
gt_oc=cv2.resize(gt_oc,dsize=(512,512))
gt_od=cv2.resize(gt_od,dsize=(512,512))
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
    compute_metrics(mask_od_final, gt_od, "Optic Disc (OD)")
    compute_metrics(mask_oc_final, gt_oc, "Optic Cup (OC)")
    cv2.imshow("OD mask",mask_od_final)
    cv2.imshow("OC mask", mask_oc_final)
    print("Press any key on the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    