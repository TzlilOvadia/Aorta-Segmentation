# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.measurements as scm
import skimage.morphology
import cv2 as cv2

FAILURE = -1
MAX_VOXEL_VALUE = 65535
MIN_VOXEL_VALUE = 0
DERIVATIVE_KERNEL = [1, -1]


class ImageSegmentation:

    def __init__(self):
        self.img = None
        self.connectivity_cmps = []
        self.result_by_mins = []
        self.minimal_axial_slice = 0
        self.maximal_axial_slice = 0

    def SegmentationByTH(self,nifty_file, Imin, Imax):
        """
        This function is given as inputs a grayscale NIFTI file (.nii.gz) and two integers – the minimal and maximal thresholds. The function generates a segmentation NIFTI file of the same dimensions, with a binary segmentation – 1 for voxels between Imin and Imax, 0 otherwise. This segmentation NIFTI file should be saved under the name <nifty_file>_seg_<Imin>_<Imax>.nii.gz.
        The function returns 1 if successful, 0 otherwise. Preferably, raise descriptive errors when returning 0.
        :param nifty_file:
        :param Imin:
        :param Imax:
        :return:
        """
        MAX_VOXEL_VALUE = 65535
        MIN_VOXEL_VALUE = 0
        img = nib.load(nifty_file)
        img_data = img.get_fdata().astype(dtype=np.uint16)
        img_data[(img_data <= Imax) & (img_data > Imin)] = MAX_VOXEL_VALUE
        img_data[img_data < MAX_VOXEL_VALUE] = MIN_VOXEL_VALUE
        opened_img = skimage.morphology.opening(img_data)
        final_image = nib.Nifti1Image(opened_img, img.affine)
        nib.save(final_image, f"out_seg_{Imin}_{Imax}.nii.gz.")
        self.img = opened_img
        return opened_img

    def SkeletonTHFinder(self,nifty_file):
        """
        This function iterates over 25 candidate Imin thresholds in the range of [150,500] (with intervals of 14).
        In each run, use the SegmentationByTH function you’ve implemented, and count the number of connectivity components
        in the resulting segmentation with the current Imin. Plot your results – number of connectivity components per Imin. Choose the Imin which is the first or second minima in the plot. Also, make sure to include that graph in your report.

        Next, you need to perform post-processing (morphological operations – clean out single pixels, close holes, etc.)
        until you are left with a single connectivity component.
        Finally, this function should save a segmentation NIFTI file called “<nifty_file>_SkeletonSegmentation.nii.gz” and
        return the Imin used for that.
        :return:
        """
        Imin_range = np.arange(150,514,14)
        structure = np.ones((3,3,3))
        self.SegmentationByTH(nifty_file, 220, 1300)

        for i_min in Imin_range:
            self.SegmentationByTH(nifty_file, i_min, 1300)
            labels, cmp = scm.label(self.img, structure)
            self.connectivity_cmps.append(cmp)

        plt.plot(self.connectivity_cmps)
        plt.title("Number of connectivity components vs. Imin value")
        plt.xlabel("Imin Value")
        plt.ylabel("Number of Connectivity Components")
        plt.show()

    def AortaSegmentation(self, nifty_file, L1_seg_nifti_file):
        full_scan, l1_scan = nib.load(nifty_file), nib.load(L1_seg_nifti_file)
        full_scan = nib.as_closest_canonical(full_scan)
        l1_scan = nib.as_closest_canonical(l1_scan)
        full_scan_img = full_scan.get_fdata().astype(dtype=np.uint16)
        l1_img = l1_scan.get_fdata().astype(dtype=np.uint16)
        # find the first ROI for the aorta in the first slice
        curr_rad, curr_x, curr_y,  self.maximal_axial_slice, self.minimal_axial_slice = self.find_first_circle(full_scan_img, l1_img)
        # Mask all pixels out of the found circle
        output = np.zeros(l1_img.shape, dtype=np.uint16)
        # Filter out irrelevant values of HU from the scan - should enhance the aorta comparing to its close environment
        full_scan_img[0 > full_scan_img] = 0
        full_scan_img[250 < full_scan_img] = 0
        max_bounding_col = 0
        # Approximating the initial ROI of the aorta given the l1 constraints:
        deltas, boundaries, boundaries_nonzero_min_ind, boundaries_nonzero_max_ind =\
            self.analyze_l1_posture(l1_img, max_bounding_col)

        max_bounding_col = 0
        # Improving the estimation for the aorta's ROI using image processing methods:
        self.estimate_aorta_ROI_in_all_slices(boundaries, boundaries_nonzero_max_ind, boundaries_nonzero_min_ind,
                                              curr_rad, curr_x, curr_y, deltas, full_scan_img, l1_img, max_bounding_col,
                                              output)
        # Create the output file and write the output to it:
        final_image = nib.Nifti1Image(output, l1_scan.affine)
        nib.save(final_image, f"output_l1.nii.gz.")
        return output

    def estimate_aorta_ROI_in_all_slices(self, boundaries, boundaries_nonzero_max_ind, boundaries_nonzero_min_ind,
                                         curr_rad, curr_x, curr_y, deltas, full_scan_img, l1_img, max_bounding_col,
                                         output):
        """

        :param boundaries:
        :param boundaries_nonzero_max_ind:
        :param boundaries_nonzero_min_ind:
        :param curr_rad:
        :param curr_x:
        :param curr_y:
        :param deltas:
        :param full_scan_img:
        :param l1_img:
        :param max_bounding_col:
        :param output:
        :return:
        """
        for s in range(self.maximal_axial_slice - 1, self.minimal_axial_slice, -1):
            next_slice = full_scan_img[:, :, s]
            row_non_zero, column_non_zero = np.nonzero(l1_img[:, :, s])
            row_non_zero, max_bounding_col = (max(row_non_zero) + min(row_non_zero)) // 2, max(max(column_non_zero),
                                                                                               max_bounding_col)
            if self.maximal_axial_slice - 1 - s < boundaries_nonzero_min_ind:
                max_bounding_col = boundaries[boundaries_nonzero_min_ind]
            elif self.maximal_axial_slice - 1 - s >= boundaries_nonzero_max_ind:
                max_bounding_col += deltas[-1]
            bounds = row_non_zero, max_bounding_col
            curr_x, curr_y, curr_rad, output[:, :, s] = self.find_next_circle((int(curr_x), int(curr_y)), int(curr_rad),
                                                                              next_slice, bounds)

    def analyze_l1_posture(self, l1_img, max_bounding_col):
        """
        given the L1 segment, we will extract the constraints of the case, using the assumption that the aorta's
        location and posture are constrained by the L1 posture, i.e., always in front and by the left of the L1
        vertebrate.
        :param l1_img: the l1 segmentation
        :param max_bounding_col:
        :return:
        """
        boundaries = []
        for s in range(self.maximal_axial_slice, self.minimal_axial_slice, -1):
            row_non_zero, column_non_zero = np.nonzero(l1_img[:, :, s])
            row_non_zero, max_bounding_col = (max(row_non_zero) + min(row_non_zero)) // 2, max(max(column_non_zero),
                                                                                               max_bounding_col)
            boundaries.append(max_bounding_col)
        deltas = np.convolve(boundaries, DERIVATIVE_KERNEL, 'same')
        deltas[0], deltas[-1], anomaly = 0, 0, 0
        deltas_nonzero_max_ind = np.max(np.nonzero(deltas))
        anomalies = np.argwhere(deltas > 15)
        if len(anomalies) > 0:
            anomaly = np.max(anomalies)
            deltas[deltas > 15] = 0
            if 0 < anomaly < deltas_nonzero_max_ind:
                deltas[:anomaly] = round(
                    np.average(deltas[anomaly: min(len(deltas) - 1, anomaly + 5)]))
                assign = int(boundaries[anomaly]) * np.ones((1, int(anomaly)))
                boundaries[:anomaly] = assign[:]
        deltas_nonzero_min_ind = np.min(np.nonzero(deltas))
        deltas[deltas_nonzero_max_ind:] = round(np.average(deltas[max(anomaly+1,deltas_nonzero_max_ind-7): deltas_nonzero_max_ind]))
        return deltas, boundaries, deltas_nonzero_min_ind, anomaly

    def find_first_circle(self, full_scan_img, l1_img):
        """
        function used to locate the first circle segment of the aorta.
        :param full_scan_img:
        :param l1_img:
        :return:
        """
        rows_center_start ,rows_center_aprx, column_center_aprx, maximal_z, minimal_z = self.find_L1_borders(l1_img)
        first_slice = full_scan_img[:, :, maximal_z]
        first_slice[:, :int(column_center_aprx)-20], first_slice[:, int(column_center_aprx) + 20:] = 0, 0
        first_slice[: rows_center_start, :], first_slice[int(rows_center_aprx):, :] = 0, 0
        first_slice, _ = self.frame_preprocessing(first_slice)
        param2 = 40
        while True:
            circles = cv2.HoughCircles(first_slice, method=3, dp=1, minDist=5,
                                       param1=10, param2=param2, minRadius=8, maxRadius=30)
            param2 -= 1
            if circles is not None:
                break

        circles = np.uint16(np.around(circles))
        best_var = np.inf
        # final circle index will be saved here:
        circle = 0
        for i, (x, y, radius) in enumerate(circles[0, :]):
            # generate mask of the estimated aorta's roi in current frame:
            mask = np.zeros(first_slice.shape)
            cv2.circle(mask, (x, y), radius, (255, 255, 255), -1)
            cur_var = np.var(first_slice[mask > 0])
            # Choose the circle with the lowest variance of the pixel grayscale values:
            if cur_var < best_var:
                best_var = cur_var
                circle = i
        aprx_x, aprx_y, aprx_r = circles[0, circle]
        return int(aprx_r), int(aprx_x), int(aprx_y), maximal_z, minimal_z

    def find_L1_borders(self, l1_img):
        """

        :param l1_img:
        :return:
        """
        x_nonzero_in, y_nonzero_in, z_nonzero_in = np.nonzero(l1_img)
        maximal_z, minimal_z = np.max(z_nonzero_in), np.min(z_nonzero_in)
        column_center_stop = np.max(y_nonzero_in)
        rows_center_start = np.min(x_nonzero_in)
        rows_center_stop = (np.max(x_nonzero_in) + np.min(x_nonzero_in))//2
        return rows_center_start ,rows_center_stop, column_center_stop, maximal_z, minimal_z

    def find_next_circle(self, prev_center: tuple[int, int], radius, next_slice, borders):
        """
        given previously detected circle, this function will detect the next ROI's circle containing the aorta.
        :param prev_x_center: previously detected circle center
        :param prev_y_center:
        :param radius:
        :param next_slice:
        :param borders:
        :return:
        """
        mask_to_crop = np.zeros(next_slice.shape, dtype=np.uint8)
        next_slice = next_slice.astype(np.uint8)
        # Process the current frame to get a better performance of HoughCircles to detect circles
        processed_slice, to_be_updated = self.frame_preprocessing(next_slice)
        # Create a mask
        cv2.circle(mask_to_crop, (prev_center[0], prev_center[1]), int(radius*3), (255, 255, 255), -1)
        processed_slice *= mask_to_crop
        circles = cv2.HoughCircles(processed_slice, method=3, dp=1, minDist=2,
                                   param1=5, param2=10, minRadius=radius-3, maxRadius=radius+2)
        best_r, best_x, best_y = self.evaluate_best_circle(borders, circles, prev_center, processed_slice, radius)
        mask = np.zeros(processed_slice.shape, dtype=np.uint8)
        # create a binary mask to capture the predicted ROI for this current CT's slice
        cv2.circle(mask, (int(best_x), int(best_y)), int(best_r), (255, 255, 255), -1)
        # Grab & return the estimated ROI for the aorta
        updated_slice = to_be_updated * mask
        return best_x, best_y, np.round(best_r), updated_slice

    def evaluate_best_circle(self, borders, circles, prev_center, processed_slice, radius):
        best_x, best_y, best_r = borders[1] + radius + 1, borders[0] - radius, radius
        score = -1
        if circles is not None:
            for x, y, rad in circles[0, :]:
                if (borders[1] + rad // 2 < x < borders[1] + 3 * rad) and (borders[0] - 4 * rad < y < borders[0] + rad):
                    prev_aorta_est_mapping = np.zeros(processed_slice.shape, dtype=np.uint8)
                    cv2.circle(prev_aorta_est_mapping, (prev_center[0] + 1, prev_center[1]), radius, (255, 255, 255),
                               -1)
                    curr_aorta_seg_est = np.zeros(processed_slice.shape, dtype=np.uint8)
                    cv2.circle(curr_aorta_seg_est, (int(round(x)), int(round(y))), int(round(rad)), (255, 255, 255), -1)
                    sum_of_volumes = (np.sum(curr_aorta_seg_est) + np.sum(prev_aorta_est_mapping))
                    curr_score = np.sum(prev_aorta_est_mapping[curr_aorta_seg_est > 0]) / sum_of_volumes
                    if curr_score > score:
                        score = curr_score
                        best_x, best_y, best_r = x, y, max(rad, 10)
        return best_r, best_x, best_y

    def frame_preprocessing(self, processed_frame):
        """
        given a frame, this function will process the frame, such that real circles will be more noticeable
        and make HoughCircles give more accurate result.
        :param processed_frame:
        :return:
        """
        to_be_updated = processed_frame.copy()
        processed_frame = cv2.equalizeHist(processed_frame.astype(np.uint8))
        processed_frame = cv2.GaussianBlur(processed_frame, (5, 5), 1.5)
        processed_frame = cv2.Canny(processed_frame, 100, 190, 3)
        processed_frame = cv2.dilate(processed_frame, cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2)), 2)

        return processed_frame, to_be_updated

    def evaluateSegmentation(self, GT_seg, est_seg):
        """
        This function is given two segmentations, a GT one and an estimated one, and returns a
        tuple of (VOD_result, DICE_result).
        :param GT_seg: Ground Truth Segment to be compared with
        :param est_seg: Your estimated segment using AortaSegmentation function
        :return: (VOD_result, DICE_result)
        """
        GT_seg[:,:,:self.minimal_axial_slice-1] = 0
        GT_seg[:, :, self.maximal_axial_slice+1:] = 0
        return self.calc_vod_result(GT_seg, est_seg), self.calc_dice_result(GT_seg, est_seg)

    def calc_dice_result(self, gt_seg, est_seg):
        """

        :param gt_seg: the ground truth segment to compare with
        :param est_seg: the estimated segment to evaluate.
        :return: score between 0 and 1, where 1 is the highest score.
        """
        est_mapping, gt_mapping = self.get_binary_map_for_segments(est_seg, gt_seg)
        sum_of_volumes = (np.sum(est_mapping) + np.sum(gt_mapping))
        return np.sum(gt_mapping[est_mapping == 1]) * 2.0 / sum_of_volumes

    def calc_vod_result (self, gt_seg, est_seg):
        est_mapping, gt_mapping = self.get_binary_map_for_segments(est_seg, gt_seg)
        sum_of_volumes = np.sum(est_mapping) + np.sum(gt_mapping)
        if sum_of_volumes > 0:
            return 1 - np.sum(est_mapping[gt_mapping == 1] == 1) / sum_of_volumes
        return FAILURE

    def get_binary_map_for_segments(self, est_seg, gt_seg):
        gt_mapping, est_mapping = np.zeros(gt_seg.shape), np.zeros(gt_seg.shape)
        gt_mapping[gt_seg != 0], est_mapping[est_seg != 0] = 1, 1
        return est_mapping, gt_mapping


if __name__ == '__main__':
    img_seg = ImageSegmentation()
    # imseg.SkeletonTHFinder("resources/Case1_CT.nii.gz")
    i = 1
    out = img_seg.AortaSegmentation(f"resources/Case{i}_CT.nii.gz", f"resources/Case{i}_L1.nii.gz")
    gt = nib.load(f"resources/Case{i}_Aorta.nii.gz")
    gt = nib.as_closest_canonical(gt)
    gt_seg = gt.get_fdata().astype(dtype=np.uint8)
    print(img_seg.evaluateSegmentation(gt_seg, out))