import datetime
import csv
import time
from sklearn.metrics import mean_squared_error, mutual_info_score, r2_score
from scipy import linalg
from scipy.signal import correlate
import numpy as np
import math
import os 

# def accuracy_sensitivity_specificity(kp1, kp2, tolerance=1e-5):
#     """
#     Compute accuracy, sensitivity, and specificity between x, y, z points of two sets of keypoints.

#     Parameters:
#     - kp1, kp2: Numpy arrays or lists representing keypoints. Each keypoint is assumed to have (x, y, z) coordinates.
#     - tolerance: Tolerance level for considering a match.

#     Returns:
#     - accuracy: Accuracy score between the two sets of keypoints.
#     - sensitivity: Sensitivity (true positive rate).
#     - specificity: Specificity (true negative rate).
#     """
#     assert len(kp1) == len(kp2), "Key point sets must have the same length."
#     num_matches = sum(np.all(np.abs(np.array(p1) - np.array(p2)) < tolerance) for p1, p2 in zip(kp1, kp2))
    
#     true_positives = num_matches
#     true_negatives = len(kp1) - num_matches

#     # Assuming positive class represents a match
#     total_positives = len(kp1)
#     total_negatives = len(kp1)

#     accuracy = num_matches / len(kp1)
#     sensitivity = true_positives / total_positives
#     specificity = true_negatives / total_negatives

#     return accuracy, sensitivity, specificity


def dice3d(im1, im2):
    """
    Compute Dice score of two 3D images.

    Parameters:
    - im1, im2: Numpy arrays representing binary masks.

    Returns:
    - dice_score: Dice score.
    """
    im1 = im1.astype(np.float32)  # Convert to float32
    im2 = im2.astype(np.float32)  # Convert to float32

    intersection = np.sum(im1 * im2)
    total_sum = np.sum(im1) + np.sum(im2)
    
    if total_sum == 0:
        return 1.0  # Dice is defined as 1 when both masks are empty.

    dice_score = 2.0 * intersection / total_sum
    return dice_score / np.max(intersection) # Return normalized ? 


def euclidean_distance_3d(xyz1, xyz2, voxel_spacing):
    """
    Calculate the 3D Euclidean distance between two points.

    Parameters:
    - xyz1, xyz2: Tuple or list representing (x, y, z) coordinates of the points.

    Returns:
    - distance: Euclidean distance between the points.
    """
    if voxel_spacing is not None:
        return math.sqrt(((xyz2[0] - xyz1[0]) * voxel_spacing[0] )**2 + ((xyz2[1] - xyz1[1]) * voxel_spacing[1] )**2 + ((xyz2[2] - xyz1[2]) * voxel_spacing[2])**2)
    else:
        return math.sqrt((xyz2[0] - xyz1[0])**2 + (xyz2[1] - xyz1[1])**2 + (xyz2[2] - xyz1[2])**2)

def cross_correlation(kp1, kp2):
    """
    Calculate the cross-correlation between two 3D images.

    Parameters:
    - image1, image2: Numpy arrays representing the 3D images.

    Returns:
    - cross_corr: Cross-correlation value.
    """
    # Ensure that the input images are of the same shape
    assert len(kp1) == len(kp2), "Key point sets must have the same length."


    # Calculate the cross-correlation using numpy's correlate function
    cross_corr = correlate(kp1, kp2, mode='full', method='auto')

    # Normalize the cross-correlation using the standard deviations
    norm_cross_corr = cross_corr / linalg.norm(cross_corr)

    # The maximum value of the cross-correlation result corresponds to the correlation peak
    mean_corr = np.mean(norm_cross_corr)
    std_corr = np.std(norm_cross_corr)

    return mean_corr, std_corr

class FullReport():

    def __init__(self, algorithm_name, image_name, logdir='training-runs/', computational_time=None):

        # Set file name as date of today gg-mm-yyyy
        self.logfile_path = f"{logdir}log_{algorithm_name}.csv"

        if not os.path.exists(self.logfile_path): 
            # Open the file and set algorithm information
            with open(self.logfile_path, mode='a', newline='') as file:
                fieldnames = ['Image', 'Metric', 'Value']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()

        # Take the computational time info
        self.start_time = time.time() if computational_time is None else computational_time

        # Save inhale image name
        self.image_name = image_name

        # Save algorithm name
        self.algorithm_name = algorithm_name

    def _log_metric(self, metric_name, metric_value):

        with open(self.logfile_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.image_name, metric_name, metric_value])

    def measure(self, 
                exhale_image, 
                exhale_keypoint_list, 
                inhale_image = None, 
                pred_image = None, 
                inhale_keypoint_list = None, 
                pred_keypoint_list = None,
                voxel_spacing = None):

        if voxel_spacing is None:
            print("Careful! Calculation will be done without physical distance information!...")

        if pred_image is None and inhale_image is None:
            return ValueError("One of the images should be provided to do report.")

        if pred_image is None: 
            pred_image = inhale_image

        # dice_coefficient = dice3d(exhale_image, pred_image)

        if pred_keypoint_list is None:
            pred_keypoint_list = inhale_keypoint_list

        # Use imported metrics to measure and log
        # accuracy, sensitivity, specificity = accuracy_sensitivity_specificity(exhale_keypoint_list, pred_keypoint_list)

        norm_error = [ 
            linalg.norm(np.array(p_fixed) - np.array(p_moving))
        for p_fixed, p_moving in zip(
            exhale_keypoint_list, pred_keypoint_list
        )
        ]

        tre = [ 
            euclidean_distance_3d(p_fixed, p_moving, voxel_spacing=voxel_spacing)
        for p_fixed, p_moving in zip(
            exhale_keypoint_list, pred_keypoint_list
        )
        ]
        
        mean_corr, std_corr = cross_correlation(exhale_keypoint_list, pred_keypoint_list)
        r2 = r2_score(exhale_keypoint_list, pred_keypoint_list)
        mse = mean_squared_error(exhale_keypoint_list, pred_keypoint_list)
        nmi = np.mean([mutual_info_score(exhale_keypoint_list[:, i], pred_keypoint_list[:, i]) for i in range(3)])

        # Log the metrics
        # self._log_metric('Accuracy', accuracy)
        # self._log_metric('Sensitivity', sensitivity)
        # self._log_metric('Specifity', specificity)

        self._log_metric('Normalized Mutual Information', nmi)
        # self._log_metric('Dice Coefficient', dice_coefficient)
        self._log_metric('R2', r2)
        self._log_metric('MSE', mse)
        self._log_metric('Mean TRE', np.mean(tre))
        self._log_metric('Standard Deviation TRE', np.std(tre))
        self._log_metric('Mean Norm Error', np.mean(norm_error))
        self._log_metric('Standard Deviation Norm Error', np.std(norm_error))
        self._log_metric('Mean Cross Correlation', mean_corr)
        self._log_metric('Standard Deviation Cross Correlation', std_corr)

        # Calculate computational time and log
        elapsed_time = time.time() - self.start_time
        self._log_metric('Computational Time (s)', elapsed_time)


if __name__ == '__main__':
    import SimpleITK as sitk

    algorithm = 'elastix-custom-prep-mask-mirc'
    for i, image_name in enumerate(['copd1', 'copd2', 'copd3', 'copd4']):
        # Step 1: Load 3D Images and Key Points
        inhale_volume = sitk.ReadImage(f'data/preprocessed/{image_name}/{image_name}_iBHCT.nii.gz')  # Assuming NumPy array for 3D volume
        exhale_volume = sitk.ReadImage(f'data/preprocessed/{image_name}/{image_name}_eBHCT.nii.gz')  # Assuming NumPy array for 3D volume
        try:
            pred_volume = sitk.ReadImage(f'{algorithm}/{image_name}/result.1.nii')  # Assuming NumPy array for 3D volume
            pred_volume = sitk.GetArrayFromImage(pred_volume)
        except Exception as e:
            pred_volume = None

        inhale_volume = sitk.GetArrayFromImage(inhale_volume)
        exhale_volume = sitk.GetArrayFromImage(exhale_volume)

        # Load ground truth key points for fixed and moving volumes
        inhale_keypoints = np.loadtxt(f'data/raw/{image_name}/{image_name}_300_iBH_xyz_r1.txt')  # NumPy array of (x, y, z) coordinates
        exhale_keypoints = np.loadtxt(f'data/raw/{image_name}/{image_name}_300_eBH_xyz_r1.txt') # NumPy array of (x, y, z) coordinates
        
        try:
            pred_keypoints = np.loadtxt(f'{algorithm}/{image_name}/output_coordinates.txt') # NumPy array of (x, y, z) coordinates
        except Exception as e:
            pred_keypoints = None

        voxel_spacing = np.loadtxt('voxel_spacing.txt')

        full_report = FullReport(algorithm_name= algorithm, image_name = f'{image_name}')

        # Call the measure method with ground truth key points
        full_report.measure(
            exhale_image=exhale_volume,
            inhale_image=inhale_volume,
            pred_image=pred_volume,
            inhale_keypoint_list=inhale_keypoints,
            exhale_keypoint_list=exhale_keypoints,
            pred_keypoint_list=pred_keypoints,
            voxel_spacing=voxel_spacing[i],
        )