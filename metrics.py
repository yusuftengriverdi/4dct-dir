import datetime
import csv
import time
from skimage.metrics import structural_similarity, mean_squared_error
from scipy.spatial.distance import dice
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy import linalg
import numpy as np

class FullReportSITK():

    def __init__(self, algorithm_name, reference_image_name, logdir='./logs/training-runs/', computational_time=None):

        # Set file name as date of today gg-mm-yyyy
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.logfile_path = f"{logdir}log_{algorithm_name}_{timestamp}.csv"

        # Open the file and set algorithm information
        with open(self.logfile_path, mode='w', newline='') as file:
            fieldnames = ['Reference Image', 'Metric', 'Value']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

        # Take the computational time info
        self.start_time = time.time() if computational_time is None else computational_time

        # Save reference image name
        self.reference_image_name = reference_image_name

        # Save algorithm name
        self.algorithm_name = algorithm_name

    def _log_metric(self, metric_name, metric_value):

        with open(self.logfile_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.reference_image_name, metric_name, metric_value])

    def measure(self, transformed_image, reference_image, reference_moving_point_list, transformed_fixed_point_list):

        # Use imported metrics to measure and log
        accuracy = accuracy_score(reference_moving_point_list, transformed_fixed_point_list)
        confusion_mat = confusion_matrix(reference_moving_point_list, transformed_fixed_point_list)
        sensitivity = confusion_mat[1, 1] / (confusion_mat[1, 0] + confusion_mat[1, 1])
        specificity = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[0, 1])

        dice_coefficient = dice(transformed_image, reference_image)
        ssim = structural_similarity(transformed_image, reference_image)
        mse = mean_squared_error(transformed_image, reference_image)

        tre = [ 
            linalg.norm(np.array(p_fixed) - np.array(p_moving))
        for p_fixed, p_moving in zip(
            transformed_fixed_point_list, reference_moving_point_list
        )
        ]
        
        # Log the metrics
        self._log_metric('Accuracy', accuracy)
        self._log_metric('Sensitivity', sensitivity)
        self._log_metric('Specificity', specificity)
        self._log_metric('Dice Coefficient', dice_coefficient)
        self._log_metric('SSIM', ssim)
        self._log_metric('MSE', mse)
        self._log_metric('Mean TRE', np.mean(tre))
        self._log_metric('Standard Deviation TRE', np.std(tre))

        # Calculate computational time and log
        elapsed_time = time.time() - self.start_time
        self._log_metric('Computational Time (s)', elapsed_time)
