# # Given the ground truth keypoints already, calculate a full report.
# # https://simpleitk.readthedocs.io/en/master/link_ImageRegistrationMethod3_docs.html
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from metrics import FullReport
# import SimpleITK as sitk

# def command_iteration(method):
#     if method.GetOptimizerIteration() == 0:
#         print("Estimated Scales: ", method.GetOptimizerScales())
#     print(
#         f"{method.GetOptimizerIteration():3} "
#         + f"= {method.GetMetricValue():7.5f} "
#         + f": {method.GetOptimizerPosition()}"
#     )

# # Step 1: Load 3D Images and Key Points
# inhale_volume = sitk.ReadImage('data/copd1/copd1_iBHCT.nii.gz')  # Assuming NumPy array for 3D volume
# exhale_volume = sitk.ReadImage('data/copd1/copd1_eBHCT.nii.gz')  # Assuming NumPy array for 3D volume

# # Load ground truth key points for fixed and moving volumes
# inhale_keypoints = np.loadtxt('data/copd1/copd1_300_iBH_xyz_r1.txt')  # NumPy array of (x, y, z) coordinates
# exhale_keypoints = np.loadtxt('data/copd1/copd1_300_eBH_xyz_r1.txt') # NumPy array of (x, y, z) coordinates

# # Step 2: Run 3D Image Registration Algorithm
# # Implement or import your 3D image registration algorithm here
# # Example: Use SimpleITK for rigid registration

# def register_3d_images(fixed_image, moving_image):
#     print("Registration starts...")
    
#     fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
#     moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

#     R = sitk.ImageRegistrationMethod()
#     # Configure registration parameters, similarity metric, optimizer, etc.
#     R = sitk.ImageRegistrationMethod()

#     R.SetMetricAsCorrelation()

#     R.SetOptimizerAsRegularStepGradientDescent(
#         learningRate=2.0,
#         minStep=1e-4,
#         numberOfIterations=500,
#         gradientMagnitudeTolerance=1e-8,
#     )
#     R.SetOptimizerScalesFromIndexShift()

#     tx = sitk.CenteredTransformInitializer(
#         fixed_image, moving_image, sitk.Similarity3DTransform()
#     )
#     R.SetInitialTransform(tx)

#     R.SetInterpolator(sitk.sitkLinear)

#     R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

#     outTx = R.Execute(fixed_image, moving_image)

#     print("-------")
#     print(outTx)
#     print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
#     print(f" Iteration: {R.GetOptimizerIteration()}")
#     print(f" Metric value: {R.GetMetricValue()}")

#     sitk.WriteTransform(outTx, 'test.txt')

#     resampler = sitk.ResampleImageFilter()
#     resampler.SetinhaleImage(fixed_image)
#     resampler.SetInterpolator(sitk.sitkLinear)
#     resampler.SetDefaultPixelValue(100)
#     resampler.SetTransform(outTx)

#     out = resampler.Execute(moving_image)

#     simg1 = sitk.Cast(sitk.RescaleIntensity(fixed_image), sitk.sitkUInt8)
#     simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
#     cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)

#     sitk.WriteImage(cimg, 'test.nii.gz')
#     return cimg

# # Convert NumPy arrays to SimpleITK images
# # inhale_image_sitk = sitk.GetImageFromArray(inhale_volume)
# # exhale_image_sitk = sitk.GetImageFromArray(exhale_volume)

# # Run 3D registration
# exhale_image_registered_sitk = register_3d_images(inhale_volume, exhale_volume)

# # Convert the registered image back to a NumPy array
# exhale_image_registered = sitk.GetArrayFromImage(exhale_image_registered_sitk)

# # Step 3: Instantiate and Use FullReport Class
# algorithm_name = 'YourAlgorithm'
# inhale_image_name = 'inhaleImage'
# log_directory = 'training-runs/'

# full_report = FullReport(algorithm_name, inhale_image_name, logdir=log_directory)

# # Call the measure method with ground truth key points
# full_report.measure(
#     exhale_image_registered,
#     inhale_volume,
#     inhale_moving_point_list=exhale_keypoints,
#     exhale_fixed_point_list=inhale_keypoints
# )
