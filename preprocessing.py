import numpy as np
import SimpleITK as sitk
from skimage.exposure import equalize_adapthist

def prepro(image):
    """
    Do image preprocessing, make the image [height, width, depth] by moving z-axis to the last index.
    """
    realigned_image = np.moveaxis(sitk.GetArrayFromImage(image), 0, -1)
    # Background zeroing.
    realigned_image[realigned_image == -2000] = 0
    # Normalize the image
    realigned_image = realigned_image - np.min(realigned_image)
    normalized = realigned_image / np.max(realigned_image)
    # Apply adaptive histogram equalization
    eq_img = equalize_adapthist(normalized)
    final_img = eq_img.astype(np.float32)
    return final_img

def save_numpy_as_nifti(numpy_array, original_sitk_image, output_file_path):
    """
    Converts a NumPy array to a SimpleITK Image and saves it as a NIfTI file.
    """
    converted_image = sitk.GetImageFromArray(np.moveaxis(numpy_array, -1, 0))
    converted_image.SetSpacing(original_sitk_image.GetSpacing())
    converted_image.SetOrigin(original_sitk_image.GetOrigin())
    converted_image.SetDirection(original_sitk_image.GetDirection())
    if not output_file_path.endswith('.nii.gz'):
      output_file_path += '.gz'
    sitk.WriteImage(converted_image, output_file_path)


import matplotlib.pyplot as plt
# Load and process the fixed image
import os 

for i, image_name in enumerate(['copd1', 'copd2', 'copd3', 'copd4']):

    fixed_image = sitk.ReadImage(f"data/{image_name}/{image_name}_eBHCT.nii.gz")
    image_data = prepro(fixed_image)

    # Save the processed image
    output_image_path = f"data/preprocessed/{image_name}_eBHCT_bg.nii.gz"
    save_numpy_as_nifti(image_data, fixed_image, output_image_path)


for i, image_name in enumerate(['copd1', 'copd2', 'copd3', 'copd4']):
    fixed_image = sitk.ReadImage(f"data/{image_name}/{image_name}_iBHCT.nii.gz")
    image_data = prepro(fixed_image)

    # Save the processed image
    output_image_path = f"data/preprocessed/{image_name}_iBHCT_bg.nii.gz"
    save_numpy_as_nifti(image_data, fixed_image, output_image_path)