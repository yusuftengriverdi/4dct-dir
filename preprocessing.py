import numpy as np
import SimpleITK as sitk
from skimage.exposure import equalize_adapthist, rescale_intensity
from tqdm import tqdm

def prepro(image):
    """
    Do image preprocessing, make the image [height, width, depth] by moving z-axis to the last index.
    """

    array_image = sitk.GetArrayFromImage(image)
    # Background zeroing.
    array_image[array_image == -2000] = 0
    # Normalize the image
    array_image = array_image - np.min(array_image)
    normalized_image = array_image / np.max(array_image)

    # Convert to a range of [0, 1] for CLAHE
    image_array_01 = rescale_intensity(normalized_image, out_range=(0, 1))
    
    # Apply adaptive histogram equalization
    eq_img = equalize_adapthist(image_array_01)

    min_val, max_val = array_image.min(), array_image.max()
    # Convert back to original range
    image_clahe_original_range = rescale_intensity(eq_img, out_range=(min_val, max_val))
    # Apply a priori threshold?
    image_clahe_original_range[image_clahe_original_range <=22] = 0

    final_img = sitk.GetImageFromArray(image_clahe_original_range.astype(array_image.dtype))
    final_img.CopyInformation(image)

    return final_img

def save_numpy_as_nifti(preprocessed_sitk_image, original_sitk_image, output_file_path):
    """
    Converts a NumPy array to a SimpleITK Image and saves it as a NIfTI file.
    """
    # preprocessed_sitk_image.SetSpacing(original_sitk_image.GetSpacing())
    # preprocessed_sitk_image.SetOrigin(original_sitk_image.GetOrigin())
    # preprocessed_sitk_image.SetDirection(original_sitk_image.GetDirection())
    if not output_file_path.endswith('.nii.gz'):
      output_file_path += '.gz'
    sitk.WriteImage(preprocessed_sitk_image, output_file_path)


import matplotlib.pyplot as plt
# Load and process the fixed image

for i, image_name in tqdm(enumerate(['copd1', 'copd2', 'copd3', 'copd4'])):

    fixed_image = sitk.ReadImage(f"data/{image_name}/{image_name}_eBHCT.nii.gz")
    image_data = prepro(fixed_image)

    # Save the processed image
    output_image_path = f"data/preprocessed/{image_name}_eBHCT.nii.gz"
    save_numpy_as_nifti(image_data, fixed_image, output_image_path)


for i, image_name in tqdm(enumerate(['copd1', 'copd2', 'copd3', 'copd4'])):
    fixed_image = sitk.ReadImage(f"data/{image_name}/{image_name}_iBHCT.nii.gz")
    image_data = prepro(fixed_image)

    # Save the processed image
    output_image_path = f"data/preprocessed/{image_name}_iBHCT.nii.gz"
    save_numpy_as_nifti(image_data, fixed_image, output_image_path)