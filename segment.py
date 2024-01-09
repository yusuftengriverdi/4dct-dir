import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import morphology
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import click

# This code script was taken from following repo. We do not claim any rights and we inplace here for educational and utility purposes only. 
# Ref: https://github.com/manasikattel/chest_ct_registration/tree/main            

thispath = Path.cwd().resolve()

def check_fov(img, threshold=-975):
    """

    Parameters
    ----------
    img (numpy): Image data
    threshold (int): threshold to detect the fov

    Returns
    -------
    answer (bool): True if there is FOV False if not
    """
    copy_img = img.copy()
    copy_img = copy_img[25, :, :]
    width, height = copy_img.shape
    top_left_corner = np.mean(copy_img[0:5, 0:5])
    top_right_corner = np.mean(copy_img[0:5, width - 5:width])
    bottom_left_corner = np.mean(copy_img[height - 5:height, 0:5])
    bottom_right_corner = np.mean(copy_img[height - 5:height, width - 5:width])

    # Check if there is FOV in at least 3 corners
    return int(top_left_corner < threshold) + int(top_right_corner < threshold) + int(bottom_left_corner < threshold)\
           + int(bottom_right_corner < threshold) > 2


def segment_kmeans(image, K=3, attempts=10):
    """
    Segment image using k-means algorithm; works for all shapes of image.


    Parameters
    ----------
    image : ndarray
        Input image.
    K : int, optional
        The number of classes to segment, by default 3
    attempts : int, optional
        The number of times the algorithm is executed using different 
        initial labellings, by default 10

    Returns
    -------
    ndarray
        Segmented output image.
    """
    image_inv = 255 - image
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    vectorized = image_inv.flatten()
    vectorized = np.float32(vectorized) / 255

    ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts,
                                    cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center * 255)
    res = center[label.flatten()]
    result_image = res.reshape((image.shape))
    return result_image


def remove_small_objects(lung_only, vis_each_slice=False):
    """
    Remove small objects from the binary image; 
    process images slice by slice.

    Parameters
    ----------
    lung_only : ndarray
        Binary image representing the mask of the lung.
    vis_each_slice : bool, optional
        Flag to visualize, by default False

    Returns
    -------
    ndarray
        Image after removing small objects.
    """
    lung_only = lung_only.astype(np.uint8)
    filled_image = np.zeros_like(lung_only)
    for i, slice in enumerate(lung_only):
        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(
            slice)
        sizes = stats[:, -1]
        sizes = sizes[1:]
        nb_blobs -= 1
        min_size = 150
        area_filtered = np.zeros_like(slice)
        # for every component in the image, keep it only if it's above min_size
        for blob in range(nb_blobs):
            if sizes[blob] >= min_size:
                # see description of im_with_separated_blobs above
                area_filtered[im_with_separated_blobs == blob + 1] = 255

        # kernel = np.ones((9, 9), np.uint8)
        # im_result = cv2.morphologyEx(area_filtered, cv2.MORPH_CLOSE, kernel)
        # im_result = cv2.morphologyEx(im_result, cv2.MORPH_DILATE, kernel)

        filled_image[i, :, :] = area_filtered
        if vis_each_slice:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            # # Plot the left lung mask
            ax[0].imshow(slice, cmap="gray")
            ax[0].set_title("slice")

            # # Plot the right lung mask
            ax[1].imshow(area_filtered, cmap="gray")
            ax[1].set_title("mask")

            # # Show the figure
            plt.show()
    return filled_image


def remove_small_3D(lung_only, vis_each_slice=False):
    """
    Remove small objects from the binary image; 
    process images directly with 3D operations.

    Parameters
    ----------
    lung_only : ndarray
        Binary image representing the mask of the lung.
    vis_each_slice : bool, optional
        Flag to visualize, by default False
    Returns
    -------
    ndarray
        Image after removing small objects.

    """
    lung_only = lung_only.astype(np.uint8)
    width = 50

    remove_holes = morphology.remove_small_holes(lung_only,
                                                 area_threshold=width**3)
    remove_objects = morphology.remove_small_objects(remove_holes,
                                                     min_size=width**3)

    return remove_objects

def fill_chest_cavity(image, vis_each_slice=False):
    """
    Fill the chest cavity to obtain the final gantry mask

    Parameters
    ----------
    image : ndarray
        Input image
    vis_each_slice : bool, optional
        Boolean to choose whether to visualize each slice when processing,
         by default False

    Returns
    -------
    ndarray
        Chest cavity filled image
    """
    image = image.astype(np.uint8)
    filled_image = np.zeros_like(image)
    for i, slice in enumerate(image):
        all_objects, hierarchy = cv2.findContours(slice, cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)
        # # Segmented mask
        # # select largest area (should be the skin lesion)
        mask = np.zeros(slice.shape, dtype="uint8")
        area = [cv2.contourArea(object_) for object_ in all_objects]
        if len(area) == 0:
            continue
        index_contour = area.index(max(area))
        cv2.drawContours(mask, all_objects, index_contour, 255, -1)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        filled_image[i, :, :] = mask

        if vis_each_slice:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            # # Plot the left lung mask
            ax[0].imshow(slice, cmap="gray")
            ax[0].set_title("slice")

            # # Plot the right lung mask
            ax[1].imshow(mask, cmap="gray")
            ax[1].set_title("mask")

            # # Show the figure
            plt.show()
    return filled_image / 255


def remove_gantry(image, segmented, visualize=True):
    """
    Remove the gantry in the orginal CT image.

    Parameters
    ----------
    image : ndarray
        Original Image.
    segmented : ndarray
        Mask of the gantry.
    visualize : bool, optional
        Flag to visualize after removal., by default True

    Returns
    -------
    ndarray
        Gantry removed image.
    """
    gantry_mask = segmented * (segmented == np.amin(segmented))
    contours = fill_chest_cavity(gantry_mask, vis_each_slice=False)
    removed = np.multiply(image, contours)
    if visualize:
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))

        # # Plot the left lung mask
        ax[0].imshow(image[60, :, :], cmap="gray")
        ax[0].set_title("image")

        # # Plot the right lung mask
        ax[1].imshow(contours[60, :, :], cmap="gray")
        ax[1].set_title("mask")

        ax[2].imshow(removed[60, :, :], cmap="gray")
        ax[2].set_title("removed")

        # # Show the figure
        plt.show()
    return removed, contours


def get_lung_segmentation(segmented, gantry_mask, visualize=False):
    """
    Extract lung masks from the masks received from kmeans segmentation.
    Removes the small objects, and fills holes.

    Parameters
    ----------
    segmented : ndarray
        segmentation image
    gantry_mask : ndarray
        Mask of gantry
    visualize : bool, optional
        Flag to visualize the segmentation mask, by default False

    Returns
    -------
    ndarray
        Lung mask.
    """
    lung = segmented * gantry_mask
    lung_only = lung * (lung == np.amax(lung))
    holes_filled = remove_small_3D(lung_only, False)

    kernel = morphology.ball(6)
    closed = morphology.closing(holes_filled, kernel)
    dilated = morphology.dilation(closed, kernel)

    if visualize:
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))

        # # Plot the left lung mask
        ax[0].imshow(lung_only[60, :, :], cmap="gray")
        ax[0].set_title("segmented")

        # # Plot the right lung mask
        ax[1].imshow(holes_filled[60, :, :], cmap="gray")
        ax[1].set_title("gantry_mask")

        ax[2].imshow(holes_filled[60, :, :], cmap="gray")
        ax[2].set_title("lung_only")
        plt.show()

    return dilated


def main(dataset_option='preprocessed_challenge',
         mask_creation=True,
         save_gantry_removed=True,
         save_lung_mask=True):
    datadir = thispath / Path(f"data/{dataset_option}")
    images_files = [i for i in datadir.rglob("*.nii.gz") if "copd0" in str(i)]
    results_dir = Path(f"data/{dataset_option}_gantry_removed")
    results_dir.mkdir(parents=True, exist_ok=True)
    # Read the chest CT scan
    for image_file in tqdm(images_files):
        ct_image = sitk.ReadImage(str(image_file))
        img_255 = sitk.Cast(sitk.RescaleIntensity(ct_image), sitk.sitkUInt8)
        seg_img = sitk.GetArrayFromImage(img_255)

        if check_fov(sitk.GetArrayFromImage(ct_image)):
            segmented = segment_kmeans(seg_img)
            print("\nFov presence: True")
        else:
            segmented = segment_kmeans(seg_img, K=2)
            print("\nFov presence: False")

        removed, gantry_mask = remove_gantry(seg_img,
                                             segmented,
                                             visualize=False)

        if save_lung_mask:
            lung_mask = get_lung_segmentation(segmented, gantry_mask)
            lung_mask = sitk.GetImageFromArray(lung_mask.astype(np.uint8))
            lung_mask.CopyInformation(ct_image)
            dataset_ = dataset_option.split('_')[0]
            save_dir = thispath / Path(
                f"data/{dataset_}_segmentations/{Path(image_file.parent.name)}"
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(
                lung_mask,
                str(Path(save_dir / f'seg_lung_ours_{image_file.name}')))

        if save_gantry_removed:
            removed_sitk = sitk.GetImageFromArray(removed)
            removed_sitk.CopyInformation(ct_image)
            save_dir = thispath / Path(
                f"{results_dir}/{Path(image_file.parent.name)}"
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(removed_sitk,
                            str(Path(save_dir / f'{image_file.name}')))

        if mask_creation:
            img_corr = sitk.GetImageFromArray(gantry_mask)
            img_corr.CopyInformation(ct_image)
            save_dir = thispath / Path(
                f"data/train_segmentations/{Path(image_file.parent.name)}")
            save_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(
                img_corr, str(Path(save_dir / f'seg_body_{image_file.name}')))


if __name__ == "__main__":
    main()
