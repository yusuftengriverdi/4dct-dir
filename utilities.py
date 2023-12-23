
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import SimpleITK as sitk

def registration_errors(
    tx,
    reference_fixed_point_list,
    reference_moving_point_list,
    display_errors=False,
    min_err=None,
    max_err=None,
    figure_size=(8, 6),
):
    """
    Distances between points transformed by the given transformation and their
    location in another coordinate system. When the points are only used to
    evaluate registration accuracy (not used in the registration) this is the
    Target Registration Error (TRE).

    Args:
        tx (SimpleITK.Transform): The transform we want to evaluate.
        reference_fixed_point_list (list(tuple-like)): Points in fixed image
                                                       cooredinate system.
        reference_moving_point_list (list(tuple-like)): Points in moving image
                                                        cooredinate system.
        display_errors (boolean): Display a 3D figure with the points from
                                  reference_fixed_point_list color corresponding
                                  to the error.
        min_err, max_err (float): color range is linearly stretched between min_err
                                  and max_err. If these values are not given then
                                  the range of errors computed from the data is used.
        figure_size (tuple): Figure size in inches.

    Returns:
     (mean, std, min, max, errors) (float, float, float, float, [float]):
      TRE statistics and original TREs.
    """
    transformed_fixed_point_list = [
        tx.TransformPoint(p) for p in reference_fixed_point_list
    ]

    errors = [
        linalg.norm(np.array(p_fixed) - np.array(p_moving))
        for p_fixed, p_moving in zip(
            transformed_fixed_point_list, reference_moving_point_list
        )
    ]
    min_errors = np.min(errors)
    max_errors = np.max(errors)
    if display_errors:
        import matplotlib.pyplot as plt
        import matplotlib

        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111, projection="3d")
        if not min_err:
            min_err = min_errors
        if not max_err:
            max_err = max_errors

        collection = ax.scatter(
            list(np.array(reference_fixed_point_list).T)[0],
            list(np.array(reference_fixed_point_list).T)[1],
            list(np.array(reference_fixed_point_list).T)[2],
            marker="o",
            c=errors,
            vmin=min_err,
            vmax=max_err,
            cmap=matplotlib.cm.hot,
            label="fixed points",
        )
        plt.colorbar(collection, shrink=0.8)
        plt.title("registration errors in mm", x=0.7, y=1.05)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    return (np.mean(errors), np.std(errors), min_errors, max_errors, errors)
