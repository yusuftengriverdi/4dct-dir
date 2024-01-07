#!/bin/bash

# Record the start time
start_time=$(date +%s)

# Define the paths and parameters
input_dir="data/preprocessed_gantry_removed"

mask_dir="data/preprocessed_segmentations/"

param_affine="params/Parameter.affine.sparse.txt"
param_elastic="params/Parameter.bsplines.sparse.txt"

elastix_output_dir="elastix-custom-prep-nogantry-mask-sparse/"
# transformix_label_output_dir="registeredSet/mni/par0009/registeredLabels/"


# List of image filenames to process
image_list=("copd1" "copd2" "copd3" "copd4")

# Loop through the list and process each image
for image_file in "${image_list[@]}"; do
  # Create the output subdirectory for the current image
  elastix_output_subdir="$elastix_output_dir/$image_file/"

  # Create the 'elastixoutput' directory within the image's output directory
  if [ ! -d "$elastix_output_subdir" ]; then
    mkdir -p "$elastix_output_subdir"
  fi

  # Run the elastix command for the current image/patient.
  elastix -f "$input_dir/${image_file}/${image_file}_iBHCT.nii.gz" -m "$input_dir/${image_file}/${image_file}_eBHCT.nii.gz" -out "$elastix_output_subdir" -p "$param_affine" -p "$param_elastic" -fMask "$mask_dir/${image_file}/seg_lung_ours_${image_file}_iBHCT.nii.gz" -mMoving "$mask_dir/${image_file}/seg_lung_ours_${image_file}_eBHCT.nii.gz"  

  # for map_file in "${transformix_input_dir_list[@]}"; do
  #   map_name=$(basename "$map_file" | sed 's/\..*//; s/probabilisticMap/probabilisticMap/')

    # transformix_label_output_subdir="$transformix_label_output_dir/$image_name/$map_name/"

    # echo $transformix_label_output_subdir
    # # Create the 'transformixoutput' directory within the image's output directory
    # if [ ! -d "$transformix_label_output_subdir" ]; then
    #   mkdir -p "$transformix_label_output_subdir"
    # fi

    #!/bin/bash

    # # Specify the path to your elastix parameter file
    # parameter_file="$elastix_output_subdir/TransformParameters.1.txt"

    # # Backup the original parameter file
    # cp "$parameter_file" "${parameter_file}.bak"

    # # Use awk to remove duplicate lines.
    # awk '!/ResultImagePixelType/' "$parameter_file" > tmpfile && mv tmpfile "$parameter_file"
    # awk '!/ResultImageFormat/' "$parameter_file" > tmpfile && mv tmpfile "$parameter_file"

    # # Define the new parameters
    # new_parameters="
    # (ResultImageFormat \"nii\")
    # (ResultImagePixelType \"float\")
    # "
    # # Append the new parameters to the parameter file
    # echo "$new_parameters" >> "$parameter_file"

    # # Display a message indicating the modification
    # echo "Modified $parameter_file with new parameters."

    # Run the transformix command for the current image with the specified comment
    # transformix -in "$map_file" -out "$transformix_label_output_subdir" -tp "$elastix_output_subdir/TransformParameters.1.txt"
  done
done

# Record the finish time
finish_time=$(date +%s)

# Calculate the total time in minutes
total_time=$(( (finish_time - start_time) / 60 ))

# Display the start and finish times
# echo "Script started at: $(date -d @$start_time '+%Y-%m-%d %H:%M:%S')"
# echo "Script finished at: $(date -d @$finish_time '+%Y-%m-%d %H:%M:%S')"
echo "Total time taken: $total_time minutes"
