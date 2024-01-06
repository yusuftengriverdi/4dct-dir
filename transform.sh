#!/bin/bash

# Record the start time
start_time=$(date +%s)

parameter_dir="elastix-prep-mask-nogantry"
input_dir="data/preprocessed"
# List of image filenames to process
image_list=("copd1" "copd2" "copd3" "copd4")

# Loop through the list and process each image
for image_file in "${image_list[@]}"; do

    transformix_parameter="$parameter_dir/$image_file/TransformParameters.1.txt"
    input_subdir="$input_dir/$image_file/${image_file}_300_iBH_xyz_r1.txt"
    output_subdir="$parameter_dir/$image_file"
    # Run the transformix command for the current image with the specified comment
    transformix -def "$input_subdir" -out "$output_subdir" -tp "$transformix_parameter"
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
