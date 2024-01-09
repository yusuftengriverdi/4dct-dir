import re

algorithm = 'elastix-challenge'
for image_name in ["copd0", "copd5", "copd6"]:
    input_file_path = f"{algorithm}/{image_name}/outputpoints.txt"  # Update with the actual file path
    output_file_path = f"{algorithm}/{image_name}/{image_name}_300_iBH_xyz_r1_transformed.txt"  # Update with the desired output file path
    output_index_fixed_points = []

    with open(input_file_path, 'r') as file:
        points_list = file.readlines()

    for point in points_list:
        match = re.search(r'OutputIndexFixed = \[ ([-0-9. ]+) \]', point)
        
        if match:
            coordinates = [float(coord) for coord in match.group(1).split()]
            output_index_fixed_points.append(coordinates)

    # Write the extracted OutputIndexFixed points to a new file
    with open(output_file_path, 'w') as output_file:
        for coordinates in output_index_fixed_points:
            output_file.write(f"{' '.join(map(str, coordinates))}\n")

    print(f"Coordinates for {image_name} saved to {output_file_path}")
