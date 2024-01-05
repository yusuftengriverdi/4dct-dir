import re

file_path = "path/to/your/points_file.txt"  # Update with the actual file path
output_index_fixed_points = []

with open(file_path, 'r') as file:
    points_list = file.readlines()

for point in points_list:
    match = re.search(r'OutputIndexFixed = \[ ([-0-9. ]+) \]', point)
    
    if match:
        coordinates = [float(coord) for coord in match.group(1).split()]
        output_index_fixed_points.append(coordinates)

# Print the extracted OutputIndexFixed points
for index, coordinates in enumerate(output_index_fixed_points, start=1):
    print(f"Point {index} - OutputIndexFixed: {coordinates}")
