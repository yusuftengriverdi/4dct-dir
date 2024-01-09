import csv
from collections import defaultdict

# Initialize a defaultdict to store scores for each metric
metric_scores = defaultdict(list)

# Read data from the text file
file_path = 'training-runs/log_testv2.csv'  # Replace with the actual file path
with open(file_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        metric = row['Metric']
        value = float(row['Value'])
        metric_scores[metric].append(value)

# Calculate average scores for each metric
average_scores = {}
for metric, values in metric_scores.items():
    average_scores[metric] = sum(values) / len(values)

# Save average scores to a new CSV file
with open(file_path, mode='a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header
    writer.writerow(['mean', 'score', 'value'])
    
    # Write average scores
    for metric, average_value in average_scores.items():
        writer.writerow(['mean', metric, average_value])

print(f"Average scores saved to {file_path}")
