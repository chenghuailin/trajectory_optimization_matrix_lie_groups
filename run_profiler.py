import cProfile
import pstats
import os
from datetime import datetime

# Create a profiler object
profiler = cProfile.Profile()

# Run the file you want to profile
profiler.run('runpy.run_path("main_SE3ddp_tracking.py")')

# Specify the output directory and ensure it exists
output_dir = 'ProfileReports'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get the current timestamp
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define the output file path with timestamp
output_file = os.path.join(output_dir, f'ProfileReport_{current_time}.txt')

# Save the profiling results to a file
with open(output_file, 'w') as f:
    stats = pstats.Stats(profiler, stream=f)
    stats.sort_stats('cumulative')  # You can sort as needed, 'cumulative' sorts by total time
    stats.print_stats()

print(f"Profile report saved to: {output_file}")