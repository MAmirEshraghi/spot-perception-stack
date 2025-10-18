import pickle
from src_perception.obs_data_buffer import ObsDataBuffer

FULL_DATA_PATH = "data/obs_buffer.pkl"
SAMPLE_OUTPUT_PATH = "tests/data/sample_obs_buffer.pkl"

# Configuration
USE_SKIPPING_PATTERN = True
# For Skipping Pattern (if USE_SKIPPING_PATTERN is True)
PROCESS_GROUP_SIZE = 5      # number of frames to KEEP in a group
SKIP_GROUP_SIZE = 10      # number of frames to SKIP in a group
CYCLE_LENGTH = PROCESS_GROUP_SIZE + SKIP_GROUP_SIZE
MAX_SAMPLE_FRAMES = 30      # maximum number of frames in final sample file
# For Simple Sampling (if USE_SKIPPING_PATTERN is False)
NUM_FIRST_FRAMES_TO_KEEP = 4


print(f"Loading full data buffer from: {FULL_DATA_PATH}")
full_buffer: ObsDataBuffer = pickle.load(open(FULL_DATA_PATH, "rb"))

# a new empty buffer for our sample
sample_buffer = ObsDataBuffer()
# copy essential static transforms
sample_buffer.static_transforms = full_buffer.static_transforms
sample_buffer._check_tf_static_complete()

print("Calculating frame indices to copy...")

# Get all data as a list to be indexed
all_items = list(full_buffer.entries.items())
num_total_frames = len(all_items)

to_use_fidxs = []
if USE_SKIPPING_PATTERN:
    print(f"Using skipping pattern: Process {PROCESS_GROUP_SIZE}, Skip {SKIP_GROUP_SIZE}")
    
    #  build the index list using the constants
    # This outer loop steps by one full cycle length
    for i in range(0, num_total_frames, CYCLE_LENGTH):
        # This inner loop takes the 'process' group from the start of the cycle
        for j in range(i, i + PROCESS_GROUP_SIZE):
            # Stop if we go past the end of the list
            if j < num_total_frames:
                to_use_fidxs.append(j)
            else:
                break
else:
    print(f"Using simple method: Taking the first {NUM_FIRST_FRAMES_TO_KEEP} frames.")
    # Create a list of the first N indices
    num_to_take = min(NUM_FIRST_FRAMES_TO_KEEP, num_total_frames)
    to_use_fidxs = list(range(num_to_take))

print(f"Copying {min(len(to_use_fidxs), MAX_SAMPLE_FRAMES)} entries to new buffer...")

# Loop over the pre-calculated indices, limited by MAX_SAMPLE_FRAMES
for i in to_use_fidxs[:MAX_SAMPLE_FRAMES]:
    timestamp, entry = all_items[i]
    sample_buffer.entries[timestamp] = entry
    
    # This new method does NOT check entry.is_frame_full().
    # It copies frames based on their index only.

print(f"\nCopied {len(sample_buffer.entries)} entries to the new buffer.")

# save the new smaller buffer
with open(SAMPLE_OUTPUT_PATH, "wb") as f:
    pickle.dump(sample_buffer, f)

print(f" Successfully created sample data file at: {SAMPLE_OUTPUT_PATH}")