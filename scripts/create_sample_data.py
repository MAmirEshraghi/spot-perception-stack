import pickle
from obs_data_buffer import ObsDataBuffer

FULL_DATA_PATH = "data/obs_buffer.pkl"       
SAMPLE_OUTPUT_PATH = "tests/data/sample_obs_buffer.pkl" 

USE_SKIPPING_PATTERN = True

PROCESS_GROUP_SIZE = 5       
SKIP_GROUP_SIZE = 25           
# total length of one cycle (e.g., 5 + 30 = 35)
CYCLE_LENGTH = PROCESS_GROUP_SIZE + SKIP_GROUP_SIZE
#  maximum number of frames in final sample file
MAX_SAMPLE_FRAMES = 50
# for Simple Sampling (if USE_SKIPPING_PATTERN is False)
NUM_FIRST_FRAMES_TO_KEEP = 4

print(f"Loading full data buffer from: {FULL_DATA_PATH}")
with open(FULL_DATA_PATH, "rb") as f:
    full_buffer: ObsDataBuffer = pickle.load(f)

# a new empty buffer for our sample
sample_buffer = ObsDataBuffer()
# copy essential static transforms
sample_buffer.static_transforms = full_buffer.static_transforms
sample_buffer._check_tf_static_complete()

print("Copying entries to new buffer...")

if USE_SKIPPING_PATTERN:
    print(f"Using skipping pattern: Process {PROCESS_GROUP_SIZE}, Skip {SKIP_GROUP_SIZE}")
    entries_copied = 0
    cycle_counter = 0 # tracks  
    
    for timestamp, entry in full_buffer.entries.items():
        # only consider entries that have a full set of observations
        if entry.is_frame_full():

            if entries_copied >= MAX_SAMPLE_FRAMES:
                break
                
            # check if we in 'process' part of cycle
            if cycle_counter < PROCESS_GROUP_SIZE:
                sample_buffer.entries[timestamp] = entry
                entries_copied += 1
                        
            cycle_counter += 1
            
            # if complete a full cycle, reset the counter
            if cycle_counter >= CYCLE_LENGTH:
                cycle_counter = 0

else: 
    print(f"Using simple method: Taking the first {NUM_FIRST_FRAMES_TO_KEEP} frames.")
    entries_copied = 0
    for timestamp, entry in full_buffer.entries.items():
        if entry.is_frame_full():
            sample_buffer.entries[timestamp] = entry
            entries_copied += 1
            if entries_copied >= NUM_FIRST_FRAMES_TO_KEEP:
                break

print(f"\nCopied {len(sample_buffer.entries)} entries to the new buffer.")

# save the new smaller buffer
with open(SAMPLE_OUTPUT_PATH, "wb") as f:
    pickle.dump(sample_buffer, f)

print(f"✅ Successfully created sample data file at: {SAMPLE_OUTPUT_PATH}")

