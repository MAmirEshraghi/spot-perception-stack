# tests/test_e2e_pipeline.py
import pytest
import os
import json
import subprocess
import sys

@pytest.mark.e2e
def test_full_pipeline_run(tmp_path):
    """
    Runs the entire pipeline as a subprocess on a small sample of real data.
    This test is slow and requires the SAM model checkpoint.
    """
    # 1) ARRANGE part:
    # check for dependencies
    sam_checkpoint_path = os.environ.get("SAM_CHECKPOINT_PATH")
    if not sam_checkpoint_path or not os.path.exists(sam_checkpoint_path):
        pytest.skip("SAM_CHECKPOINT_PATH env var not set or file not found. Skipping E2E test.")

    sample_data_path = "tests/data/sample_obs_buffer.pkl"
    if not os.path.exists(sample_data_path):
        pytest.skip(f"Sample data not found at {sample_data_path}. Run scripts/create_sample_data.py.")

    # make command-line args
    command = [
        sys.executable,  # use same python interpreter 
        "-m", "perception_pipeline.run_pipeline",
        "-p", sample_data_path,
        "-s", sam_checkpoint_path,
        "-l", "0",  
        "--min_mask_area", "500"
    ]

    # 2) ACT part: Run the pipeline as a separate process
    result = subprocess.run(command, capture_output=True, text=True, cwd=".")
    
    # print the script's output for debugging if the test fails
    print("--- STDOUT ---")
    print(result.stdout)
    print("--- STDERR ---")
    print(result.stderr)
    
    # assert that the script ran successfully
    assert result.returncode == 0, "The pipeline script failed to execute."

    # ASSERT: Check that the output artifacts were created correctly.
    output_dirs = [d for d in os.listdir("logs/perception_logs_offline") if os.path.isdir(os.path.join("logs/perception_logs_offline", d))]
    assert len(output_dirs) > 0, "No output directory was created in the logs folder."
    latest_log_dir = sorted(output_dirs)[-1] # test run is the most recent one
    log_path = os.path.join("logs/perception_logs_offline", latest_log_dir)

    print(f"Verifying outputs in: {log_path}")
    
    assert os.path.exists(os.path.join(log_path, "log.json"))
    assert os.path.exists(os.path.join(log_path, "performance_summary.txt"))
    assert os.path.exists(os.path.join(log_path, "frame_visualizations"))

    # check that at least one visualization image was created
    viz_files = os.listdir(os.path.join(log_path, "frame_visualizations"))
    assert len(viz_files) > 0, "No visualization images were created."

    # sanity-check the content of the log file
    with open(os.path.join(log_path, "log.json")) as f:
        log_data = json.load(f)
        assert len(log_data["images"]) > 0
        assert len(log_data["unique_objects"]) > 0, "Pipeline ran but found no unique objects."