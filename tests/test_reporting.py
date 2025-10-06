# tests/test_reporting.py
import os
from unittest.mock import MagicMock
from perception_pipeline.reporting import analyze_and_save_timing_log

def test_analyze_and_save_timing_log_creates_files(tmp_path):
    """
    Tests that the reporting function creates all expected output files.
    """
    # ARRANGE:
    # 1. Create a fake "perception_log" object using MagicMock.
    #    We only need it to have a `scan_dir` attribute pointing to our temp directory.
    mock_perception_log = MagicMock()
    mock_perception_log.scan_dir = tmp_path

    # 2. Create a sample timing_log dictionary.
    sample_timing_log = {
        "Total Time": 123.45,
        "3a. SAM per image": [
            {"image_id": "img_000", "time_ms": 150.0},
            {"image_id": "img_001", "time_ms": 160.0}
        ],
        "3e. Masks per image": [
            {"image_id": "img_000", "mask_count": 10},
            {"image_id": "img_001", "mask_count": 12}
        ]
    }
    
    # ACT: Run the function with our mock objects and sample data.
    analyze_and_save_timing_log(mock_perception_log, sample_timing_log, process_limit=2)

    # ASSERT: Check that the output files were created in the temp directory.
    timing_dir = tmp_path / "timing_analysis"
    assert os.path.exists(timing_dir / "timing_log.json")
    assert os.path.exists(timing_dir / "3a._SAM_per_image_performance.png")
    assert os.path.exists(timing_dir / "3e._Masks_per_image_performance.png")
    assert os.path.exists(tmp_path / "performance_summary.txt")