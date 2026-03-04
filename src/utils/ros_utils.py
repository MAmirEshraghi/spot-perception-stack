

import yaml
import time
from functools import lru_cache
import io
import functools
import cProfile
import pstats
import time
import math


# Bbox helper functions are now imported from bbox_utils
class DummyLogger:
    def info(self, message):
        pass
    def warning(self, message):
        pass
    def error(self, message):
        pass
    def debug(self, message):
        pass
    def critical(self, message):
        pass
    def fatal(self, message):
        pass
    def trace(self, message):
        pass
    def exception(self, message):
        pass


@lru_cache(maxsize=1000)
def camera_name_to_topic(camera_name, sensor_topics_path):
    """Read sensor_topics.yaml and return mapping of camera names to ROS topic names.
    
    Args:
        sensor_topics_path: Path to the sensor_topics.yaml file
        
    Returns:
        dict: Mapping of camera names to ROS topic names
    """
    with open(sensor_topics_path, 'r') as f:
        config = yaml.safe_load(f)
    
    mapping = {}
    sensors = config.get('sensors', {})
    
    # Extract RGB camera mappings from all sensors
    for sensor_name, sensor_data in sensors.items():
        rgb_section = sensor_data.get('rgb', {})
        mapping.update(rgb_section)
    
    return mapping[camera_name]


def stamp_to_str(msg):
    """Convert ROS message timestamp to string"""
    if hasattr(msg, "header") and msg.header.stamp.sec != 0:
        return f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"
    else:
        return f"{int(time.time())}.000000000"


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """
    Extract yaw angle (rotation around z-axis) from quaternion.
    
    Args:
        qx, qy, qz, qw: Quaternion components
    
    Returns:
        yaw: Angle in radians around z-axis (-π to π)
    """
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)