import os
import logging
from pathlib import Path
from importlib.util import find_spec
import pickle as pk
LOGS_FOLDER = Path(os.path.dirname(os.path.dirname(find_spec('tiamat_agent').origin)), "logs")

class SessionLogger:
    def __init__(self, session_id: str, node_name: str):
        self.session_id = session_id
        self.node_name = node_name
        # path join the logs folder and the session id
        self.log_dir = LOGS_FOLDER.joinpath(session_id)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create node-specific log file
        self.log_file = self.log_dir.joinpath(f"{node_name}.log")
        
        # Setup logger
        self.logger = logging.getLogger(f"{session_id}_{node_name}")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler (optional)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def get_logger(self):
        return self.logger
    
    def get_child_logger(self, child_name: str):
        """
        Return a child logger under this session's hierarchy.
        Example: session_id.node_name.child_name
        """
        child_logger = logging.getLogger(f"{self.logger.name}.{child_name}")
        child_logger.setLevel(logging.INFO)
        # Child loggers inherit handlers from their parent automatically.
        return child_logger

    def save_obj(self, name, obj):
        filepath = self.log_dir.joinpath(f"{name}.pkl")
        with open(filepath, 'wb') as f:
            pk.dump(obj, f)
