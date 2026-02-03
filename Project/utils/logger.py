"""
Centralized Logging Utility for Invoice Processing System
Provides structured logging with different levels and formatters
"""

import logging
import sys
from datetime import datetime
from typing import Optional
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        
        return super().format(record)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup centralized logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name
    
    Args:
        name: Logger name (typically module or class name)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class StructuredLogger:
    """
    Structured logger for agent operations with consistent formatting
    """
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.name = name
    
    def log_agent_start(self, agent_name: str, process_id: str, **kwargs):
        """Log agent execution start"""
        self.logger.info(
            f"[START] Agent {agent_name} started for process {process_id}",
            extra={"agent": agent_name, "process_id": process_id, **kwargs}
        )
    
    def log_agent_complete(self, agent_name: str, process_id: str, 
                          duration_ms: int, **kwargs):
        """Log agent execution completion"""
        self.logger.info(
            f"[COMPLETE] Agent {agent_name} completed for process {process_id} "
            f"in {duration_ms}ms",
            extra={
                "agent": agent_name, 
                "process_id": process_id, 
                "duration_ms": duration_ms,
                **kwargs
            }
        )
    
    def log_agent_error(self, agent_name: str, process_id: str, 
                       error: Exception, **kwargs):
        """Log agent execution error"""
        self.logger.error(
            f"[ERROR] Agent {agent_name} failed for process {process_id}: {str(error)}",
            extra={
                "agent": agent_name,
                "process_id": process_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
                **kwargs
            },
            exc_info=True
        )
    
    def log_decision(self, agent_name: str, process_id: str, 
                    decision: str, reasoning: str, confidence: float = None):
        """Log agent decision"""
        message = f"[DECISION] Agent {agent_name} decided: {decision} - {reasoning}"
        if confidence:
            message += f" (confidence: {confidence:.2f})"
        
        self.logger.info(
            message,
            extra={
                "agent": agent_name,
                "process_id": process_id,
                "decision": decision,
                "reasoning": reasoning,
                "confidence": confidence
            }
        )
    
    def log_escalation(self, agent_name: str, process_id: str, 
                      reason: str, **kwargs):
        """Log escalation event"""
        self.logger.warning(
            f"[ESCALATION] Agent {agent_name} escalating process {process_id}: {reason}",
            extra={
                "agent": agent_name,
                "process_id": process_id,
                "escalation_reason": reason,
                **kwargs
            }
        )
    
    def log_workflow_start(self, workflow_type: str, process_id: str, **kwargs):
        """Log workflow start"""
        self.logger.info(
            f"[WORKFLOW] Starting {workflow_type} workflow for process {process_id}",
            extra={
                "workflow_type": workflow_type,
                "process_id": process_id,
                **kwargs
            }
        )
    
    def log_workflow_complete(self, workflow_type: str, process_id: str, 
                             duration_ms: int, **kwargs):
        """Log workflow completion"""
        self.logger.info(
            f"[WORKFLOW_COMPLETE] Completed {workflow_type} workflow for process {process_id} "
            f"in {duration_ms}ms",
            extra={
                "workflow_type": workflow_type,
                "process_id": process_id,
                "duration_ms": duration_ms,
                **kwargs
            }
        )
    
    def log_metric(self, metric_name: str, value: float, **kwargs):
        """Log metric value"""
        self.logger.info(
            f"[METRIC] {metric_name}: {value}",
            extra={
                "metric_name": metric_name,
                "metric_value": value,
                **kwargs
            }
        )


# Initialize default logging
setup_logging()

# Create default structured logger
default_logger = StructuredLogger("invoice_system")