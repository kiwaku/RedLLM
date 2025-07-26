"""
Enhanced logging configuration for RedLLM to filter out noisy LiteLLM debug messages
and provide better error visibility.
"""

import logging
import sys
from typing import Dict, Any


class LiteLLMFilter(logging.Filter):
    """Custom filter to reduce LiteLLM debug noise while keeping important messages."""
    
    def __init__(self):
        super().__init__()
        # Patterns that indicate important LiteLLM messages we want to keep
        self.important_patterns = [
            "error", "failed", "exception", "timeout", "unauthorized", 
            "forbidden", "rate limit", "quota", "billing", "payment",
            "authentication", "api key", "connection refused",
            "server error", "service unavailable", "critical"
        ]
        
        # Patterns that indicate noise we want to filter out
        self.noise_patterns = [
            "model_info:", "Final returned optional params:", "self.optional_params:",
            "Only some together models support", "Docs - https://docs.together.ai",
            "chat.py:", "utils.py:", "litellm_logging.py:", "add_model_to_model_list"
        ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records based on content and severity."""
        message = record.getMessage().lower()
        
        # Always allow ERROR and CRITICAL levels
        if record.levelno >= logging.ERROR:
            return True
        
        # For DEBUG level LiteLLM messages, be very selective
        if record.name.startswith("litellm") and record.levelno == logging.DEBUG:
            # Keep important messages
            if any(pattern in message for pattern in self.important_patterns):
                return True
            
            # Filter out noise
            if any(pattern.lower() in message for pattern in self.noise_patterns):
                return False
            
            # Filter out very long debug dumps
            if len(record.getMessage()) > 500:
                return False
        
        return True


class ColoredConsoleFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    def __init__(self):
        super().__init__()
        
        # Color codes
        self.COLORS = {
            'DEBUG': '\033[36m',     # Cyan
            'INFO': '\033[32m',      # Green  
            'WARNING': '\033[33m',   # Yellow
            'ERROR': '\033[31m',     # Red
            'CRITICAL': '\033[35m',  # Magenta
        }
        self.RESET = '\033[0m'
        
        # Icons for different log levels
        self.ICONS = {
            'DEBUG': '🔧',
            'INFO': '✅', 
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨'
        }
    
    def format(self, record: logging.LogRecord) -> str:
        # Get color and icon for log level
        color = self.COLORS.get(record.levelname, '')
        icon = self.ICONS.get(record.levelname, '')
        
        # Format the message
        formatted = f"{color}{icon} {record.levelname}{self.RESET}: {record.getMessage()}"
        
        # Add module info for non-root loggers
        if record.name != "root" and not record.name.startswith("redllm"):
            formatted = f"{formatted} [{record.name}]"
        
        return formatted


def setup_enhanced_logging(verbose_level: int = 1) -> None:
    """
    Set up enhanced logging configuration with LiteLLM filtering.
    
    Args:
        verbose_level: 0=errors only, 1=normal, 2=debug, 3=verbose debug
    """
    
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    # Set base log level based on verbosity
    if verbose_level >= 3:
        base_level = logging.DEBUG
        litellm_level = logging.DEBUG
    elif verbose_level >= 2:
        base_level = logging.DEBUG
        litellm_level = logging.INFO
    elif verbose_level >= 1:
        base_level = logging.INFO
        litellm_level = logging.WARNING
    else:
        base_level = logging.ERROR
        litellm_level = logging.ERROR
    
    # Configure root logger
    logging.basicConfig(
        level=base_level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[]  # We'll add custom handlers
    )
    
    # File handler (always detailed, no filtering)
    file_handler = logging.FileHandler("redllm.log", mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler with filtering and colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(base_level)
    console_handler.setFormatter(ColoredConsoleFormatter())
    console_handler.addFilter(LiteLLMFilter())
    
    # Add handlers to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Configure specific loggers
    logger_configs = {
        "litellm": litellm_level,
        "httpx": logging.WARNING,
        "openai": logging.WARNING,
        "anthropic": logging.WARNING,
        "urllib3": logging.WARNING,
        "requests": logging.WARNING,
        "httpcore": logging.WARNING,
    }
    
    for logger_name, level in logger_configs.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        # Don't propagate to avoid duplicate messages
        if logger_name in ["urllib3", "requests", "httpcore"]:
            logger.propagate = False
    
    # Create a RedLLM-specific logger for our application
    redllm_logger = logging.getLogger("redllm")
    redllm_logger.setLevel(base_level)
    
    print(f"🔧 Enhanced logging configured (verbose level: {verbose_level})")
    print(f"📝 Full logs saved to: redllm.log")
    
    if verbose_level < 2:
        print("💡 Use -v 2 or higher to see debug messages")
    
    return redllm_logger


def log_system_info():
    """Log important system information at startup."""
    logger = logging.getLogger("redllm.system")
    
    try:
        import litellm
        logger.info(f"LiteLLM version: {litellm.__version__}")
    except:
        logger.warning("Could not get LiteLLM version")
    
    try:
        import dspy
        logger.info(f"DSPy available: {hasattr(dspy, '__version__')}")
    except:
        logger.warning("DSPy not available")


def create_error_context_logger(context: str):
    """Create a logger with specific context for error tracking."""
    return logging.getLogger(f"redllm.{context}")


# Convenience function for backwards compatibility
def setup_logging(verbose_level: int = 1):
    """Backwards compatible logging setup function."""
    return setup_enhanced_logging(verbose_level)
