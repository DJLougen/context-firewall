"""Configuration management for Honey-Comb.

Supports configuration via:
1. Default values
2. Configuration files (YAML/JSON)
3. Environment variables (HONEYCOMB_*)
4. Runtime overrides

Usage:
    from honeycomb.config import get_config
    
    config = get_config()
    print(config.cool_loop_interval)
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class HoneyCombConfig:
    """Honey-Comb configuration."""
    
    # Hot loop settings
    hot_loop_max_retries: int = 3
    hot_loop_timeout_ms: int = 5000
    
    # Cool loop settings
    cool_loop_interval: int = 10  # Run every N turns
    cool_loop_stale_threshold: int = 5  # Mark as stale after N turns
    
    # Compression settings
    compression_min_ratio: float = 0.1  # Minimum acceptable compression
    compression_max_tokens: int = 4096  # Max tokens per message before compression
    
    # Metrics settings
    metrics_enabled: bool = True
    metrics_prometheus_port: Optional[int] = None  # None = no Prometheus server
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "text"
    
    # Thread safety
    thread_safe: bool = True
    
    # Model settings
    model_path: Optional[str] = None
    fallback_to_rules: bool = True

    # Failure tee settings (rtk-style)
    tee_enabled: bool = True
    tee_mode: str = "failures"  # "failures", "always", or "never"
    tee_dir: Optional[str] = None  # None = ~/.local/share/honeycomb/tee

    # Gain tracking settings (rtk-style analytics)
    gain_enabled: bool = True
    gain_dir: Optional[str] = None  # None = ~/.local/share/honeycomb
    
    def update_from_dict(self, data: dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        prefix = "HONEYCOMB_"
        for key in dir(self):
            if key.startswith("_") or callable(getattr(self, key)):
                continue
            
            env_key = f"{prefix}{key.upper()}"
            if env_key in os.environ:
                value = os.environ[env_key]
                current = getattr(self, key)
                
                # Type coercion
                if isinstance(current, bool):
                    setattr(self, key, value.lower() in ("true", "1", "yes"))
                elif isinstance(current, int):
                    setattr(self, key, int(value))
                elif isinstance(current, float):
                    setattr(self, key, float(value))
                elif current is None:
                    # Keep as string or None
                    setattr(self, key, value if value.lower() != "none" else None)
                else:
                    setattr(self, key, value)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            key: getattr(self, key)
            for key in dir(self)
            if not key.startswith("_") and not callable(getattr(self, key))
        }
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        if self.hot_loop_max_retries < 1:
            issues.append("hot_loop_max_retries must be >= 1")
        
        if self.hot_loop_timeout_ms < 100:
            issues.append("hot_loop_timeout_ms must be >= 100")
        
        if self.cool_loop_interval < 1:
            issues.append("cool_loop_interval must be >= 1")
        
        if self.compression_min_ratio <= 0 or self.compression_min_ratio > 1:
            issues.append("compression_min_ratio must be in (0, 1]")
        
        if self.compression_max_tokens < 100:
            issues.append("compression_max_tokens must be >= 100")
        
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            issues.append(f"Invalid log_level: {self.log_level}")
        
        if self.log_format not in ("json", "text"):
            issues.append(f"Invalid log_format: {self.log_format}")

        if self.tee_mode not in ("failures", "always", "never"):
            issues.append(f"Invalid tee_mode: {self.tee_mode}")
        
        return issues


# Global configuration instance
_config: Optional[HoneyCombConfig] = None


def load_config(config_path: Optional[str] = None) -> HoneyCombConfig:
    """Load configuration from file and environment.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
    
    Returns:
        Loaded configuration
    """
    global _config
    
    config = HoneyCombConfig()
    
    # Load from file if provided
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                if path.suffix in (".yaml", ".yml"):
                    try:
                        import yaml
                        data = yaml.safe_load(f)
                        config.update_from_dict(data or {})
                    except ImportError:
                        print(f"Warning: PyYAML not installed, skipping {config_path}")
                elif path.suffix == ".json":
                    data = json.load(f)
                    config.update_from_dict(data)
    
    # Override with environment variables
    config.update_from_env()
    
    # Validate
    issues = config.validate()
    if issues:
        print(f"Configuration warnings: {', '.join(issues)}")
    
    _config = config
    return config


def get_config() -> HoneyCombConfig:
    """Get current configuration, loading defaults if needed."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset configuration to None (for testing)."""
    global _config
    _config = None
