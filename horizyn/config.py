"""
Configuration system for Horizyn.

This module provides a simple configuration system for loading and validating
YAML config files. It supports dot-notation access and command-line overrides.
"""

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class DotDict(dict):
    """A dictionary that supports dot notation access.

    Example:
        >>> config = DotDict({'model': {'layers': 3}})
        >>> config.model.layers  # Returns 3
    """

    def __init__(self, *args, **kwargs):
        """Initialize without mutating caller-owned mappings."""
        data: dict[str, Any] = {}
        for arg in args:
            if not isinstance(arg, dict):
                raise TypeError(f"DotDict expected a mapping, got {type(arg).__name__}")
            data.update(copy.deepcopy(arg))
        data.update(copy.deepcopy(kwargs))
        super().__init__()
        for key, value in data.items():
            super().__setitem__(key, self._convert(value))

    @classmethod
    def _convert(cls, value: Any) -> Any:
        if isinstance(value, DotDict):
            return DotDict(value)
        if isinstance(value, dict):
            return DotDict(value)
        if isinstance(value, list):
            return [cls._convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._convert(item) for item in value)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, self._convert(value))

    def update(self, *args, **kwargs) -> None:
        incoming = dict(*args, **kwargs)
        for key, value in incoming.items():
            self[key] = value

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    def __ior__(self, other):
        self.update(other)
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute using dot notation."""
        if isinstance(name, str):
            self[name] = value
        else:
            raise TypeError(f"attribute name must be string, not '{type(name).__name__}'")

    def __getattr__(self, name: str) -> Any:
        """Get attribute using dot notation."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from None

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default fallback."""
        try:
            return self[key]
        except KeyError:
            return default


def load_config(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> DotDict:
    """
    Load a YAML configuration file and apply overrides.

    Args:
        config_path: Path to YAML config file.
        overrides: Dictionary of config overrides (supports dot notation keys).
        validate: Whether to validate the config structure.

    Returns:
        Loaded and validated configuration as a DotDict.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config validation fails.

    Example:
        >>> config = load_config('configs/sota.yaml')
        >>> config = load_config('configs/sota.yaml', {'training.max_epochs': 50})
    """
    # Load YAML file
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Make sure the path is correct relative to the working directory."
        )

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        raise ValueError(f"Config file is empty: {config_path}")

    # Convert to DotDict
    config = DotDict(config_dict)
    if "data" in config and "reaction_representation" not in config.data:
        config.data.reaction_representation = "fingerprint"

    # Apply overrides
    if overrides:
        config = apply_overrides(config, overrides)

    # Validate config structure
    if validate:
        validate_config(config)

    return config


def apply_overrides(config: DotDict, overrides: Dict[str, Any]) -> DotDict:
    """
    Apply command-line overrides to config.

    Supports dot notation for nested keys:
        {'training.max_epochs': 50} -> config.training.max_epochs = 50

    Args:
        config: Base configuration.
        overrides: Dictionary of overrides with dot notation keys.

    Returns:
        The same configuration object, updated in place.  This preserves the
        public API used by command-line and integration callers.
    """
    result = config
    for key, value in overrides.items():
        keys = key.split(".")
        current = result

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = DotDict()
            elif not isinstance(current[k], (dict, DotDict)):
                value_type = type(current[k]).__name__
                raise ValueError(f"Cannot override '{key}': '{k}' is not a dict (got {value_type})")
            current = current[k]

        # Set the final value
        current[keys[-1]] = value

    return result


def apply_overrides_copy(config: DotDict, overrides: Dict[str, Any]) -> DotDict:
    """Return an independently copied configuration with overrides applied."""

    return apply_overrides(DotDict(config), overrides)


def _has_any_config_value(config: DotDict, names: tuple[str, ...]) -> bool:
    return any(config.get(name, None) for name in names)


def _has_global_or_split_values(
    config: DotDict,
    *,
    global_names: tuple[str, ...],
    train_names: tuple[str, ...],
    validation_names: tuple[str, ...],
) -> bool:
    return _has_any_config_value(config, global_names) or (
        _has_any_config_value(config, train_names)
        and _has_any_config_value(config, validation_names)
    )


def validate_config(config: DotDict) -> None:
    """Validate configuration through the shared, behavior-compatible validator."""
    # Import lazily: the validator uses DotDict and the helpers defined here.
    from horizyn.config_validation import validate_config as validate

    validate(config)


def parse_overrides(args: list[str]) -> Dict[str, Any]:
    """
    Parse command-line overrides in the format --key=value or --key value.

    Args:
        args: List of command-line arguments.

    Returns:
        Dictionary of parsed overrides.

    Example:
        >>> parse_overrides(['--training.max_epochs=50', '--training.learning_rate', '1e-3'])
        {'training.max_epochs': 50, 'training.learning_rate': 0.001}
    """
    overrides = {}
    i = 0

    while i < len(args):
        arg = args[i]

        if arg.startswith("--"):
            # Remove leading dashes
            arg = arg[2:]

            # Check for = format
            if "=" in arg:
                key, value = arg.split("=", 1)
                overrides[key] = _parse_value(value)
                i += 1
            else:
                # Check for space-separated format
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    key = arg
                    value = args[i + 1]
                    overrides[key] = _parse_value(value)
                    i += 2
                else:
                    # Boolean flag (no value provided)
                    overrides[arg] = True
                    i += 1
        else:
            i += 1

    return overrides


def _parse_value(value: str) -> Any:
    """
    Parse a string value to the appropriate Python type.

    Tries to parse as int, float, bool, or keeps as string.

    Args:
        value: String value to parse.

    Returns:
        Parsed value with appropriate type.
    """
    # Numeric 0/1 must stay integers for count-valued CLI overrides. Use
    # explicit words (or a bare flag) for boolean configuration fields.
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False

    # Try int
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Keep as string
    return value
