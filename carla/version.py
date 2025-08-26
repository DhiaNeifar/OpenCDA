import argparse

# Allowed CARLA version range
MIN_VERSION = (0, 9, 11)
MAX_VERSION = (0, 9, 15)

def parse_version(value: str) -> tuple[int, ...]:
    """Convert version string (e.g., '0.9.14') into a tuple of integers."""
    try:
        return tuple(map(int, value.split(".")))
    except ValueError:
        raise ValueError(f"Invalid version format: {value}")

def check_version(value: str) -> str:
    """
    Argparse-compatible validator.
    Ensures version is between MIN_VERSION and MAX_VERSION (inclusive).
    """
    nums = parse_version(value)
    if nums < MIN_VERSION or nums > MAX_VERSION:
        raise argparse.ArgumentTypeError(
            f"CARLA version must be between "
            f"{'.'.join(map(str, MIN_VERSION))} and {'.'.join(map(str, MAX_VERSION))}, "
            f"but got {value}"
        )
    return value

def verify_runtime(version_str: str):
    """
    Verify at runtime (outside argparse).
    Example: verify_runtime(carla.__version__)
    """
    nums = parse_version(version_str)
    if nums < MIN_VERSION or nums > MAX_VERSION:
        raise RuntimeError(
            f"CARLA version must be between "
            f"{'.'.join(map(str, MIN_VERSION))} and {'.'.join(map(str, MAX_VERSION))}, "
            f"but got {version_str}"
        )
    return True
