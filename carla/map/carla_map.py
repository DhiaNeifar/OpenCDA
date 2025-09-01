import argparse


def check_map(value: str) -> str:
    """
    Argparse-compatible validator.
    Ensures version is between MIN_VERSION and MAX_VERSION (inclusive).
    """
    towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']

    if value not in towns:
        raise argparse.ArgumentTypeError(
            f"CARLA map must be one of {towns}, but got {value}.")
    return value
