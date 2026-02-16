
import argparse

def parse_resolution(resolution_str):
    if not resolution_str.strip():
        return []
    try:
        return [int(x.strip()) for x in resolution_str.split(',')]
    except:
        raise argparse.ArgumentTypeError("Resolution must be comma-separated integers (e.g., '12' or '12,24')")