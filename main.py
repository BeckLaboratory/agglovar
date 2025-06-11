#!/usr/bin/env python3

import argparse


#
# Main
#

if __name__ == "__main__":

    # Parse command-line arguments
    epilog = \
"""
Agglovar is licensed under the MIT license. See LICENSE for details.
"""

    parser = argparse.ArgumentParser(
        description='Agglovar: Variant transformations and intersects.',
        epilog=epilog
    )

    args = parser.parse_args()

    print("Agglovar is under development and does not yet have a command-line interface.")
