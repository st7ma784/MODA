import os
import sys

FIRMWARE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'firmware'))
if FIRMWARE not in sys.path:
    sys.path.insert(0, FIRMWARE)
