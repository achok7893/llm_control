# -*- coding: utf-8 -*-
import os
from datetime import datetime


def add_timestamp_to_filename(filename):
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]

    # Split the filename and its extension
    base, extension = os.path.splitext(filename)

    # Append timestamp to the base filename
    new_filename = f"{base}_{timestamp}{extension}"

    return new_filename
