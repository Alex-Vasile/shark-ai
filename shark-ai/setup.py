# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
from pathlib import Path

from setuptools import setup

THIS_DIR = Path(__file__).parent

# Setup and get version information.
# The `version_local.json` is generated by calling:
# `build_tools/python_deploy/compute_common_version.py -stable --write-json`
VERSION_FILE_LOCAL = os.path.join(THIS_DIR, "version_local.json")


def load_version_info(version_file):
    with open(version_file, "rt") as f:
        return json.load(f)


version_info = load_version_info(VERSION_FILE_LOCAL)

PACKAGE_VERSION = version_info.get("package-version")
print(f"Using PACKAGE_VERSION: '{PACKAGE_VERSION}'")

setup(
    version=f"{PACKAGE_VERSION}",
)
