"""FlashQLA-Blackwell — fork of Qwen team's FlashQLA with GB10 (SM_120/121) fixes.

Original library: https://github.com/QwenLM/FlashQLA (MIT, Qwen team / Alibaba).
Blackwell port: see docs/BLACKWELL_FIXES.md.
"""

import os
import subprocess
from setuptools import setup, find_packages

this_dir = os.path.dirname(os.path.abspath(__file__))

rev = os.getenv("QLA_VERSION_SUFFIX", "")
if not rev:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        rev = "+" + subprocess.check_output(cmd, cwd=this_dir).decode("ascii").rstrip()
    except Exception:
        rev = ""

_readme_path = os.path.join(this_dir, "README.md")
try:
    with open(_readme_path) as _f:
        _long_description = _f.read()
except FileNotFoundError:
    _long_description = __doc__ or ""

setup(
    name="flash_qla",
    version="0.1.0+blackwell" + rev,
    description="FlashQLA: Fused TileLang kernels for Linear Attention (Blackwell GB10 port)",
    long_description=_long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.8",
        "tilelang==0.1.8",
        "apache-tvm-ffi==0.1.9",
    ],
    zip_safe=False,
)
