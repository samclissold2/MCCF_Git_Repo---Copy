from pathlib import Path
from setuptools import setup, find_packages

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def read_requirements() -> list:
    """Parse requirements.txt, ignoring comments and blank lines."""
    req_path = Path(__file__).parent / "requirements.txt"
    if not req_path.exists():
        # Fallback to nested requirements list (legacy location)
        req_path = Path(__file__).parent / "MCCF" / "requirements.txt"
    if req_path.exists():
        with req_path.open(encoding="utf-8") as fp:
            return [line.strip() for line in fp if line.strip() and not line.startswith("#")]
    return []


# -----------------------------------------------------------------------------
# Package meta-data
# -----------------------------------------------------------------------------
NAME = "mccf"
DESCRIPTION = "Vietnam power infrastructure mapping & analysis toolkit"
URL = "https://example.com/mccf"  # replace with real repo URL if hosted
EMAIL = "maintainer@example.com"
AUTHOR = "MCCF Team"
LICENSE = "MIT"
PYTHON_REQUIRES = ">=3.9"
VERSION = "0.1.0"


# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=(Path(__file__).with_name("README.md").read_text(encoding="utf-8")
                      if Path(__file__).with_name("README.md").exists() else DESCRIPTION),
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=PYTHON_REQUIRES,
    url=URL,
    packages=find_packages(exclude=("tests", "docs", "examples")),
    include_package_data=True,  # relies on MANIFEST.in for non-py files
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            # command        = module:function
            "mccf-map = MCCF.PDP8.map:main",
        ]
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ],
    license=LICENSE,
) 