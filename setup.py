from setuptools import find_packages, setup

setup(
    name="imgsim",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    version="0.1.0",
    description="Search for the original of edited image",
    author="fropych",
    license="MIT",
)
