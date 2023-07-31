from setuptools import find_packages, setup

setup(
    name="imgfinder",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    version="0.0.1",
    description="Search for the original of edited image",
    author="fropych",
    license="MIT",
)
