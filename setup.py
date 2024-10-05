from setuptools import setup, find_packages

with open("README.md", "r") as doc:
    long_desc = doc.read()

setup(
        name="cvtkit",
        description="A Python library including general functions and operations on various computer vision related structures.",
        long_description=long_desc,
        long_description_content_type="text/markdown",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        install_repuires=[],
        extras_require={
            "dev": ["pytest>=7.0"],
            },
        python_requires=">=3.8"
)
