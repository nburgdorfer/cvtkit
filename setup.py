from setuptools import setup, find_packages

with open("README.md", "r") as doc:
    long_desc = doc.read()

setup(
        name="cvtkit",
        version="0.0.3",
        description="A Python library including general functions and operations on various computer vision related structures.",
        long_description=long_desc,
        long_description_content_type="text/markdown",
        url="https://github.com/nburgdorfer/vision_toolkit",
        author="Nathaniel Burgdorfer",
        author_email="nburgdorfer@gmail.com",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: C++",
            "Topic :: Multimedia",
            "Topic :: Multimedia :: Graphics",
            "Topic :: Multimedia :: Graphics :: 3D Modeling",
            "Topic :: Multimedia :: Graphics :: 3D Rendering",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Image Processing",
            "Topic :: Scientific/Engineering :: Visualization",
            "Topic :: Utilities",
        ],
        install_repuires=[],
        extras_require={
            "dev": ["pytest>=7.0"],
            },
        python_requires=">=3.9"
)
