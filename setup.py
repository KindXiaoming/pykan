import setuptools

# Load the long_description from README.md
with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pykan",
    version="0.0.5",
    author="Ziming Liu",
    author_email="zmliu@mit.edu",
    description="Kolmogorov Arnold Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/kindxiaoming/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
