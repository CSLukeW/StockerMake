import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Stocker",
    version="0.0.1",
    author="Luke Williams",
    author_email="williams.luke.2001@gmail.com",
    description="Modular Neural Network Protyping for Stock Market Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CSLukeW/Stocker",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: LGPLv3 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)