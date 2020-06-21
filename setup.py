import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="StockerMake",
    version="0.0.10",
    author="Luke Williams",
    author_email="williams.luke.2001@gmail.com",
    description="Modular Neural Network Protyping for Stock Market Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CSLukeW/StockerMake",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires='>=3.6',
    scripts=['./scripts/StockerMake']
)