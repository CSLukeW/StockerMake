import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="StockerMake",
    version="0.0.31",
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
    entry_points={
        'console_scripts': [
            'StockerMake = StockerMake.StockerMake:main'
        ]
    },
    install_requires=[
        'absl-py==0.9.0',
        'aiohttp==3.6.2',
        'alpha-vantage==2.2.0',
        'astunparse==1.6.3',
        'async-timeout==3.0.1',
        'attrs==19.3.0',
        'cachetools==4.1.0',
        'certifi==2020.6.20',
        'chardet==3.0.4',
        'gast==0.3.3',
        'google-auth==1.18.0',
        'google-auth-oauthlib==0.4.1',
        'google-pasta==0.2.0',
        'grpcio==1.29.0',
        'h5py==2.10.0',
        'idna==2.9',
        'idna-ssl==1.1.0',
        'importlib-metadata==1.6.1',
        'joblib==0.15.1',
        'Keras-Preprocessing==1.1.2',
        'Markdown==3.2.2',
        'matplotlib==3.2.2',
        'multidict==4.7.6',
        'numpy==1.19.0',
        'oauthlib==3.1.0',
        'opt-einsum==3.2.1',
        'pandas==1.0.5',
        'protobuf==3.12.2',
        'pyasn1==0.4.8',
        'pyasn1-modules==0.2.8',
        'python-dateutil==2.8.1',
        'pytz==2020.1',
        'requests==2.24.0',
        'requests-oauthlib==1.3.0',
        'rsa==4.6',
        'scikit-learn==0.23.1',
        'scipy==1.4.1',
        'six==1.15.0',
        'sklearn==0.0',
        'tensorboard==2.2.2',
        'tensorboard-plugin-wit==1.6.0.post3',
        'tensorflow==2.5.1',
        'tensorflow-estimator==2.2.0',
        'termcolor==1.1.0',
        'threadpoolctl==2.1.0',
        'typing-extensions==3.7.4.2',
        'urllib3==1.26.5',
        'Werkzeug==1.0.1',
        'wrapt==1.12.1',
        'yarl==1.4.2',
        'zipp==3.1.0',

    ]
)