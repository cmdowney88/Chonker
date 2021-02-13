import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name='chonker',
    version='1.2.2',
    author="C.M. Downey",
    author_email="cmdowney@uw.edu",
    description="A utility package for NLP and Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cmdowney88/chonker",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=[
        "numpy",
        "pyyaml",
        "torch"
    ]
)
