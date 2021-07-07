import setuptools

requirements = ["ipython>=6", "nbformat>=4", "nbconvert>=5", "requests>=2"]
setuptools.setup(
    name="location_time_extractor",
    version=1.0,
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.6',
)