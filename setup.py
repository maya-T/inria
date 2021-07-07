import setuptools

requirements = ["requests", "numpy", "nltk", "flair", "ip2geotools", "ipinfo", "datetime", "stanza"]
setuptools.setup(
    name="location_time_extractor",
    version=1.0,
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.6',
)
