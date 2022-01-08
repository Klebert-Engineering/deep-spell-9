import setuptools

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

with open("VERSION", "r") as v:
    version = v.read().strip()

setuptools.setup(
    name="deep-spell-9",
    version=version,
    author="Klebert Engineering",
    author_email="ds9@klebert-engineering.de",
    description="Neural geographic query processor.",
    url="https://github.com/klebert-engineering/deep-spell-9",
    packages=setuptools.find_packages("modules"),
    package_dir={'': 'modules'},
    install_requires=[req for req in requirements],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
