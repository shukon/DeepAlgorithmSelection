import setuptools

import dlas


with open('requirements.txt') as fh:
    requirements = fh.read()
requirements = requirements.split('\n')
requirements = [requirement.strip() for requirement in requirements]

with open("dlas/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")


setuptools.setup(
    name="dlas",
    version=version,
    author="shukon",
    author_email="joshua.marben@neptun.uni-freiburg.de",
    description=("DLAS"),
    license="3-clause BSD",
    keywords="machine learning algorithm configuration hyperparameter "
             "deep learning",
    url="",
    packages=setuptools.find_packages(exclude=['test', 'source']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: BSD License",
    ],
    platforms=['Linux'],
    install_requires=requirements,
    tests_require=['mock',
                   'nose'],
    test_suite='nose.collector'
)
