from setuptools import find_packages, setup

setup(
    name="deep_sort",
    version="0.1.0",
    description="A Python package for deep sort",
    author="Eduardo Guerra",
    author_email="eduardo.vilasboas.guerra@protonmail.com",
    url="https://github.com/Eduardo-Vilas-Boas/deep_sort",
    packages=find_packages(),
    install_requires=[
        "torch",
        # Add other dependencies as needed
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)
