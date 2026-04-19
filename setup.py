"""
Coffee Text Analytics - Setup Configuration

Enables installation via: pip install -e .
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="coffee-text-analytics",
    version="1.0.0",
    author="Marcelo Seijas",
    author_email="marcelo.seijas@erasmusuniversity.nl",
    description="Text analytics and ML pipeline for predicting coffee review ratings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mseijse01/coffee-text-analytics",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.9",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "coffee-analytics=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
