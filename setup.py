from setuptools import setup, find_packages

setup(
    name="PBQA",
    version="0.1.0",
    description="Pattern Based Question and Answer",
    author="Bart Haagsma",
    author_email="dev.baagsma@gmail.com",
    packages=find_packages(),
    install_requires=[
        "PyYAML",
        "requests",
        "qdrant-client",
        "sentence-transformers",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
)
