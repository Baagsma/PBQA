from setuptools import setup, find_packages
from pathlib import Path

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="PBQA",
    version="1.2.3",
    description="Pattern Based Question and Answer (PBQA) is a Python library that provides tools for querying LLMs and managing text embeddings. It combines guided generation with multi-shot prompting to improve response quality and consistency.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Bart Haagsma",
    author_email="dev.baagsma@gmail.com",
    url="https://github.com/Baagsma/PBQA",
    packages=find_packages(),
    install_requires=[
        "PyYAML",
        "requests",
        "accelerate",
        "transformers",
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
