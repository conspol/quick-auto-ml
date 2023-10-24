from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="quick-auto-ml",
    version="0.1",
    author="Konstantin Polev",
    author_email="conspol7@gmail.com",
    description="Performs AutoML for binary classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/conspol/quick-auto-ml",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # TODO: requirements
    ],
    python_requires=">=3.6",
)