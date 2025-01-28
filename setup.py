from setuptools import setup, find_packages

setup(
    name="diffusion",                          # Name of your package
    version="0.1.0",                            # Version
    author="Dalen Shi",                         # Author name
    description="A framwork for diffuison models",
    long_description=open("README.md").read(),  # Long description (README)
    long_description_content_type="text/markdown",
    # url="https://github.com/username/my_project",  # Project URL
    packages=find_packages(),                   # Automatically find packages
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",                    # Minimum Python version
    install_requires=[
        "numpy",                                # List dependencies here
        "torch",
    ],
)
