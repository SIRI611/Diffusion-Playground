from setuptools import setup, find_packages

setup(
    name="diffusion",                         
    version="0.1.0",                            
    author="Dalen Shi",                         
    description="A framwork for diffuison models",
    long_description=open("README.md").read(),  # Long description (README)
    long_description_content_type="text/markdown",
    packages=find_packages(),                 
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",                  
    install_requires=[
        "numpy",                                
        "torch",
    ],
)
