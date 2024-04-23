import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().split("\n")

setuptools.setup(
    name="bodo_platform_extensions",
    version="2.0.0",
    author="Bodo, Inc.",
    author_email="noreply@bodo.ai",
    description="IPython magic extensions for the Bodo Cloud Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bodo-inc/bodo-platform-extensions",
    packages=setuptools.find_packages(),
    # include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
