import setuptools

import sphinx_rtd_dark_mode

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=sphinx_rtd_dark_mode.__name__,
    version=sphinx_rtd_dark_mode.__version__,
    author=sphinx_rtd_dark_mode.__author__,
    description=sphinx_rtd_dark_mode.__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MrDogeBro/sphinx_rtd_dark_mode",
    download_url=f"https://github.com/MrDogeBro/sphinx_rtd_dark_mode/archive/v{sphinx_rtd_dark_mode.__version__}.tar.gz",
    packages=setuptools.find_packages(exclude=["tests"]),
    package_data={
        "sphinx_rtd_dark_mode": [
            "static/dark_mode_css/general.css",
            "static/dark_mode_css/dark.css",
            "static/dark_mode_js/theme_switcher.js",
            "static/dark_mode_js/default_light.js",
            "static/dark_mode_js/default_dark.js",
        ]
    },
    license=sphinx_rtd_dark_mode.__license__,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.4",
    install_requires=["sphinx-rtd-theme"],
)
