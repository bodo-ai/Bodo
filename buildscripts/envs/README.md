To rebuild / update the lock file from the sources,
1) First, install [pipx](https://pypa.github.io/pipx/) (best on regular Python or your base Conda environment)
2) Run the following
```
pipx run conda-lock -f main.yml -f build.yml -f test.yml
```

To install the environment locally, run
```
pipx run conda-lock install --mamba -n {environment name} conda-lock.yml 
```

Note that you can also install `conda-lock` via `condax` or directly via `pip`, `conda` or `mamba`. However, it is recommended to use either `pipx` or `condax` since they will automatically create an isolated environment for `conda-lock`, so its dependencies will not conflict with any other dependencies. See the [Installation](https://github.com/conda-incubator/conda-lock#installation) section. In addition, `pipx` and `condax` can automatically update `conda-lock` before every run.

If you do use `conda` or `mamba`, you can install it onto your main environment via:
```
mamba install conda-lock -c conda-forge -n base
```
