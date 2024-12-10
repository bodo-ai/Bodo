# Managing Packages on the cluster using Jupyter magics - Conda and Pip {#managing-packages}

We recommend all packages to be installed using Conda as that is what we use in our environments.
Any conda command can be run in parallel on all the nodes of your cluster using `%pconda`.
To install a new package on all the nodes of your cluster you can use `%pconda install`.
All conda install arguments work as expected, e.g., `-c conda-forge` to set the channel.

```shell
%pconda install -c conda-forge <PACKAGE_NAME>
```

To learn more about the packages installed on the cluster nodes `%pconda list`.

```shell
%pconda list
```

To remove a conda package on all your cluster nodes, use `%pconda remove`.

```shell
%pconda remove <PACKAGE_NAME>
```

![Conda-Magic](../../platform2-gifs/conda-magic.gif#center)

Any pip command can be run in parallel on all the nodes of your cluster using `%ppip`.

Example:

```shell
%ppip install <PACKAGE_NAME>
```

To learn about the installed packages, you can use `%ppip show`
to get the details of the package.

```shell
%ppip show <PACKAGE_NAME>
```

To remove the same package on all the nodes of your cluster, use `%ppip uninstall`.

```shell
%ppip uninstall <PACKAGE_NAME> -y
```

![Pip-Magic](../../platform2-gifs/pip-magic.gif#center)
