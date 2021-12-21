# Bodo Documentation

This documentation covers the basics of using Bodo and provides a reference of supported Python features and APIs. In a nutshell, Bodo provides a JIT compilation workflow using the @bodo.jit decorator. It replaces the decorated Python functions with an optimized and parallelized binary version automatically. Below are how-tos on how developers can add and edit the documentation

## Adding Top Level Entries

To add a new major category to the documentation (Getting Started, Installation and Setup, Help and Reference, etc.) you must both create the necessary rst file as well as update the root `index.rst` file

1. Start by creating a folder in `docs/source` that has your category name separated by underscores. In this folder, include an `index.rst` file containing the text you want to put into this entry.
2. In the root of the `/docs` there should be a file called `index.rst`. At the bottom of this file, you can reference this folder and add it to the navigation bar. It should be formatted as follows...

```
.. toctree::
   :maxdepth: 4

   source/[folder_name]/index
```
3. If your category has multiple entries with their own sub-chapters, change the `maxdepth` for each layer of sub-chapters
4. Run `make clean` and then `make html` to see your changes

## Adding Entries

To add a new entry to an existing section, you must simply create an `.rst` file and add the necessary information in order it to be read correctly

1. Create an `.rst` file to the folder if the desired category.
2. In the file, add the file name (in this case "file_name") at the top like so...
```
.. _file_name:
```

3. If you have sub-headings in this entry that you want to show up in the navigator, label your headers (in this case sub_header_name) like so...
```
.. _sub_header_name:
```

## Adding Code

If you want to add an example block of code into an entry, format it like this...
```
    .. code-block:: python

        import inspect
        import bodo

        @bodo.jit
        def calc_pi(n):
            t1 = time.time()
            x = 2 * np.random.ranf(n) - 1
            y = 2 * np.random.ranf(n) - 1
            pi = 4 * np.sum(x ** 2 + y ** 2 < 1) / n
            print("Execution time:", time.time() - t1, "\nresult:", pi)
            return pi
```