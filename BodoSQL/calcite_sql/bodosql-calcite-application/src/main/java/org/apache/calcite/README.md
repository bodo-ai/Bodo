# Modifying Calcite

This package allows modifying Apache Calcite.

This functionality should be utilized sparingly as there are many caveats and dangers to modifying a dependency in the
way done within this package. Utilize best judgment and restraint when making modifications. Always ask yourself if this
could be done in a different way such as through a custom SQL call, the converlet table, inheritance, or other
capabilities.

For circumstances where this is not possible, this README details the methods that should be utilized.

## Create a BodoX version of the dependency in the original package

This method is primarily used when a class would normally be extended with inheritance but Calcite does not provide
the proper public extension methods to perform that extension.

In Java, there are 4 levels of API protection: private, protected, public, and package-private.

Public and protected are the well-known public extension points. These are reliable ways to extend Calcite
that will not change between versions without a proper deprecation notice.

Private APIs are areas you should not touch. They are labeled that way to allow the developer to modify them as required.
The only way to modify or change these methods is through overwriting the dependency class files as detailed below.

Package-private APIs are pseudo-private APIs. Their intention is to be private and they should not be touched similar
to private APIs, but they can be accessed freely without build system intervention.

The way to access these APIs is to create a new class file inside the same package. Your class should be named
something like `BodoX` with `X` being the name of the class being substituted. This will allow you to access
package-private APIs.

If you write an original file, the copyright header is not required. In the more likely situation that you
are modifying an upstream file, the copyright header **must** be retained and have a message appended to it
that the file was modified by Bodo.

## Overwriting dependency class files

In very rare circumstances, mostly involving existing code, it is required to overwrite class files
directly from the dependency. In these circumstances, follow the following instructions.

**It is highly advised you do not go this route for new files.**

It is incredibly error-prone and can make upgrades difficult or cause them to break in unpredictable ways.
At the same time, if monkey-patching is required, it is really the only way to do that. In general, the
method described above works better and is less error-prone even if it is also subject to its own problems.

* You **MUST** copy the original file exactly as-is into the same file location.
* You **MUST** retain the copyright header.
* You **MUST** append a message right below the copyright header detailing that the file has been modified by Bodo.

After that is done, compile the code. Look in the `target/classes` directory for the `.class` files that were generated
during the build. You will find files that look like this:

    org/apache/calcite/sql/validate/SqlValidator.class
    org/apache/calcite/sql/validate/SqlValidator$1.class

Java will compile top-level classes to a file with the same name and the `.class` extension. It will also
compile any inner classes, both anonymous and named, to their own class files. Above, you see the file associated
with one of the anonymous inner classes in `SqlValidator`. Lambda functions are syntactic sugar for anonymous
inner classes so they will also produce their own class files.

You **MUST** include every class filepath in the list of excluded files from the `maven-shade-plugin`.
This plugin will exclude these class files from the `calcite-core` artifact and use the ones we have
copied here.

While the advice above was to look in our own `target/classes` folder, the exclude list is to exclude class files
from the dependency so the class files from upstream are the important list. Be aware the local list of class files
may not be accurate and you may need to look at each individual file that was overwritten on an upgrade to ensure
things are correct.
