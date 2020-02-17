.. _dev_ir_extensions:

IR Extensions
-------------

Bodo uses IR extensions for operations that are too complex for
sentinel functions to represent. For example, Join and Aggregate nodes
represent `merge` and `groupby/aggregate` operations of Pandas respectively.
IR extensions have full transformation and analysis support (usually
more extensive that sentinel functions).