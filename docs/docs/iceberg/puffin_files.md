Using Puffin Files and Theta Sketches with Bodo {#iceberg_puffin}
=================


### Theta Sketches {#iceberg-theta-sketch}

A Theta Sketch is a data structure used to approximate the number of distinct values (NDV) in data columns, which is critical for SQL planner optimizations. 
BodoSQL creates Theta Sketches as it writes data to an Iceberg table (either creating a table or inserting into one). 
This way, future queries that are run on those tables will have access to the NDV values so the planner can make better decisions about how to optimize queries.

If BodoSQL is used to insert into an existing Iceberg table, BodoSQL will attempt to union any existing Theta Sketches with the Theta Sketches created on the newly inserted data, thus obtaining NDV estimates for the entire table (both old and new data combined).

Currently, BodoSQL uses the following rules to determine when it should create Theta Sketches for a column:

- If BodoSQL is creating a table with a `CREATE TABLE AS SELECT` (CTAS) query, then currently it creates Theta Sketches for all columns of the following BodoSQL types: `Int32`, `Int64`, `Date`, `Time`, `Timestamp`, `Timestamp_LTZ`, `String`, `Binary` and `Decimal`.
- If BodoSQL is creating a table with an `INSERT INTO` query, then currently it creates and unions Theta Sketches for all columns that already have a Theta Sketch and are of one of the data types above, as well as `FLOAT` and `DOUBLE`.

!!! note
    If the environment variable `BODO_ENABLE_THETA_SKETCHES` is set to `0`, then Theta Sketches are disabled always, no matter what the column types are.

Bodo uses the Apache DataSketches library to implement Theta Sketches.

To learn more about Theta Sketches, [see the documentation](https://datasketches.apache.org/docs/Theta/ThetaSketchFramework.html).

### Puffin Files {#iceberg-puffin-files}

The way that Theta Sketches are stored when being written is via Puffin files. 
A Puffin file is an Iceberg statistics file located in the metadata folder of an Iceberg table. 
As of this writing, the only statistics it supports is a Theta Sketch. 
Each Puffin file can contain one or more Theta Sketches for various columns in the table. 
The Theta Sketches are serialized and stored in sections of the Puffin file referred to as "blobs". Each blob is associated with a specific snapshot-id and sequence number. 

When the BodoSQL planner is attempting to infer metadata about tables for the purposes of optimization, it will try to find any Puffin files that exist for the current snapshot, and will use the NDV values from the Theta Sketches whose snapshot id and sequence number indicate that they are fresh. 
If other engines have inserted rows into the table without writing a new Puffin file since the last Puffin file was created, or rows have been dropped from the table, then the sketches and their NDV estimates are no longer reliable so BodoSQL cannot use them.

To learn more about Puffin files, [see the documentation](https://iceberg.apache.org/puffin-spec/).
