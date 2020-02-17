.. _dev_compiler_stages:

Compiler Stages
---------------

`BodoCompiler <https://github.com/Bodo-inc/Bodo/blob/master/bodo/compiler.py#L68>`_
class defines the compiler pipeline. Below are the main stages.

- `TranslateByteCode`, ... before `BodoUntypedPass`:
  Numba's frontend passes that process function byte code, generate
  the IR, and prepare for type inference.
- `BodoUntypedPass`: transforms the IR to remove features that Numba's type
  inference cannot support such as non-uniform dictionary input of
  `pd.DataFrame({})`.
- `NopythonTypeInference`: Numba's type inference.
- `BodoDataFramePass`: converts data frame operations to Series and Array
  operations as much as possible to provide implementation and enable
  optimization. Creates specialized IR nodes for complex operations like Join.
- `BodoSeriesPass`: converts Series operations to array operations as much as
  possible to provide implementation and enable optimization.
- `ParforPass`: converts Numpy operations into parfors, fuses all parfors
  if possible, and performs basic optimizations such as copy propagation and
  dead code elimination.
- `BodoDistributedPass`: analyzes the IR to decide parallelism of arrays and
  parfors for distributed transformation, then
  parallelizes the IR for distributed execution and inserts MPI calls.
- `NoPythonBackend`: Numba's backend to generate LLVM IR and eventually binary.


For demonstration of these passes, follow the compiler pipeline (input/output IRs) for a simple function like
`Series.sum()` for initial understanding of the transformations.
See the :ref:`Numba development page <numba>`
for information about Numba, which is critical for Bodo development.
See the :ref:`Bodo install page <build_bodo>`
for information about setting up the enviroment for Bodo development.