from IPython.core.magic import (
    Magics,
    magics_class,
    cell_magic,
    needs_local_scope,
)
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from bodo_platform_extensions.sql.transform import gen_bodosql_code


@magics_class
class SQLMagic(Magics):
    def __init__(self, shell):
        Magics.__init__(self, shell=shell)
        self.shell.configurables.append(self)

    @needs_local_scope
    @cell_magic
    @magic_arguments()
    @argument("--catalog_name", type=str)
    @argument("--output", type=str)
    @argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Print information such as the generated BodoSQL code being run.",
    )
    def sql(self, line="", cell="", local_ns={}):
        # Parse variables (words wrapped in {})
        cell = self.shell.var_expand(cell)
        args = parse_argstring(self.sql, line)

        if not args.catalog_name:
            raise ValueError("No catalog name specified.")
        if not args.output:
            raise ValueError("No output location specified.")

        catalog_name = args.catalog_name
        sql_command_text = cell
        result_var = args.output

        # Generate the boilerplate code to run
        bodosql_code = gen_bodosql_code(
            sql_command_text,
            result_var,
            catalog_name=catalog_name,
        )

        if args.verbose:
            print(f"[SQL] Catalog: {catalog_name}")
            print(f"[SQL] Result Variable: {result_var}")
            print(f"[SQL] Generated Code:\n{bodosql_code}")

        # We don't need to capture the output, etc. and can just run
        # the code as is. The output will be added to the namespace
        # as part of this execution.
        eval_result = self.shell.run_cell(bodosql_code)

        if args.verbose:
            if eval_result.error_in_exec:
                # If there's an error, print it (might be useful for debugging purposes)
                print(f"[SQL] eval_result.error_in_exec: {eval_result.error_in_exec}")
            # Print the result as is.
            print(f"[SQL] eval_result.result: {eval_result.result}")

        # Returning eval_result.error_in_exec can break stdout handling (for yet to be determined reason),
        # so the safest thing is to not return anything (same as None)
