#!/usr/bin/env python

"""The main obfuscation script. It converts a python file to a harder to read version
by renaming variables.

Use:

  ./obfuscate.py stdout file1.py ... fileN.py
  and the obfuscated code is put in the standard output.
  ./obfuscate.py file file1.py ... fileN.py
  and the obfuscated code of fileI.py is put in fileI.py

Right now the obfuscation is done with the local variables which are scrambled
to random integers. In the future, we want to further obfuscate the program.

ALGORITHM:

The algorithm uses the visitor pattern provided by the AST library.
It works in two passes:
* First pass on the whole code to determine what variables names may not
change (need to be fixed) and compute the new name of variable according
to a random procedure.
* Then a second pass where the change to the AST is implemented according to
the first pass. The fixed names are left alone and for the others we assign
them to their new name.

Two classes are built:
* One for finding all the variables in a node. This is needed for lambdas
since variables of lambdas can be assigned to global variable in so many
ways that it is better to have them fixed.
* Another for looking at all the variables of the node and doing the first
or second pass.
  * In the first pass the variables are all considered. The ones whose value
  is going to be fixed are indicated and the ones whose value is going to
  change are indicated with their new name being computed.
  * In the second pass the changes to the program are made.

The reading reading and writing of the data is done by the AST module.

A major limitation of the code is that the notion of scope is not considered.
This means that if a variable is renamed in one scope, it is renamed to the same
name in all other scopes in which it occurs. Also, if a variable cannot be changed
to another name in one scope, then in all scopes it will be fixed.


TODO:
* obfuscate strings and numbers.
* obfuscate variable name differently on each function which require a notion of scope
to be implemented.
* improve coverage to all the functions of BODO.
* improve coverage on all the possibilities of Python.
* obfuscate function arguments when it is determined that we can do it. (we can for default
functions most of the time. Emphasis on most)

REFERENCE

The description of the AST is available on:
https://greentreesnakes.readthedocs.io/en/latest/nodes.html
For each type of the AST, we need to have a corresponding entry in the function.


"""

import ast
import random
import sys

reserved_keywords = {
    "async",
    "await",
    "False",
    "class",
    "finally",
    "is",
    "return",
    "None",
    "continue",
    "for",
    "lambda",
    "try",
    "True",
    "def",
    "from",
    "nonlocal",
    "while",
    "and",
    "del",
    "global",
    "not",
    "with",
    "as",
    "elif",
    "if",
    "or",
    "yield",
    "assert",
    "else",
    "import",
    "pass",
    "break",
    "except",
    "in",
    "raise",
    "self",
    "args",
    "kwargs",
    # builtin functions/classes (could be used like variables, e.g. typ == str)
    # https://docs.python.org/3/library/functions.html
    "abs",
    "delattr",
    "hash",
    "memoryview",
    "set",
    "all",
    "dict",
    "help",
    "min",
    "setattr",
    "any",
    "dir",
    "hex",
    "next",
    "slice",
    "ascii",
    "divmod",
    "id",
    "object",
    "sorted",
    "bin",
    "enumerate",
    "input",
    "oct",
    "staticmethod",
    "bool",
    "eval",
    "int",
    "open",
    "str",
    "breakpoint",
    "exec",
    "isinstance",
    "ord",
    "sum",
    "bytearray",
    "filter",
    "issubclass",
    "pow",
    "super",
    "bytes",
    "float",
    "iter",
    "print",
    "tuple",
    "callable",
    "format",
    "len",
    "property",
    "type",
    "chr",
    "frozenset",
    "list",
    "range",
    "vars",
    "classmethod",
    "getattr",
    "locals",
    "repr",
    "zip",
    "compile",
    "globals",
    "map",
    "reversed",
    "__import__",
    "complex",
    "hasattr",
    "max",
    "round",
}


def random_string(minlength, maxlength):
    return "".join(
        chr(random.randint(ord("a"), ord("z")))
        for x in range(random.randint(minlength, maxlength))
    )


class VariableRetriever(ast.NodeTransformer):
    """This class takes a node and allows to compute all the variables
    that occur in that block.
    This is used for lambdas for which basically every variable can a priori be
    a global one and they should be fixed."""

    def __init__(self, treat_local):
        """The treat_local=False means that function and classes variables are not computed"""
        ast.NodeTransformer.__init__(self)
        self.list_variables = set()
        self.treat_local = treat_local

    def add_var(self, name):
        self.list_variables.add(name)

    # The visitor functions

    def visit_If(self, node):
        self.visit(node.test)
        for x in node.body:
            self.visit(x)
        for x in node.orelse:
            self.visit(x)
        return node

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)
        return node

    def visit_Slice(self, node):
        if node.lower != None:
            self.visit(node.lower)
        if node.upper != None:
            self.visit(node.upper)
        if node.step != None:
            self.visit(node.step)
        return node

    def visit_Tuple(self, node):
        # All entries are covered.
        for erec in node.elts:
            self.visit(erec)
        return node

    def visit_Attribute(self, node):
        # We should not visit the node.attribute since we want only the attribute name
        self.visit(node.value)
        return node

    def visit_Lambda(self, node):
        for earg in node.args.args:
            self.add_var(earg.arg)
        self.visit(node.body)
        return node

    def visit_Subscript(self, node):
        self.visit(node.slice)
        self.visit(node.value)
        return node

    def visit_FunctionDef(self, node):
        self.add_var(node.name)
        # The arguments and variable in a functions are local
        if self.treat_local:
            for earg in node.args.args:
                self.visit(earg.arg)
            for x in node.body:
                self.visit(x)
        return node

    def visit_ClassDef(self, node):
        # The arguments and variable in a functions are local
        if self.treat_local:
            for x in node.body:
                self.visit(x)
        return node

    def visit_Name(self, node):
        self.add_var(node.id)
        return node


class Obfuscator(ast.NodeTransformer):
    """The class allows to find out all of the names occurring and do obfuscation.
    The enconding is done randomly and varies from one run to the next.
    Two passes are made:
    * Computing the fixed variables and for the moving variable their assigned one.
    * In a second pass, we changed the names occurring."""

    def __init__(self):
        ast.NodeTransformer.__init__(self)
        self.processing_pass = 0
        self.replace_vars = {}
        self.fixed_names = set()
        self.replacement_vars = set()

    def insert_fixed_names(self, name):
        """In the first pass, we store the names that will remain unchanged"""
        if self.processing_pass == 0:
            self.fixed_names.add(name)

    def obfuscate_local(self, name):
        """The obfuscation that uses random function.
        It is made sure that no collision happens and that reserved words are not used.
        """
        while True:
            newname = random_string(3, 5) + "__" + random_string(3, 5)
            if (
                newname not in self.replacement_vars
                and newname not in reserved_keywords
                and newname not in self.fixed_names
            ):
                self.replacement_vars.add(newname)
                return newname

    def mapping_var(self, insert_as_replaceable, name):
        if name == None:
            return None

        if self.processing_pass == 0:
            if insert_as_replaceable and name not in reserved_keywords:
                if name not in self.replace_vars:
                    newname = self.obfuscate_local(name)
                    self.replace_vars[name] = newname
            return name
        if name in self.fixed_names:
            return name
        if name not in self.replace_vars:
            return name
        # Check to make sure the name isn't reserved (i.e. an import name or func)
        # This must be done after the first pass to make sure there are no
        # conflicts anywhere.
        while self.replace_vars[name] in self.fixed_names:
            self.replace_vars[name] = self.obfuscate_local(name)
        return self.replace_vars.get(name)

    def retrieve_assigned_name(self, node):
        """For an entry which is a subscript an attribute or just a name, we want to return
        the name that needs to be renamed"""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Subscript):
            return self.retrieve_assigned_name(node.value)
        if isinstance(node, ast.Attribute):
            return self.retrieve_assigned_name(node.value)
        if isinstance(node, ast.arg):
            return self.retrieve_assigned_name(node.arg)
        print("type(node)=", type(node))
        print("dir(node)=", dir(node))
        print("Missing unsupported case")
        exit(1)

    # The visitor functions

    #
    # visitor functions that could be further obfucasted in the future.
    #

    def visit_Str(self, node):
        # String entry. Could be obfuscated in the future.
        return node

    def visit_Num(self, node):
        # Numerical entry. Could be obfuscated in the future.
        return node

    #
    # Obfuscation related to the names.
    #

    def visit_Call(self, node):
        # Call case. We have to handle separately the instance that we want to be careful about.
        # The attribute starargs and kwargs have been removed since version 3.5
        if isinstance(node.func, ast.Name):
            if node.func.id == "isinstance":
                if len(node.args) != 2:
                    print("The number of entries in args of isinstance must be 2")
                    exit(0)
                if isinstance(node.args[1], ast.Name):
                    self.insert_fixed_names(node.args[1].id)
        self.visit(node.func)
        for earg in node.args:
            self.visit(earg)
        for erec in node.keywords:
            self.visit(erec.value)
        return node

    def visit_ExceptHandler(self, node):
        # node.type is not a variable, it is the type of exception, IE: ValueError
        # Therefore, we should not obfuscate it...
        # unless possibly it's of an exception class that we've created, which has been obfuscated?
        # can this occur?

        newName = self.mapping_var(True, node.name)
        if (node.name != None and newName == None) or (
            node.name == None and newName != None
        ):
            raise Exception(
                "Error in obfuscation: self.mapping_var should not map None to a variable name, or visa versa"
            )

        node.name = newName
        node.body = [self.visit(x) for x in node.body]
        return node

    def visit_Attribute(self, node):
        # Here we intentionally do not visit the "attr" array, because we do not want to rename it.
        self.visit(node.value)
        return node

    # All variables inside a lambda are set to fixed:
    # ---the arguments themselves for which the same reasoning as FunctionDef
    #    applies
    # ---the other variables which may be global and are difficult to track.
    def visit_Lambda(self, node):
        retriev = VariableRetriever(True)
        retriev.visit(node)
        for e_var in retriev.list_variables:
            # All the names occurring in a lambda are set to fixed because they can
            # all be global variables or having their name being significant.
            self.insert_fixed_names(e_var)
        self.visit(node.body)
        return node

    # For several reasons we have to fix function arguments:
    # ---For overloaded function the argument names is important
    # ---For default arguments we of course need fixed names
    # ---For function created via exec(...) the name of the argument is actually important.
    # Thus for simplicity all function arguments are fixed.
    def visit_FunctionDef(self, node):
        node = self._remove_docstring(node)
        # The function name itself is set to fixed because it can be potentially part of the API.
        self.insert_fixed_names(node.name)
        for earg in node.args.args:
            # The arguments are set to fixed because they are either default ones or
            # have their name of significance in say decorator functions.
            # TODO: Remove this when there is no decorator.
            self.insert_fixed_names(earg.arg)

        node.args.args = [self.visit(x) for x in node.args.args]
        node.body = [self.visit(x) for x in node.body]
        return node

    # This is for treating the arguments to the function.
    def visit_arg(self, node):
        node.arg = self.mapping_var(True, node.arg)
        return node

    def visit_withitem(self, node):
        # The code is needed due to "with objmode()" requiring consistent variable
        # names in the with statement and in the body. Currently, we avoid doing any
        # obfuscation for any calls in the with clause
        # which is overly conservative. We probably need a better solution for
        # this problem in the long term
        if node.context_expr != None:
            if isinstance(node.context_expr, ast.Call):
                for xkey in node.context_expr.keywords:
                    # In withitem case, the value itself must be fixed.
                    self.insert_fixed_names(xkey.arg)

        node = self.generic_visit(node)
        return node

    # The global entries are indicated as fixed.
    def visit_ImportFrom(self, node):
        for x in node.names:
            if isinstance(x, ast.alias):
                # import statement need to have fixed names
                self.insert_fixed_names(x.name)
                if x.asname != None:
                    # import statements need have fixed names
                    self.insert_fixed_names(x.asname)
            self.visit(x)
        return node

    def visit_Module(self, node):
        node = self.generic_visit(node)
        retriev = VariableRetriever(False)
        retriev.visit(node)
        for x in retriev.list_variables:
            self.insert_fixed_names(x)
        for x in node.body:
            self.visit(x)
        return node

    # The class entries are fixed by default since they may be accessed from outside
    def visit_ClassDef(self, node):
        node = self._remove_docstring(node)
        # Need to consider supporting: name, bases, keywords, decorator_list
        for x in node.body:
            if isinstance(x, ast.Assign):
                for target in x.targets:
                    # variables occurring in class definition need to be fixed
                    self.insert_fixed_names(self.retrieve_assigned_name(target))
            elif isinstance(x, ast.AnnAssign):
                self.insert_fixed_names(self.retrieve_assigned_name(x.target))
            self.visit(x)
        return node

    # The key function here. This is the one that changes the names.
    def visit_Name(self, node):
        # Only the id attribute matters here.
        node.id = self.mapping_var(isinstance(node.ctx, ast.Store), node.id)
        return node

    # The classes for which the entries are treated uniformly

    def visit_Expr(self, node):
        # Only one attribute in this case.
        self.visit(node.value)
        return node

    def visit_MatchAs(self, node):
        if node.pattern != None:
            node.pattern = self.visit(node.pattern)
        node.name = self.mapping_var(True, node.name)
        return node

    def _remove_docstring(self, node):
        if len(node.body) == 0:
            return node

        first_node = node.body[0]
        if isinstance(first_node, ast.Expr) and isinstance(first_node.value, ast.Str):
            node.body = node.body[1:]
            if len(node.body) == 0:
                # add a pass statement in case the function/class/module has
                # only a docstring
                node.body.append(ast.Pass())
        return node


#
# The processing of individual files
#


def process_file(efile, stdoutput):
    root = ast.parse(open(efile, "rb").read())
    sys.stderr.write("  Initial source code has been read\n")
    if stdoutput:
        sys.stderr.write("------------------------------------------------\n")
        sys.stderr.write("Before preprocessing.  AST = " + ast.dump(root) + "\n")
        sys.stderr.write("------------------------------------------------\n")

    obf = Obfuscator()
    root = obf.visit(root)
    sys.stderr.write("  First reading pass on the source code has been done\n")

    obf.processing_pass = 1
    root = obf.visit(root)
    sys.stderr.write("  Second renaming pass has been done\n")

    if stdoutput:
        sys.stderr.write("------------------------------------------------\n")
        sys.stderr.write(
            "After preprocessing, the Abstract Syntax Tree : " + ast.dump(root) + "\n"
        )
        sys.stderr.write("------------------------------------------------\n")
        sys.stdout.write(ast.unparse(root))
    else:
        f = open(efile, "w")
        f.write(ast.unparse(root))


def process_input(argv):
    if len(argv) < 3:
        sys.stderr.write("Usage:\n")
        sys.stderr.write(
            "--- For obfuscating and printing the obfuscated to standard output:\n"
        )
        sys.stderr.write("     obfuscate.py stdout file1.py ... fileN.py\n")
        sys.stderr.write("--- For obfuscating the files themselves:\n")
        sys.stderr.write("     obfuscate.py file file1.py ... fileN.py\n")
        exit(0)

    if argv[1] == "stdout":
        stdoutput = True
    elif argv[1] == "file":
        stdoutput = False
    else:
        sys.stderr.write("Value of first argument is stdout or file\n")
        exit(0)

    nb_file = len(argv) - 2
    for ifile in range(nb_file):
        efile = argv[ifile + 2]
        sys.stderr.write(f"ifile={ifile}/{nb_file} file={efile}")
        process_file(efile, stdoutput)


process_input(sys.argv)
