#!/usr/bin/env python
# Copyright (C) 2020 Bodo Inc. All rights reserved.

"""The main obfuscate script. It does the job of taking a python file and returning
another one which is harder to read.

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

The reading of the data is done by the AST module. But the writing is more
problematic. There are 3 modules available: codegen, astunparse and astor.
codegen is old and out. astunparse could not be made to work and astor
happens to work as required.

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

"""

import ast
import astor
import random
import sys


list_reserved_keywords = {
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
    def __init__(self):
        ast.NodeTransformer.__init__(self)
        self.list_variables = set()

    def mapping_var(self, name):
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
        for erec in node.elts:
            self.visit(erec)
        return node

    def visit_Attribute(self, node):
        self.visit(node.value)
        return node

    def visit_Lambda(self, node):
        for earg in node.args.args:
            self.mapping_var(earg.arg)
        self.visit(node.body)
        return node

    def visit_Subscript(self, node):
        self.visit(node.slice)
        self.visit(node.value)
        return node

    def visit_Name(self, node):
        self.mapping_var(node.id)
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
        It is made sure that no collision happens and that reserved words are not used."""
        if self.processing_pass == 1:
            return name
        while True:
            newname = random_string(3, 10)
            if (
                not newname in self.replacement_vars
                and not newname in list_reserved_keywords
            ):
                self.replacement_vars.add(newname)
                return newname

    def mapping_var(self, insert_as_replaceable, name):
        if self.processing_pass == 0:
            if insert_as_replaceable and not name in list_reserved_keywords:
                if not name in self.replace_vars:
                    newname = self.obfuscate_local(name)
                    self.replace_vars[name] = newname
            return name
        if name in self.fixed_names:
            return name
        if not name in self.replace_vars:
            return name
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

    def visit_If(self, node):
        node.test = self.visit(node.test)
        node.body = [self.visit(x) for x in node.body]
        node.orelse = [self.visit(x) for x in node.orelse]
        return node

    def visit_Str(self, node):
        return node

    def visit_Num(self, node):
        return node

    def visit_ExceptHandler(self, node):
        node.name = self.mapping_var(True, node.name)
        node.body = [self.visit(x) for x in node.body]
        return node

    def generic_visitor(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Slice(self, node):
        if node.lower != None:
            self.visit(node.lower)
        if node.upper != None:
            self.visit(node.upper)
        if node.step != None:
            self.visit(node.step)
        return node

    def visit_Tuple(self, node):
        for erec in node.elts:
            self.visit(erec)
        return node

    def visit_Attribute(self, node):
        self.visit(node.value)
        return node

    # All variables inside a lambda are set to fixed:
    # ---the arguments themselves for which the same reasoning as FunctionDef
    #    applies
    # ---the other variables which may be global and are difficult to track.
    def visit_Lambda(self, node):
        retriev = VariableRetriever()
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
        # The function name itself is set to fixed because it can be potentially part of the API.
        self.insert_fixed_names(node.name)
        for earg in node.args.args:
            # The arguments are set to fixed because they are either default ones or
            # have their name of significance in say decorator functions.
            self.insert_fixed_names(earg.arg)

        node.body = [self.visit(x) for x in node.body]
        return node

    # The code is numba specific and that is bad.
    # We need a better solution for this problem.
    def visit_withitem(self, node):
        is_numba = False
        if node.context_expr != None:
            if isinstance(node.context_expr, ast.Call):
                if isinstance(node.context_expr.func, ast.Attribute):
                    if isinstance(node.context_expr.func.value, ast.Name):
                        if node.context_expr.func.value.id == "numba":
                            is_numba = True
        if is_numba:
            for xkey in node.context_expr.keywords:
                # In numba withitem case, the value itself must be fixed.
                self.insert_fixed_names(xkey.arg)
        if isinstance(node.context_expr, ast.Call):
            for xarg in node.context_expr.args:
                self.visit(xarg)

        return node

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
        for x in node.body:
            if isinstance(x, ast.Assign):
                for target in x.targets:
                    # Global variable in module statements need to be fixed.
                    self.insert_fixed_names(self.retrieve_assigned_name(target))
            self.visit(x)
        return node

    def visit_ClassDef(self, node):
        for x in node.body:
            if isinstance(x, ast.Assign):
                for target in x.targets:
                    # variables occurring in class definition need to be fixed
                    self.insert_fixed_names(self.retrieve_assigned_name(target))
            self.visit(x)
        return node

    def visit_Subscript(self, node):
        self.visit(node.slice)
        self.visit(node.value)
        return node

    def visit_Name(self, node):
        node.id = self.mapping_var(isinstance(node.ctx, ast.Store), node.id)
        return node


#
# The processing of individual files
#


def process_file(efile, stdoutput):
    root = ast.parse(open(efile, "rb").read())
    sys.stderr.write("  Initial source code has been read\n")

    obf = Obfuscator()
    root = obf.visit(root)
    sys.stderr.write("  First reading pass on the source code has been done\n")

    obf.processing_pass = 1
    root = obf.visit(root)
    sys.stderr.write("  Second renaming pass has been done\n")

    if stdoutput:
        sys.stderr.write("------------------------------------------------\n")
        sys.stderr.write("The abstract Syntax Tree : ", ast.dump(root), "\n")
        sys.stderr.write("------------------------------------------------\n")
        sys.stdout.write(astor.to_source(root))
    else:
        f = open(efile, "w")
        f.write(astor.to_source(root))




def process_input(argv):
    if len(argv) < 3:
        sys.stderr.write("Usage:\n")
        sys.stderr.write("--- For obfuscating and printing the obfuscated to standard output:\n")
        sys.stderr.write("     obfuscate.py stdout file1.py ... fileN.py\n")
        sys.stderr.write("--- For obfuscating the files themselves:\n")
        sys.stderr.write("     obfuscate.py file file1.py ... fileN.py\n")
        exit(0)

    if argv[1]=="stdout":
        stdoutput=True
    elif argv[1]=="file":
        stdoutput=False
    else:
        sys.stderr.write("Value of first argument is stdout or file\n")
        exit(0)

    nb_file=len(argv)-2
    for ifile in range(nb_file):
        efile=argv[ifile+2]
        sys.stderr.write("ifile={}/{} file={}".format(ifile, nb_file, efile))
        process_file(efile, stdoutput)

process_input(sys.argv)
