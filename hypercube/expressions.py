#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 SKA South Africa
#
# This file is part of hypercube.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

# Based on http://stackoverflow.com/a/9558001

import ast
import itertools
import operator as op

# supported operators
operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv : op.floordiv,
    ast.Pow: op.pow,
    ast.BitXor: op.xor,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    # Comparisons
    ast.Eq: op.eq,
    ast.NotEq: op.ne,
    ast.Lt: op.lt,
    ast.LtE: op.le,
    ast.Gt: op.gt,
    ast.GtE: op.ge
}

class HCNodeVisitor(ast.NodeVisitor):
    def __init__(self, variables=None):
        self.vars = {} if variables is None else variables

    def visit_Module(self, node):
        res = [self.visit(n) for n in node.body]
        return res if len(res) > 1 else res[0]

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Str(self, node):
        return self.visit(node.value)

    def visit_Num(self, node):
        return node.n

    def visit_Compare(self, node):
        ops = node.ops
        operands = [node.left] + node.comparators
        # This produces a pairwise set of iteratorss
        a, b  = itertools.tee(operands)
        next(b, None)
        operand_pairs = zip(a,b)

        result = True

        for operation, (lhs, rhs) in zip(ops, operand_pairs):
            result = result and \
                operators[type(operation)](
                    self.visit(lhs),
                    self.visit(rhs))

        return result

    def visit_BinOp(self, node):
        return operators[type(node.op)](
            self.visit(node.left),
            self.visit(node.right))

    def visit_UnaryOp(self, node):
        return operators[type(node.op)](
            self.visit(node.operand))

    def visit_Name(self, node):
        value = self.vars.get(node.id, None)

        if value is None:
            raise ValueError(
                ("Cannot find a matching variable for "
                 "parse tree name '{n}'").format(node.id))

        # We got a string (another expression) from the dictionary. Parse.
        # TODO: extract above variable from dict in loop to optimise edge
        # case where variable value is just another variable. Infinite loops?
        if isinstance(value, str):
            return HCNodeVisitor(self.vars).visit(ast.parse(value))
        else:
            return value

    def generic_visit(self, node):
        raise SyntaxError('Unhandled node of type %s' % type(node))

def parse_expression(expr, variables=None):
    """
    parse_expression('nvis', variables={
                'ntime' : 100,
                'na' : 16,
                'nchan': 64,
                'nbl': 'na*(na-1)//2',
                'nvis': 'ntime*nbl*nchan',
            })
    """

    return HCNodeVisitor(variables).visit(ast.parse(expr))
