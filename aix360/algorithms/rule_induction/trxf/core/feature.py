import numbers
import tokenize
from io import BytesIO
from typing import Dict, Any

OPERATORS = ['+', '-', '*', '/']
PARENTHESES = ['(', ')']


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class Feature:
    def __init__(self, expression: str):
        """
        @param expression: Infix arithmetic expression encoded as utf-8 string involving constants, operators (+, -,
        *, /), parentheses, and variable names. Variable names shall not have any of the above operators or
        whitespace as part of the name. Care must be taken not to use the utf-8 long dash '\xe2' for the minus sign,
        when copy-pasting.
        """
        self._expression = expression
        self._rpn = _shunting_yard(self._expression)

    @property
    def variable_names(self):
        """
        A list of variable names in the feature representation
        """
        tokens = _get_tokens(self._expression)
        return [x for x in tokens if x not in OPERATORS + ['(', ')'] and not is_number(x)]

    @property
    def reverse_polish_notation(self):
        """
        The Reverse Polish Notation (https://en.wikipedia.org/wiki/Reverse_Polish_notation) representation of the
        feature as a list of token strings. This serves as the symbolic representation of the feature.
        """
        return self._rpn

    def evaluate(self, assignment: Dict[str, Any]) -> Any:
        """
        Evaluate the value of the feature for the provided variable assignment

        @param assignment: dict mapping variable name to value
        @return: the feature value w.r.t. the assignment
        """
        operand_stack = []
        for token in self.reverse_polish_notation:
            if token in OPERATORS:
                rhs = operand_stack.pop()
                _validate_number(rhs)
                lhs = operand_stack.pop()
                _validate_number(lhs)
                expression = str(lhs) + token + str(rhs)
                try:
                    operand_stack.append(eval(expression))
                except NameError as e:
                    raise ValueError(
                        'eval() failed for expression "{}"'.format(expression)) from e
            else:
                operand = assignment[token] if token in assignment else float(token)
                operand_stack.append(operand)
        assert len(operand_stack) == 1

        result = operand_stack.pop()
        try:
            result = float(result)
        except ValueError:
            pass
        return result

    def __repr__(self):
        return '%s(%r)' % (self.__class__, self.reverse_polish_notation)

    def __str__(self):
        return ' '.join(self._expression.split())

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()


def _shunting_yard(expression):
    """
    Converts infix expression string into Reverse Polish Notation.
    https://en.wikipedia.org/wiki/Shunting-yard_algorithm
    """
    tokens = _get_tokens(expression)
    if len(tokens) > 0 and (tokens[0] == '-' or tokens[0] == '+'):
        tokens.insert(0, 0)
    op_precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    output = []
    op_stack = []
    for token in tokens:
        if token in OPERATORS:
            while len(op_stack) > 0 and op_stack[-1] != '(' and op_precedence[op_stack[-1]] >= op_precedence[token]:
                output.append(op_stack.pop())
            op_stack.append(token)
        elif token == '(':
            op_stack.append(token)
        elif token == ')':
            while len(op_stack) > 0 and op_stack[-1] != '(':
                output.append(op_stack.pop())
            assert op_stack[-1] == '('
            op_stack.pop()
        else:
            output.append(token)
    while len(op_stack) > 0:
        assert op_stack[-1] != '('
        output.append(op_stack.pop())

    _validate_rpn(output, expression)
    return output


def _get_tokens(expression):
    no_whitespace = ''.join(expression.split())
    g = tokenize.tokenize(BytesIO(no_whitespace.encode('UTF-8')).readline)
    assert next(g).string == 'utf-8'
    tokens = []
    buffer = ''
    result = []
    for _, v, _, _, _ in g:
        tokens.append(v)
    for token in tokens:
        if token not in OPERATORS + PARENTHESES:
            buffer += token
        else:
            result.append(buffer)
            result.append(token)
            buffer = ''
    result.append(buffer)
    return [x for x in result if x != '']


def _validate_rpn(rpn, expression):
    stack_size = 0
    for token in rpn:
        valence = 2 if token in OPERATORS else 0
        stack_size += 1 - valence
        if stack_size <= 0:
            raise ValueError('Invalid RPN {} for expression "{}"'.format(rpn, expression))
    if stack_size != 1:
        raise ValueError('Invalid RPN {} for expression "{}"'.format(rpn, expression))


def _validate_number(value):
    if not isinstance(value, numbers.Number) or isinstance(value, bool):
        raise ValueError('{} is not a number.'.format(value))
