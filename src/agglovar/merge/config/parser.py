"""
Lexer and parser for intersect configuration strings. Contains code for
parsing string and generating an AST from according to grammar rules.
Downstream objects will generate a configuration object from the AST.
"""

__all__ = [
    'IntersectConfigLexer',
    'IntersectConfigParser',
]

import ply.lex
import ply.yacc

from typing import Any


class IntersectConfigLexer(object):
    """Token lexer for intersect configuration strings."""

    lexer: ply.lex.Lexer
    tokens: tuple[str]
    literals: tuple[str]

    def __init__(
            self,
            **kwdargs: dict
    ):
        """Initialize lexer.

        Keyword arguments are passed to `ply.lex.lex`.

        :raises Exception: If the ply lexer throws an exception. May throw a range of exception types.
        """
        self.lexer = ply.lex.lex(module=self, **kwdargs)

    tokens = (
        'T_FLOAT_EXP',
        'T_INT_MULT',
        'T_FLOAT',
        'T_INT',
        'T_UNLIMITED',
        'KW_MATCH',
        'KW_TRUE',
        'KW_FALSE',
        'KEYWORD'
    )

    literals = [':', ';', ',', '=', '(', ')', '[', ']']

    # Parse numbers
    def parse_float(
            self,
            str_val: str
    ) -> float:
        """Parse float from string.

        :param str_val: String to parse.

        :returns: Float value.

        :raises ValueError: If the string does not represent a float value.
        """
        str_val = str_val.lower()

        if 'e' in str_val:
            str_val, exp = str_val.split('e', 1)
        else:
            exp = 0

        return float(str_val) * 10 ** float(exp)

    def parse_int(
            self,
            str_val: str
    ) -> int:
        """Parse int from string.

        :param str_val: String to parse.

        :returns: Integer value.

        :raises ValueError: If the string does not represent an int value.
        """
        str_val_lower = str_val.lower()

        if str_val_lower.endswith('k'):
            multiplier = int(1e3)
            str_val = str_val[:-1]

        elif str_val_lower.endswith('m'):
            multiplier = int(1e6)
            str_val = str_val[:-1]

        elif str_val_lower.endswith('g'):
            multiplier = int(1e9)
            str_val = str_val[:-1]

        else:
            multiplier = 1

        return int(float(str_val) * multiplier)

    # Number tokens. Defined as functions to guarantee precedence in ply.
    def t_T_FLOAT_EXP(self, t):
        r'[+-]?((\d+\.?\d*)|(\.\d+))[eE][+-]?((\d+\.?\d*)|(\.\d+))'
        t.value = self.parse_float(str(t.value))
        return t

    def t_T_INT_MULT(self, t):
        r'[+-]?((\d+\.?\d*)|(\.\d+))[kKmMgG]'
        t.value = self.parse_int(str(t.value))
        return t

    def t_T_FLOAT(self, t):
        r'[+-]?(\d+\.\d*)|(\d*\.\d+)'
        t.value = self.parse_float(str(t.value))
        return t

    def t_T_INT(self, t):
        r'[+-]?(\d+)'
        t.value = self.parse_int(str(t.value))
        return t

    def t_T_UNLIMITED(self, t):
        r'[uU][nN][lL][iI][mM][iI][tT][eE][dD]'
        return t

    # Keywords. Defined as functions to guarantee precdence in ply.
    def t_KW_MATCH(self, t):
        r'[mM][aA][tT][cC][hH]'
        return t

    def t_KW_TRUE(self, t):
        r'[tT][rR][uU][eE]'
        return t

    def t_KW_FALSE(self, t):
        r'[fF][aA][lL][sS][eE]'
        return t

    def t_KEYWORD(self, t):
        r'\w(\w|\d)*'
        return t

    # Handle errors
    def t_error(self, t):
        """Handle lexer errors.

        :param t: Token that caused the error.

        :raises ValueError: Always raised for illegal characters.
        """
        raise ValueError(
            'Illegal character in input: "{}" at position {} ({})"'.format(
                t.value[0],
                t.lexpos,
                '"{}{}"'.format(
                    t.value[:20], '...' if len(t.value) > 20 else ''
                ) if len(t.value) > 5 else 'end of config string'
            )
        )


class IntersectConfigParser(object):
    """Intersect configuration parser."""
    tokens: tuple[str]
    lexer: ply.lex.Lexer
    parser: ply.yacc.LRParser

    tokens = IntersectConfigLexer.tokens

    def __init__(
            self,
            **kwdargs: Any
    ) -> None:
        """Initialize parser.

        Keyword arguments are passed to `ply.yacc.yacc`.

        :raises Exception: If the ply parser throws an exception. May throw a range of exception types.
        """
        if 'write_tables' not in kwdargs.keys():
            kwdargs['write_tables'] = False

        if 'debug' not in kwdargs.keys():
            kwdargs['debug'] = False

        self.lexer = IntersectConfigLexer().lexer
        self.parser = ply.yacc.yacc(module=self, **kwdargs)

    def p_merge_config(self, p) -> None:
        """
        merge_config : KEYWORD ':' ':' spec_list
        """
        p[0] = {
            'strategy': p[1],
            'spec_list': p[4]
        }

    def p_spec_list(self, p) -> None:
        """
        spec_list : spec
                  | spec ':' spec_list
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = p[1] + p[3]

    def p_spec(self, p) -> None:
        """
        spec : KEYWORD
             | KEYWORD '(' val_list ')'
             | t_match
        """
        is_match = (
            len(p) == 2 and
            issubclass(p[1].__class__, list) and
            len(p[1]) > 0 and
            issubclass(p[1][0].__class__, tuple) and
            len(p[1][0]) > 0 and
            p[1][0][0] == 'match'
        )

        if is_match:
            p[0] = [{
                'type': 'match',
                'val_list': p[1][0][1]
            }]
        else:
            p[0] = [{
                'type': p[1],
                'val_list': p[3] if len(p) == 5 else []
            }]

    # spec_primitive and primitive types
    #
    # Tuples of:
    # 1) type
    # 2) value (None if unlimited)
    # 3) key (if key=val, else None)
    def p_val_primitive(self, p) -> None:
        """
        val_primitive : t_int
                      | t_float
                      | t_bool
                      | t_unlimited
                      | t_match
                      | t_primitive_list
        """
        p[0] = p[1]

    def p_t_int(self, p) -> None:
        """
        t_int : T_INT
              | T_INT_MULT
        """
        p[0] = [('int', p[1], None)]

    def p_t_float(self, p) -> None:
        """
        t_float : T_FLOAT
                | T_FLOAT_EXP
        """
        p[0] = [('float', p[1], None)]

    def p_t_unlimited(self, p) -> None:
        """
        t_unlimited : T_UNLIMITED
        """
        p[0] = [('unlimited', None, None)]

    def p_t_bool(self, p) -> None:
        """
        t_bool : KW_TRUE
               | KW_FALSE
        """
        if p[1].lower() == 'true':
            p[0] = [('bool', True, None)]
        elif p[1].lower() == 'false':
            p[0] = [('bool', False, None)]
        else:
            raise AssertionError('Parser bug: Expected "true" or "false", found %s' % p[1])

    def p_t_primitive_list(self, p) -> None:
        """
        t_primitive_list : '[' primitive_list_elements ']'
        """
        p[0] = [('list', p[2], None)]

    def p_primitive_list_elements(self, p) -> None:
        """
        primitive_list_elements : primitive_list_element
                                | primitive_list_element ',' primitive_list_elements
                                | ',' primitive_list_elements
        """
        if len(p) == 2:
            p[0] = [p[1]]
        elif len(p) == 4:
            p[0] = [p[1]] + p[3]
        elif len(p) == 3:
            p[0] = p[2]

    def p_primitive_list_element(self, p) -> None:
        """
        primitive_list_element : t_int
                               | t_float
        """
        p[0] = p[1][0]

    def p_val_list(self, p) -> None:
        """
        val_list : val
                 | val ',' val_list
                 | ',' val_list
        """
        if len(p) == 2:
            p[0] = p[1]
        elif len(p) == 4:
            p[0] = p[1] + p[3]
        else:
            p[0] = [None] + p[2]

    def p_val(self, p) -> None:
        """
        val : val_primitive
            | KEYWORD '=' val_primitive
            | KW_MATCH '=' val_primitive
        """
        if len(p) == 2:
            p[0] = p[1]
        elif len(p) == 4:
            p[0] = [(p[3][0][0], p[3][0][1], p[1])]
        else:
            raise AssertionError('Parser bug: Expected 2 or 4 elements for "spec" rule, found %d' % len(p))

    def p_t_match(self, p) -> None:
        """
        t_match : KW_MATCH '(' val_list ')'
                | KW_MATCH
        """
        if len(p) == 5:
            match_list = p[3]
        elif len(p) == 2:
            match_list = []
        else:
            raise AssertionError('Parser bug: Expected 2 or 5 elements for "match" rule, found %d' % len(p))

        p[0] = [('match', match_list, None)]

    def p_error(self, p) -> None:  # noqa: D102
        self.parser.errtok = p

        if p is not None:
            raise ValueError(
                'Syntax error at position {} ("{}"): {}'.format(
                    p.lexpos,
                    p.value,
                    '"' + (
                        p.lexer.lexdata[p.lexpos:(p.lexpos + 20)] + '...'
                        if len(p.lexer.lexdata) - p.lexpos > 20
                        else ''
                    ) + (
                        '"' if len(p.lexer.lexdata) - p.lexpos > 5 else 'at end of input'
                    )
                )
            )
        else:
            # p is None if an error occurs at the end
            raise ValueError(
                'Syntax error at end of format string (incomplete expression): Possibly missing a closing ")"?'
            )

    # Parse
    def parse(self, *args, **kwdargs) -> dict:
        """Parse a configuration string.

        Arguments are passed to the parser.

        :returns: AST as a dictionary.

        :raises Exception: If the ply parser or lexer throws an exception. Multiple exception types may be thrown.
        """
        return self.parser.parse(*args, **kwdargs)
