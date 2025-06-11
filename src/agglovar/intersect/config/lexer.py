import ply.lex

__all__ = ['IntersectConfigLexer']

class IntersectConfigLexer(object):
    """
    Token lexer for intersect configuration strings.
    """
    lexer: ply.lex.lex
    tokens: tuple[str]
    literals: tuple[str]

    def __init__(self,
        **kwdargs: dict
    ):
        """
        Initialize lexer.

        :param kwdargs: Passed to `ply.lex.lex`
        """
        self.lexer = ply.lex.lex(module=self, **kwdargs)

    tokens = (
        'T_FLOAT_EXP',
        'T_INT_MULT',
        'T_FLOAT',
        'T_INT',
        'T_UNLIMITED',
        'KW_MATCH',
        'KEYWORD'
    )

    literals = [':', ';', ',', '=', '(', ')']

    # Parse numbers
    def parse_float(self,
        str_val: str
    ) -> float:
        """
        Parse float from string.

        :param str_val: String.
        :return: Float value.
        """
        str_val = str_val.lower()

        if 'e' in str_val:
            str_val, exp = str_val.split('e', 1)
        else:
            exp = 0

        return float(str_val) * 10 **float(exp)

    def parse_int(self,
        str_val: str
    ) -> int:
        """
        Parse int from string.

        :param str_val: String.
        :return: Integer value.
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
        t.value = self.parse_float(t.value)
        return t

    def t_T_INT_MULT(self, t):
        r'[+-]?((\d+\.?\d*)|(\.\d+))[kKmMgG]'
        t.value = self.parse_int(t.value)
        return t

    def t_T_FLOAT(self, t):
        r'[+-]?(\d+\.\d*)|(\d*\.\d+)'
        t.value = self.parse_float(t.value)
        return t

    def t_T_INT(self, t):
        r'[+-]?(\d+)'
        t.value = self.parse_int(t.value)
        return t

    def t_T_UNLIMITED(self, t):
        r'[uU][nN][lL][iI][mM][iI][tT][eE][dD]'
        return t

    # Keywords. Defined as functions to guarantee precdence in ply.
    def t_KW_MATCH(self, t):
        r'[mM][aA][tT][cC][hH]'
        return t

    def t_KEYWORD(self, t):
        r'\w(\w|\d)+'
        return t

    # Handle errors
    def t_error(self, t):
        raise RuntimeError(
            'Illegal character in input: "{}" at position {} ({})"'.format(
                t.value[0],
                t.lexpos,
                '"{}{}"'.format(t.value[:20], '...' if len(t.value) > 20 else '') if len(t.value) > 5 else 'end of config string'
            )
        )

        t.lexer.skip(1)
