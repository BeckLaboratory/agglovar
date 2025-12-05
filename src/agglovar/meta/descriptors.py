"""Descriptors for common variable control.

Descriptors control behavior of instance and class variables, such as setting default values or checking values as they
are assigned to attributes or class variables.

Classes:
    - `AutoInitBase`: Base class for descriptors that should auto-initialize their private values. Implements
        boilerplate code for descriptors in this submodule.
    - `OneWayBool`: One-way boolean descriptor. Initialized to either True or False, and when changed, it cannot be
        changed back.
    - `BoundedInt`: An integer type with optional minimum and maximum value enforcement.
"""

__all__ = [
    'AutoInitBase',
    'CheckedBool',
    'CheckedObject',
    'CheckedString',
    'OneWayBool',
    'BoundedInt',
    'BoundedFloat',
]

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
import functools
import operator
import re

from typing import (
    Any,
    Generic,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from types import UnionType

T = TypeVar('T')

class AutoInitBase(ABC, Generic[T]):
    """Base class for descriptors that should auto-initialize their private values.

    When objects of a specific class have the same attributes (i.e. keys in __dict__ are the same), instances can share
    keys instead of each instance having its own keys. If a descriptor is used as a public interface for a private
    attribute, the private attribute is not created in each instance. This base class modifies the __init__ method of
    instances so that all private attributes are initialized to some default value before __init__ completes. This
    ensures that when private variables are first accessed, they do not create a new key in the object __dict__ and
    force the keys to be copied instead of shared across instances.
    """
    name_priv: str = ''
    default: Optional[T]
    name_pub: str = ''
    optional: Optional[bool] = True
    _is_base: bool = True

    def __init__(
            self,
            default: Optional[T] = None,
            optional: Optional[bool] = None,
            name_priv: Optional[str] = None,
    ) -> None:
        """Create a descriptor.

        :param default: Default value. May not be `None` if the field is not optional.
        :param optional: If not `None`, Override typing hints and explicitly set if this attribute is optional. This
            should only be used for descriptors where a missing value does not make sense, otherwise, use typing hints.
        :param name_priv: Private name. Defaults to public name prepended with "_". Can be a pattern including a "name"
            wildcard (e.g. "_{name}").
        """
        if self._is_base:
            raise TypeError(f'Cannot instantiate {type(self).__name__} directly')

        self.default = default
        self.name_priv = str(name_priv).strip() if name_priv is not None else ''

        self.optional = optional

    @abstractmethod
    def non_optional_default(self) -> T:
        """Called to get a default value when the parameter is not optional and the default is None."""
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Mark subclasses as non-base
        cls._is_base = False

    def __set_name__(self, owner, name) -> None:
        """
        Set name of the public and private variable names.

        :param owner: Class that owns the descriptor.
        :param name: Public name.
        """
        self.name_pub = name
        self.name_priv = self.name_priv.format(name=name) if self.name_priv != '' else '_' + name

        if self.optional is None:
            self.optional = self._detect_optional(owner, name)

        if not self.optional and self.default is None:
            self.default = self.non_optional_default()

            if self.default is None:
                raise ValueError(f'Non-optional descriptor {self.name_pub} has a default value of None.')

        # Store descriptor info on the class
        if not hasattr(owner, '_auto_init_descriptors'):
            owner._auto_init_descriptors = []

            original_init = owner.__init__

            # Wrap __init__ once for all descriptors
            def new_init(self, *args, **kwargs):
                # Initialize all descriptor private attributes FIRST
                for descriptor in type(self)._auto_init_descriptors:
                    setattr(self, descriptor.name_priv, descriptor.default)

                # Then call the original __init__
                original_init(self, *args, **kwargs)

            owner.__init__ = new_init

        # Add this descriptor to the list
        owner._auto_init_descriptors.append(self)

    def __get__(self, obj, objtype=None) -> T:
        """Get value."""
        return getattr(obj, self.name_priv)

    def __set__(self, obj, value: T) -> None:
        """Set value."""
        if value is None and not self.optional:
            raise ValueError(f'Cannot set "None" for non-optional parameter: {self.name_pub}')

        setattr(obj, self.name_priv, value)

    def __delete__(self, obj) -> None:
        """Delete (disabled).

        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(f'Attribute {self.name_pub}: Delete not allowed.')

    def _detect_optional(self, owner, name):
        """Check if the field is Optional based on type hints.

        Assumes a field is optional if it is missing or cannot be checked.
        """
        try:
            hints = get_type_hints(owner)

            if name not in hints:
                return True

            hint = hints[name]

            # Union detects "Optional[Type]", UnionType detects "Type | None"
            if get_origin(hint) in {Union, UnionType}:
                args = get_args(hint)
                # Optional[X] is Union[X, None]
                return type(None) in args

            return False

        except Exception:
            return True

class CheckedBool(AutoInitBase[bool]):

    def __init__(
            self,
            default: Optional[bool] = None,
            optional: Optional[bool] = None,
            name_priv: Optional[str] = None,
    ) -> None:
        """
        Create a checked boolean.

        If not specified, the value is initialized to None if optional or False if not optional.

        :param default: Default value. May not be `None` if the field is not optional.
        :param optional: If not `None`, Override typing hints and explicitly set if this attribute is optional (not
            recommended).
        :param name_priv: Private name. Defaults to public name prepended with "_". Can be a pattern including a "name"
            wildcard (e.g. "_{name}").
        """
        super().__init__(
            default=default,
            optional=optional,
            name_priv=name_priv
        )

    def __set__(self, obj, value) -> None:
        """Set value."""
        try:
            super().__set__(obj, bool(value) if value is not None else None)
        except TypeError:
            raise TypeError(f'Attribute {self.name_pub}: Value must be an integer: {value!r}')

    def non_optional_default(self) -> bool:
        """Called to get a default value when the parameter is not optional and the default is None."""
        return False

class CheckedObject(AutoInitBase[object]):

    def __init__(
            self,
            default: Optional[object] = None,
            optional: Optional[bool] = None,
            name_priv: Optional[str] = None,
    ) -> None:
        """
        Create a checked boolean.

        If not specified, the value is initialized to None if optional or False if not optional.

        :param default: Default value. May not be `None` if the field is not optional.
        :param optional: If not `None`, Override typing hints and explicitly set if this attribute is optional (not
            recommended).
        :param name_priv: Private name. Defaults to public name prepended with "_". Can be a pattern including a "name"
            wildcard (e.g. "_{name}").
        """
        super().__init__(
            default=default,
            optional=optional,
            name_priv=name_priv
        )

    def non_optional_default(self) -> object:
        """Called to get a default value when the parameter is not optional and the default is None."""
        raise ValueError(f'Attribute {self.name_pub}: Default value is required when not optional.')

class OneWayBool(CheckedBool):
    """One-way boolean. Once it changes from its default, it cannot change back."""
    def __init__(
            self,
            default: bool = False,
            name_priv: Optional[str] = None,
    ) -> None:
        """Create a OneWayBool descriptor.

        This value is not allowed to be optional. If type hints suggest it is optional, it is overridden.

        :param default: Default value. Once the instance value is changed from this value, it cannot be changed back.
        :param name_priv: Private name. Defaults to public name prepended with "_". Can be a pattern including a "name"
            wildcard (e.g. "_{name}").
        """
        super().__init__(
            default=bool(default),
            optional=False,
            name_priv=name_priv
        )

    def __set__(self, obj, value) -> None:
        """
        Check if value is frozen and set if not.

        :param obj: Object.
        :param value: Value.
        """
        try:
            value = bool(value)
        except TypeError:
            raise TypeError(f'Attribute {self.name_pub}: Value must be a boolean: {value}')

        value_existing = getattr(obj, self.name_priv)

        if value == value_existing:
            return

        if value_existing != self.default:
            raise ValueError(f'Attribute {self.name_pub}: Cannot set {value}: One-way boolean is frozen on {value_existing}')

        super().__set__(obj, value)


class BoundedInt(AutoInitBase[int]):
    """Integer descriptor with optional minimum and maximum value enforcement."""
    min_val: Optional[int]
    max_val: Optional[int]

    def __init__(
            self,
            min_val: Optional[int | tuple[int, bool]] = None,
            max_val: Optional[int | tuple[int, bool]] = None,
            allow_truncation: bool = False,
            default: Optional[int] = None,
            optional: Optional[bool] = None,
            name_priv: Optional[str] = None,
    ) -> None:
        """Create a bounded int descriptor.

        Minimum and maximum values may be specified as a numeric value or a tuple of a numeric value and a boolean.
        If specified, the boolean value determines if the bound is inclusive (True) or exclusive (False). Defaults
        to True (inclusive). If the numeric value is None (no bounds), then it is always exclusive (second parameter
        ignored if specified).

        If a value is optional, then it can accept `None`, otherwise, attempts to set `None` raise an exception. If
        the value is not optional and `default` is not specified, values are initialized to 0 instead of `None`.

        :param min_val: Minimum value. Can be a tuple of (value, inclusive).
        :param max_val: Maximum value. Can be a tuple of (value, inclusive).
        :param allow_truncation: Whether to allow truncation of the value when converting to integer. When `False`, the
            value "1.000" would be allowed, but "1.001" would not. When `True`, "1.001" would be silently truncated to
            "1".
        :param default: Default value. May not be `None` if the field is not optional.
        :param optional: If not `None`, Override typing hints and explicitly set if this attribute is optional (not
            recommended).
        :param name_priv: Private name. Defaults to public name prepended with "_". Can be a pattern including a "name"
            wildcard (e.g. "_{name}") where "name" is the public nome of this attribute.
        """
        super().__init__(
            default=default,
            optional=optional,
            name_priv=name_priv
        )

        if isinstance(min_val, tuple):
            min_val, min_inclusive = min_val
        else:
            min_val, min_inclusive = min_val, True

        if isinstance(max_val, tuple):
            max_val, max_inclusive = max_val
        else:
            max_val, max_inclusive = max_val, True

        self.min_val = int(min_val) if min_val is not None else None
        self.max_val = int(max_val) if max_val is not None else None
        self.min_inclusive = bool(min_inclusive) if self.min_val is not None else False
        self.max_inclusive = bool(max_inclusive) if self.max_val is not None else False
        self.allow_truncation = bool(allow_truncation)

    def __set__(self, obj, value) -> None:
        """Set value."""
        if value is None:
            super().__set__(obj, value)
            return

        org_value = value

        try:
            value = int(value)
        except TypeError:
            raise TypeError(f'Attribute {self.name_pub}: Value must be an integer: {value!r}')

        if not self.validate(value):
            raise ValueError(f'Attribute {self.name_pub}: Value out of range "{self.range_str}": {value:,g}')

        if not self.allow_truncation and value != float(org_value):
            raise ValueError(f'Attribute {self.name_pub}: Value must be an integer (truncation detected): {org_value!r}')

        super().__set__(obj, value)

    def validate(self, value) -> bool:
        """Check value against bounds."""
        return (
            (
                self.min_val is None
                or value > self.min_val
                or (self.min_inclusive and value == self.min_val)
            )
            & (
                self.max_val is None
                or value < self.max_val
                or (self.max_inclusive and value == self.max_val)
            )
        )

    def non_optional_default(self) -> int:
        """Called to get a default value when the parameter is not optional and the default is None."""
        return 0

    @property
    def range_str(self) -> str:
        """Get range string."""
        bracket_l = '[' if self.min_inclusive else '('
        bracket_r = ']' if self.max_inclusive else ')'

        lower_val = ('-Inf' if self.min_val is None else f'{self.min_val:,g}')
        upper_val = ('Inf' if self.max_val is None else f'{self.max_val:,g}')

        return f'{bracket_l}{lower_val}, {upper_val}{bracket_r}'


class BoundedFloat(AutoInitBase[float]):
    """Integer descriptor with optional minimum and maximum value enforcement."""
    min_val: Optional[float]
    max_val: Optional[float]
    min_inclusive: bool
    max_inclusive: bool

    def __init__(
            self,
            min_val: Optional[float | tuple[float, bool]] = None,
            max_val: Optional[float | tuple[float, bool]] = None,
            default: Optional[float] = None,
            optional: Optional[bool] = None,
            name_priv: Optional[str] = None,
    ) -> None:
        """Create a bounded float descriptor.

        Minimum and maximum values may be specified as a numeric value or a tuple of a numeric value and a boolean.
        If specified, the boolean value determines if the bound is inclusive (True) or exclusive (False). Defaults
        to True (inclusive). If the numeric value is None (no bounds), then it is always exclusive (second parameter
        ignored if specified).

        If a value is optional, then it can accept `None`, otherwise, attempts to set `None` raise an exception. If
        the value is not optional and `default` is not specified, values are initialized to 0 instead of `None`.

        :param min_val: Minimum value. Can be a tuple of (value, inclusive).
        :param max_val: Maximum value. Can be a tuple of (value, inclusive).
        :param default: Default value. May not be `None` if the field is not optional.
        :param optional: If not `None`, Override typing hints and explicitly set if this attribute is optional (not
            recommended).
        :param name_priv: Private name. Defaults to public name prepended with "_". Can be a pattern including a "name"
            wildcard (e.g. "_{name}") where "name" is the public nome of this attribute.
        """
        super().__init__(
            default=default,
            optional=optional,
            name_priv=name_priv
        )

        if isinstance(min_val, tuple):
            min_val, min_inclusive = min_val
        else:
            min_val, min_inclusive = min_val, True

        if isinstance(max_val, tuple):
            max_val, max_inclusive = max_val
        else:
            max_val, max_inclusive = max_val, True

        self.min_val = float(min_val) if min_val is not None else None
        self.max_val = float(max_val) if max_val is not None else None
        self.min_inclusive = bool(min_inclusive) if self.min_val is not None else False
        self.max_inclusive = bool(max_inclusive) if self.max_val is not None else False

    def __set__(self, obj, value) -> None:
        """Set value."""
        if value is None:
            super().__set__(obj, value)
            return

        try:
            value = float(value)
        except TypeError:
            raise TypeError(f'Attribute {self.name_pub}: Value must be an float: {value!r}')

        if not self.validate(value):
            raise ValueError(f'Attribute {self.name_pub}: Value out of range "{self.range_str}": {value:,g}')

        super().__set__(obj, value)

    def validate(self, value) -> bool:
        """Check value against bounds."""
        return (
            (
                self.min_val is None
                or value > self.min_val
                or (self.min_inclusive and value == self.min_val)
            )
            & (
                self.max_val is None
                or value < self.max_val
                or (self.max_inclusive and value == self.max_val)
            )
        )

    def non_optional_default(self) -> float:
        """Called to get a default value when the parameter is not optional and the default is None."""
        return 0.0

    @property
    def range_str(self) -> str:
        """Get range string."""
        bracket_l = '[' if self.min_inclusive else '('
        bracket_r = ']' if self.max_inclusive else ')'

        lower_val = ('-Inf' if self.min_val is None else f'{self.min_val:,g}')
        upper_val = ('Inf' if self.max_val is None else f'{self.max_val:,g}')

        return f'{bracket_l}{lower_val}, {upper_val}{bracket_r}'


class CheckedString(AutoInitBase[str]):
    """Enforces string constraints and supports a set of transformations before assignment."""
    min_len: Optional[int] = BoundedInt(0)
    max_len: Optional[int] = BoundedInt(0)
    strip: Callable[[str], str]
    match: Callable[[str], bool]

    def __init__(
            self,
            min_len: Optional[int] = None,
            max_len: Optional[int] = None,
            strip: bool | str = False,
            match: Optional[
                re.Pattern
                | str
                | Callable[[str], Any]
                | set[str]
            ] = None,
            sub: Optional[
                tuple[str, str]
                | tuple[re.Pattern, str]
                | Callable[[str], Optional[Any]]
                | Mapping[str, str]
                | tuple[Mapping[str, str], str]
            ] = None,
            default: Optional[str] = None,
            optional: Optional[bool] = None,
            name_priv: Optional[str] = None,
    ) -> None:
        """Create a checked string descriptor.

        Pattern matcher forms:

            * None: Always passes (all strings pass).
            * re.Pattern: Passes if pattern.fullmatch(val) has a match.
            * str: Complied to a pattern and treated like re.Pattern (above).
            * set[str]: Passes if the string is an exact match for any element in the set.
            * Callable[[str], bool]: Passes if the callable returns True. Can accept arbitrary lambdas.

        Pattern substitutions forms:

            * (str, str): A tuple of (match, replace) values where "match" is a regular expression and
                the second argument is a replacement string.
            * (re.Pattern, str): A compiled regular expression pattern and a replacement string.
            * Mapping[str, str]: A mapping of (match, replace) values.
            * Callable[[str], Optional[Any]]: A callable taking a string and returning a transformed
                string or None.
            * Mapping[str, str]: A mapping object that transforms keys to values. Raises KeyError if the
                key is not in the mapping.
            * (Mapping[str, str], str): A tuple of (mapping, missing) where "missing" is substituted
                for values not in the mapping.

        Substitution patterns with regex call pattern.sub().

        Order of evaluation:

            1. If value is None, it is not transformed further. It is set as a missing value (if
                allowed, raises an exception if the field is not optional).
            2. Apply strip.
            3. Apply substitutions.
            4. Check length
            5. Check pattern match.
            6. Set value

        :param min_len: Minimum string length.
        :param max_len: Maximum string length.
        :param strip: Strip whitespace if True, do not alter string if False. If this attribute is
            a string, use it as the strip pattern.
        :param match: Match patterns. If defined, the string must pass this pattern.
        :param sub: Substitution patterns. If defined, the string is transformed before setting.
        :param default: Default value. May not be `None` if the field is not optional.
        :param optional: If not `None`, Override typing hints and explicitly set if this attribute is optional (not
            recommended).
        :param name_priv: Private name. Defaults to public name prepended with "_". Can be a pattern including a "name"
            wildcard (e.g. "_{name}") where "name" is the public nome of this attribute.
        """
        super().__init__(
            default=default,
            optional=optional,
            name_priv=name_priv
        )

        if strip is not None:
            if isinstance(strip, bool):
                strip = str.strip if strip else lambda val: val
            elif isinstance(strip, str):
                strip = lambda val: val.strip(strip)
            else:
                raise TypeError(f'Attribute {self.name_pub}: strip must be a bool or str: {strip!r}')
        else:
            strip = lambda val: val  # Identity function

        self.min_len = min_len
        self.max_len = max_len
        self.strip = strip
        self.match = _get_pattern_matcher(match)
        self.sub = _get_pattern_sub(sub)

    def __set__(self, obj, value) -> None:
        """Set value."""
        super().__set__(obj, self.validate(value))

    def non_optional_default(self) -> T:
        """Non-optional default value."""
        return ''

    def validate(self, value) -> Optional[str]:
        """Transform and validate string.

        :param value: Value string or None.

        :return: Transformed and validated value or `None` if `value` is `None`.

        :raises ValueError: If value fails validation.
        """
        if value is None:
            return None

        value = self.strip(str(value))
        value = self.sub(value)

        if self.min_len is not None and len(value) < self.min_len:
            raise ValueError(
                f'Attribute {self.name_pub}: String length must be at least {self.min_len} '
                f'(len={len(value)}): {value!r}'
            )

        if self.max_len is not None and len(value) > self.max_len:
            raise ValueError(
                f'Attribute {self.name_pub}: String length must be at most {self.max_len} '
                f'(len={len(value)}):  {value!r}'
            )

        if not self.match(value):
            raise ValueError(f'Attribute {self.name_pub}: String fails pattern matcher {self.match!r} (len={len(value)}): {value!r}')

        return value

def _get_pattern_matcher(
    pattern: Optional[re.Pattern | str | Callable[[str], Any]] | set[str] = None,
) -> Callable[[str], Any]:
    """Get a pattern matcher taking on several different forms.

    See `meth:CheckedString.__init__` for details on input parameters.

    The returned callable takes a single string argument and returns a value. The value should be
    interpreted as "truthy" or "falsy" by the caller, but may or may not be a boolean. Using
    the returned value in an `if` statement will work. For example, if "matcher" is a matching
    function returned by this function, then `if matcher(val): ...` will work as expected. It
    can also be explicitly transformed to bool (e.g. `bool(matcher(val))` or
    `operator.truth(matcher(val))`) to yield a boolean value.

    :param pattern: Pattern to match.

    :return: A callable taking a single string (not optional, never None) and returning a
        truthy or falsy value (see above).

    :raises TypeError: If `pattern` is not one of the expected types.
    """
    if pattern is None:
        return lambda val: True

    if isinstance(pattern, re.Pattern):
        return pattern.fullmatch

    if isinstance(pattern, str):
        return re.compile(pattern).fullmatch

    if isinstance(pattern, set):
        return functools.partial(operator.contains, pattern)

    if callable(pattern):
        return pattern

    raise TypeError(f'Pattern must be a re.Pattern, str, Callable[[str], Any], or None]: {pattern!r}')

def _get_pattern_sub(
        sub: Optional[
            tuple[str, str]
            | tuple[re.Pattern, str]
            | Callable[[str], Optional[Any]]
            | Mapping[str, str]
            | tuple[Mapping[str, str], str]
        ] = None,
):
    """Get a pattern substitution function.

    See `meth:CheckedString.__init__` for details on input parameters.

    :param sub: Sub pattern or function.

    :return: A callable taking a string and returning a transformed string.

    :raises TypeError: If `sub` is not one of the expected types including the case where it is a
        tuple with the wrong number of elements.
    """
    if sub is None:
        return lambda val: val

    if isinstance(sub, Mapping):
        return _MappableSub(sub)

    if callable(sub):
        return _StrOrNone(sub)

    if not isinstance(sub, tuple):
        raise TypeError(f'Pattern must be a of (match, replace) values or None: {sub!r}')

    if len(sub) != 2:
        raise TypeError(f'Pattern must be a of (match, replace) values or None: Found {len(sub)} elements: {sub!r}')

    pattern, replace = sub

    if isinstance(pattern, re.Pattern):
        if not isinstance(replace, str):
            raise TypeError(f'Pattern must be a of (match, replace) values or None: Found non-str replace: {replace!r}')

        return functools.partial(pattern.sub, replace)

    if isinstance(pattern, str):
        if not isinstance(replace, str):
            raise TypeError(f'Pattern must be a of (match, replace) values or None: Found non-str replace: {replace!r}')

        return functools.partial(re.compile(pattern).sub, replace)

    if isinstance(pattern, Mapping):
        return _MappableSub(pattern, replace, True)

    raise TypeError(f'Pattern must be a of (match, replace) values or None: Found non-str pattern: {pattern!r}')

class _StrOrNone:
    """Call function, cast value to string if not None, else, return None."""
    c: Callable[[str], Optional[Any]]

    def __init__(self, c: Callable[[str], Optional[Any]]):
        self.c = c

    def __call__(self, val: str):
        if (new_val := self.c(val)) is None:
            return None
        return str(new_val)

class _MappableSub:
    """Calls a mapping to transform a string."""
    m: Mapping[str, Any]
    default: Optional[str]
    allow_missing: bool

    def __init__(
            self,
            m: Mapping[str, Any],
            default: Optional[str] = None,
            allow_missing: bool = False,
    ):
        self.m = m
        self.default = default
        self.allow_missing = allow_missing

    def __call__(self, val: str):
        try:
            new_val = self.m[val]
        except KeyError:
            if not self.allow_missing:
                raise ValueError(f'Key {val!r} not found in {_key_err_str(self.m.keys())}')
            new_val = self.default

        return str(new_val) if new_val is not None else None

def _key_err_str(i: Iterable[Any], limit=50):
    """Get a string representation of a collection of keys up to a limit on the total length of keys."""
    total_len = 0
    iter_len = 0
    stopped_at_limit = False

    key_list = []

    for val in i:
        if total_len > limit:
            stopped_at_limit = True
            break

        iter_len += 1
        val = repr(val)
        total_len += len(val)
        key_list.append(val + ',')

    return f'{{{" ".join(key_list)}{" ..." if stopped_at_limit else ""}}}'
