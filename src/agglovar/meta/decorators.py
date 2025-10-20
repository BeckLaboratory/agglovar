"""Decorators for common functionality.

Defines several decorators for adding common functionality to classes and methods.

Decorators include:

- `immutable`: Make a class immutable after __init__ completes.
- `lockable`: Sets a lock flag (o.is_locked) and provides a `lock()` method and on optional `unlock()` method when the
    lock is reversible. The meaning of the lock is up to the class using `lockable`. Object attributes are immutable
    while locked (can be disabled) or are always immutable (`immutable` flag).

These decorators use instance variables to track state, which are automatically when they collide with variables defined
in the decorated class.
"""

__all__ = [
    'lockable',
    'immutable',
]

from typing import Container

from ..util.str import collision_rename


def lockable(
        cls=None, *,
        reversible: bool = False,
        attrs: bool = True,
        immutable: bool = False,
        var_name: str = '_lock',
):
    """Decorator to make a class lockable.

    Creates a lock variable that can be checked to see if the object is locked. Semantics of what "locked" means is
    defined by the class using this decorator. If "attrs" is `True`, then object attributes cannot be modified while
    the object is locked. If "immutable" is `True`, then object attributes cannot be modified at any time.

    A lock variable is added to the class with name `var_name`. If the class defines a variable with the same name, it
    will be renamed to avoid conflicts. For immutable objects, a flag is added to the object so __init__ can modify
    attributes, and the flag is removed after __init__ completes. The flag is called "_lockable_in_init", which is
    also renamed to avoid collisions.

    :param reversible: If True, allow object to be unlocked. Adds an "unlock()" method.
    :param attrs: If True, attributes cannot be modified while the object is locked.
    :param immutable: If True, class attributes are always immutable whether or not the instance is locked.
    :param var_name: Lock variable name. Change if "_lock" is already used by the class.
    """

    # Handle @lockable() syntax
    if cls is None:
        return lambda cls: lockable(
            cls,
            reversible=reversible,
            attrs=attrs,
            immutable=immutable,
            var_name=var_name
        )

    # Handle @lockable(True) case
    if not (isinstance(cls, type) or isinstance(cls, type)):
        raise TypeError(
            "@lockable() decorator requires keyword arguments. "
            "Use @lockable or @lockable(reversible=True/False, ...)"
        )

    # Save original methods if they exist
    original_init = cls.__init__

    # Avoid name conflicts
    var_name = _collision_rename(var_name, cls.__dict__)
    in_init_flag = _collision_rename('_lockable_in_init', cls.__dict__)

    # Set class functions
    if immutable:
        def new_init(self, *args, **kwargs):
            object.__setattr__(self, in_init_flag, True)
            object.__setattr__(self, var_name, False)
            original_init(self, *args, **kwargs)
            object.__delattr__(self, in_init_flag)

        def new_setattr(self, name, value):
            if getattr(self, in_init_flag, False):
                object.__setattr__(self, name, value)
                return

            raise AttributeError(f'Cannot modify attribute "{name}": Object is immutable')

        def new_delattr(self, name):
            if getattr(self, in_init_flag, False):
                object.__delattr__(self, name)
                return

            raise AttributeError(f"Cannot delete attribute '{name}': Object is immutable")

        cls.__init__ = new_init
        cls.__setattr__ = new_setattr
        cls.__delattr__ = new_delattr

    elif attrs:
        # Create a new __setattr__ method
        def new_setattr(self, name, value):
            if getattr(self, var_name, False):
                raise AttributeError(f"Cannot modify attribute '{name}': Object is locked")

            if name == var_name:
                raise AttributeError(f"Cannot modify lock '{name}'")

            object.__setattr__(self, name, value)

        # Create a new __delattr__ method
        def new_delattr(self, name, value):
            if getattr(self, var_name, False):
                raise AttributeError(f"Cannot modify attribute '{name}': Object is locked")

            if name == var_name:
                raise AttributeError(f"Cannot delete lock '{name}'")

            object.__setattr__(self, name, value)

        cls.__setattr__ = new_setattr
        cls.__delattr__ = new_delattr

    if not immutable:
        def new_init(self, *args, **kwargs):
            object.__setattr__(self, var_name, False)
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init

    def lock(self):
        """Lock the object to prevent further modifications"""
        object.__setattr__(self, var_name, True)

    cls.lock = lock

    def check_lock(
            self,
            msg: str = None,
            suffix: bool = True
    ):
        """Check if object is locked.

        :param msg: Message to include in the error.
        :param suffix: If True, append ": Object is locked" to the message.
        """
        if getattr(self, var_name, False):
            if msg is not None:
                tail = ': Object is locked' if suffix else ''
                raise AttributeError(f'{msg}{tail}')
            raise AttributeError(f"Object is locked")

    cls.check_lock = check_lock

    if reversible:
        def unlock(self):
            """Unlock the object to allow further modifications"""
            object.__setattr__(self, var_name, False)

        cls.unlock = unlock

    @property
    def is_locked(self):
        """Check if object is locked"""
        return getattr(self, var_name)

    cls.is_locked = is_locked

    return cls


def immutable(cls=None):
    """Decorator to make a class immutable after __init__ completes."""

    # Handle @immutable() syntax
    if cls is None:
        return lambda cls: immutable(cls)

    # Store the original methods
    original_init = cls.__init__

    in_init_flag = _collision_rename('_immutable_in_init', cls.__dict__)

    # Create a new __init__ method
    def new_init(self, *args, **kwargs):
        object.__setattr__(self, in_init_flag, True)
        original_init(self, *args, **kwargs)
        delattr(self, in_init_flag)

    # Create a new __setattr__ method
    def new_setattr(self, name, value):
        if getattr(self, in_init_flag, False):
            object.__setattr__(self, name, value)
            return

        raise AttributeError(f'Cannot modify attribute "{name}": Object is immutable')

    # Create a new __delattr__ method
    def new_delattr(self, name):
        if getattr(self, in_init_flag, False):
            object.__delattr__(self, name)
            return

        raise AttributeError(f'Cannot delete attribute "{name}": Object is immutable')

    # Replace the methods
    cls.__init__ = new_init
    cls.__setattr__ = new_setattr
    cls.__delattr__ = new_delattr

    return cls

def _collision_rename(
        var_name: str,
        *args: Container[str]
) -> str:
    """Rename a variable to avoid collisions with existing variables.

    Adds "_n" to a variable name where "n" is the lowest integer that does not collide with an existing name. If
    `var_name` does not collide, it is returned unmodified.

    :param var_name: Variable name.
    :param args: One or more (like "cls") to search.

    :return: Variable name.
    """

    return collision_rename(var_name, '_', *args)
