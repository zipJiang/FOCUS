"""Lightweight registries for construction from config mappings.

Registration and construction are *free functions* keyed by an interface class -- not
methods on a base class. Any class can be a registry namespace: call ``register(cls,
...)`` to add an implementation and ``build(cls, config)`` to construct one, with no
``Registrable`` inheritance required. The interface class is just the dict key that
gives each namespace its own set of names.

Non-default construction semantics are declared with the optional ``@registrable``
decorator (a config-bundle that ``builds_self``, or a custom error ``slot`` label).

**Annotations are what makes construction recursive.** A config is a nested mapping, and
nothing in it says which of its values are objects to build -- the *constructor* does,
in its own type hints. So ``build`` resolves a name to an entry and calls
``_construct``, which reads the target's annotations (``_signature_annotations``, class
hints and ``__init__`` hints together, resolved in the module each was written in) and
hands every value it is about to pass to ``_resolve_nested_value``. That asks the
annotation two questions: does it name a registry namespace (``_config_target``, which
unwraps ``X | None`` and generic aliases), and is it a homogeneous sequence of one
(``_sequence_member``)? A yes to either sends the value back through ``build`` -- one
level deeper, under the interface the field declared -- and a no leaves it exactly as it
came, so an already-built object, an injected dependency or a plain list of ints is
never forced through a registry. A ``_target_`` mapping short-circuits the walk into
``construct_target``, which is the escape hatch for anything unregistered.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import re
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from functools import partial
from typing import (
    Any,
    Protocol,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

NestedBuilder = Callable[..., Any]
Constructor = Callable[..., Any]


@dataclass(frozen=True)
class _RegistryEntry:
    target: Any
    constructor: str | Constructor | None
    nested_builders: Mapping[str, NestedBuilder]


@dataclass(frozen=True)
class _Interface:
    """The construction semantics of one registry namespace (an interface class).

    Defaults suit a polymorphic interface: pick an implementation by ``name`` and
    validate the result with ``isinstance``. ``builds_self`` lets a name-less config
    build the interface class itself (a config bundle); ``slot`` overrides the
    humanized class name used in error messages.
    """

    builds_self: bool = False
    slot: str | None = None


# Central tables keyed by interface class -- one namespace per class, no shared bucket.
_REGISTRY: dict[type, dict[str, _RegistryEntry]] = {}
_INTERFACES: dict[type, _Interface] = {}
_DEFAULT_INTERFACE = _Interface()


def _interface(cls: type) -> _Interface:
    # Walk the MRO so a subclass inherits its interface's declared semantics.
    for base in cls.__mro__:
        spec = _INTERFACES.get(base)
        if spec is not None:
            return spec
    return _DEFAULT_INTERFACE


def registrable[T](
    *,
    builds_self: bool = False,
    slot: str | None = None,
) -> Callable[[type[T]], type[T]]:
    """Declare an interface's construction semantics. Returns a class decorator.

    Optional: a class needs this only for non-default behavior. Plain ``register`` /
    ``build`` already work on any undecorated class with the default semantics.
    """

    def decorator(target: type[T]) -> type[T]:
        _INTERFACES[target] = _Interface(builds_self=builds_self, slot=slot)
        return target

    return decorator


def _humanize(name: str) -> str:
    """CamelCase to a lower-case spaced label (``StateStore`` -> ``state store``)."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name)
    return spaced.lower()


def _slot_name(interface: type) -> str:
    return _interface(interface).slot or _humanize(interface.__name__)


def _mapping_without_name(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in {"name", "_target_"}}


def _dataclass_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, type) or not is_dataclass(value):
        return None
    return {
        config_field.name: getattr(value, config_field.name)
        for config_field in fields(value)
    }


def construct_target(value: Mapping[str, Any], **dependencies: Any) -> Any:
    """Construct an object from a mapping with a ``_target_`` import path."""

    target = value.get("_target_")
    if not isinstance(target, str):
        raise ValueError("_target_ must be an import string")
    module_name, _, attr_name = target.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError("_target_ must include a module and attribute")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    return _construct(factory, _mapping_without_name(value), dependencies)


def _constructor_for(entry: _RegistryEntry) -> Constructor:
    # ``target`` is deliberately ``Any`` -- the registry holds arbitrary classes and
    # factories -- so the two dynamic lookups are cast rather than inferred. What makes
    # them callable is checked where they are registered, not here.
    if entry.constructor is None:
        return cast(Constructor, entry.target)
    if isinstance(entry.constructor, str):
        return cast(Constructor, getattr(entry.target, entry.constructor))
    return entry.constructor


def _accepts(constructor: Constructor) -> tuple[bool, set[str]]:
    """``(takes **kwargs, the keyword names it takes)`` for one callable.

    Both halves are one question -- what may be passed by name -- and both callers ask
    it the same way: a ``**kwargs`` constructor takes everything, so nothing is filtered
    out and no key is "unexpected"; one without takes exactly the names below, and
    anything else is a config key naming a parameter that does not exist.

    Positional-only parameters are deliberately absent: this module only ever calls by
    keyword, so a name that cannot be passed that way is not a name it can fill.
    """
    parameters = inspect.signature(constructor).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    accepted_names = {
        name
        for name, parameter in parameters.items()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    return accepts_kwargs, accepted_names


def _call_nested_builder(
    builder: NestedBuilder,
    value: Any,
    dependencies: Mapping[str, Any],
) -> Any:
    accepts_kwargs, accepted_names = _accepts(builder)
    if accepts_kwargs:
        return builder(value, **dependencies)
    kwargs = {
        name: dependency
        for name, dependency in dependencies.items()
        if name in accepted_names
    }
    return builder(value, **kwargs)


def _hints_in_module(source: Any, module_name: str | None) -> dict[str, Any]:
    globalns = sys.modules[module_name].__dict__ if module_name in sys.modules else None
    try:
        return get_type_hints(source, globalns=globalns)
    except Exception:
        return getattr(source, "__annotations__", {})


def _signature_annotations(constructor: Constructor) -> dict[str, Any]:
    if not inspect.isclass(constructor):
        return _hints_in_module(constructor, getattr(constructor, "__module__", None))

    # Class-level hints first: ``get_type_hints(cls)`` walks the MRO and resolves each
    # base's annotations in its own module (so cross-module / local-subclass fields
    # resolve); then add ``__init__``-only params without clobbering the resolved hints.
    hints: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        hints.update(get_type_hints(constructor))
    for name, annotation in _hints_in_module(
        constructor.__init__, constructor.__module__
    ).items():
        hints.setdefault(name, annotation)
    return hints


def _without_none(annotation: Any) -> Any:
    """``X | None`` / ``Optional[X]`` -> ``X``; anything else unchanged.

    A field is optional for reasons that have nothing to do with construction (``None``
    meaning "the scheduler's own default"), so the option wrapper must not hide the
    annotation that says how to build the value inside it.
    """
    if get_origin(annotation) in (Union, types.UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _config_target(annotation: Any) -> type | None:
    """Return the interface class an annotation resolves to, else ``None``.

    An interface is any class that is a registry namespace -- one with registered
    implementations or an explicit ``@registrable`` declaration. Unwraps ``X | None`` /
    ``Optional[X]`` and generic aliases first.
    """
    annotation = _without_none(annotation)
    annotation = get_origin(annotation) or annotation
    if inspect.isclass(annotation) and any(
        base in _REGISTRY or base in _INTERFACES for base in annotation.__mro__
    ):
        return annotation
    return None


def _sequence_member(annotation: Any) -> tuple[type, Any] | None:
    """``(container, member)`` for a homogeneous sequence field, else ``None``.

    A field holding *several* implementations -- ``passes: tuple[SearchPass,
    ...]`` -- is as buildable from a config file as one holding a single one, and it is
    the annotation that says so. Without this a list of ``{"name": ...}`` mappings
    arrives as a list of dicts: a field that looks configured and behaves defaulted, or
    raises three layers down. Heterogeneous tuples (``tuple[A, B]``) are left alone --
    there is no one member type to build against.
    """
    annotation = _without_none(annotation)
    origin = get_origin(annotation)
    if not inspect.isclass(origin) or not issubclass(origin, Sequence):
        return None
    if issubclass(origin, (str, bytes)):
        return None
    args = [arg for arg in get_args(annotation) if arg is not Ellipsis]
    if len(args) != 1:
        return None
    return list if issubclass(origin, list) else tuple, args[0]


def _resolve_nested_value(
    value: Any,
    annotation: Any,
    dependencies: Mapping[str, Any],
) -> Any:
    if isinstance(value, Mapping) and "_target_" in value:
        return construct_target(value, **dependencies)
    # Only build config forms (a mapping, or a string registry-name/spec); anything else
    # -- an already-constructed object (incl. an injected dependency) or ``None`` -- is
    # used as-is, so non-registrable instances are never forced through ``build``. Guard
    # first so the annotation walk is skipped entirely for those.
    if isinstance(value, (Mapping, str)):
        target = _config_target(annotation)
        if target is not None:
            return build(target, value, **dependencies)
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        member = _sequence_member(annotation)
        # Only descend when the members are themselves buildable: every other list on a
        # config (an ``env`` mapping's payload, a plain list of ints) is data, and
        # rebuilding its container would be a silent retyping nobody asked for.
        if member is not None and _config_target(member[1]) is not None:
            container, member_annotation = member
            return container(
                _resolve_nested_value(item, member_annotation, dependencies)
                for item in value
            )
    return value


def _construct(
    constructor: Constructor,
    config_kwargs: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    *,
    nested_builders: Mapping[str, NestedBuilder] | None = None,
) -> Any:
    annotations = _signature_annotations(constructor)
    accepts_kwargs, accepted_names = _accepts(constructor)

    unknown_config = set(config_kwargs) - accepted_names
    if unknown_config and not accepts_kwargs:
        unknown = ", ".join(sorted(unknown_config))
        raise TypeError(f"unexpected config keys for {constructor}: {unknown}")

    merged = dict(dependencies)
    merged.update(config_kwargs)
    if not accepts_kwargs:
        merged = {
            name: value for name, value in merged.items() if name in accepted_names
        }

    explicit_nested_builders = dict(nested_builders or {})
    for name, value in list(merged.items()):
        if name in explicit_nested_builders:
            merged[name] = _call_nested_builder(
                explicit_nested_builders[name],
                value,
                dependencies,
            )
            continue
        if name in annotations:
            merged[name] = _resolve_nested_value(value, annotations[name], dependencies)

    return constructor(**merged)


@overload
def register[T](
    interface: type,
    name: str,
    target: T,
    *,
    aliases: tuple[str, ...] = (),
    constructor: str | Constructor | None = None,
    nested_builders: Mapping[str, NestedBuilder] | None = None,
    replace: bool = False,
) -> T: ...


@overload
def register[T](
    interface: type,
    name: str,
    target: None = None,
    *,
    aliases: tuple[str, ...] = (),
    constructor: str | Constructor | None = None,
    nested_builders: Mapping[str, NestedBuilder] | None = None,
    replace: bool = False,
) -> Callable[[T], T]: ...


def register(
    interface: type,
    name: str,
    target: Any | None = None,
    *,
    aliases: tuple[str, ...] = (),
    constructor: str | Constructor | None = None,
    nested_builders: Mapping[str, NestedBuilder] | None = None,
    replace: bool = False,
) -> Any:
    """Register an implementation under ``interface``, or return a decorator."""

    def decorator[T](registered_target: T) -> T:
        names = (name, *aliases)
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate {_slot_name(interface)} registry names")
        bucket = _REGISTRY.setdefault(interface, {})
        if not replace:
            for registered_name in names:
                if registered_name in bucket:
                    raise ValueError(
                        f"{_slot_name(interface)} {registered_name!r} "
                        + "is already registered"
                    )
        entry = _RegistryEntry(
            target=registered_target,
            constructor=constructor,
            nested_builders=dict(nested_builders or {}),
        )
        for registered_name in names:
            bucket[registered_name] = entry
        return registered_target

    return decorator if target is None else decorator(target)


class Registrar(Protocol):
    """:func:`register` with one interface already bound.

    Both call forms of :func:`register` survive the binding (pass a target, or use it
    as a decorator); the protocol exists so a package-level helper such as
    ``register_policy`` keeps that typing instead of degrading to ``Any``.
    """

    @overload
    def __call__[T](
        self,
        name: str,
        target: T,
        *,
        aliases: tuple[str, ...] = (),
        constructor: str | Constructor | None = None,
        nested_builders: Mapping[str, NestedBuilder] | None = None,
        replace: bool = False,
    ) -> T: ...

    @overload
    def __call__[T](
        self,
        name: str,
        target: None = None,
        *,
        aliases: tuple[str, ...] = (),
        constructor: str | Constructor | None = None,
        nested_builders: Mapping[str, NestedBuilder] | None = None,
        replace: bool = False,
    ) -> Callable[[T], T]: ...


def registrar(interface: type) -> Registrar:
    """Bind :func:`register` to ``interface``, e.g. for a package's ``register_x``."""

    bound = partial(register, interface)
    # A bare partial reports ``functools.partial``'s own docstring, and these bound
    # helpers are what the docs send an author to -- so ``help(register_policy)``
    # answering "create a new function with partial application" is the wrong answer to
    # the only question anyone asks it.
    bound.__doc__ = (
        f"Register an implementation of {interface.__name__} under a name: "
        + "``register_x(name, cls)``, or ``@register_x(name)`` as a decorator. "
        + "See :func:`focus.registry.register`."
    )
    return bound


def registered_interfaces() -> tuple[type, ...]:
    """Every interface class that has registered implementations.

    With :func:`registered_names`, this makes the config surface *enumerable* rather
    than something a reader reconstructs by grepping for ``@register``: the docs' slot
    table and the test that keeps it honest both read it from here.
    """

    return tuple(_REGISTRY)


def registered_names(interface: type) -> tuple[str, ...]:
    """The names registered under ``interface``, sorted; aliases included."""

    return tuple(sorted(_REGISTRY.get(interface, {})))


def build[T](
    interface: type[T], config: Mapping[str, Any] | T, **dependencies: Any
) -> T:
    """Construct an object registered under ``interface`` from ``config``.

    ``config`` may be an already-built value (returned as-is), a ``_target_`` import
    mapping, a name-less bundle when ``interface`` ``builds_self``, or a ``{"name":
    ...}`` mapping / dataclass selecting a registered implementation.
    """

    if isinstance(config, interface):
        return config

    mapping = config if isinstance(config, Mapping) else _dataclass_mapping(config)
    if mapping is None:
        config_type = type(config).__name__
        slot = _slot_name(interface)
        raise TypeError(
            f"unsupported {slot} config {config_type}: expected a mapping like "
            + '{"name": ...} with the constructor\'s keyword arguments beside it, '
            + f"a built {interface.__name__}, or a dataclass of either shape"
        )

    if "_target_" in mapping:
        built = construct_target(mapping, **dependencies)
        return _validate(interface, "_target_", built)

    name = mapping.get("name")
    if name is None and _interface(interface).builds_self:
        # Concrete config bundle: build this interface itself from the mapping.
        built = _construct(interface, _mapping_without_name(mapping), dependencies)
        return _validate(interface, interface.__name__, built)
    if not isinstance(name, str):
        raise ValueError(f"{_slot_name(interface)} config must include a string name")
    entry = _REGISTRY.get(interface, {}).get(name)
    if entry is None:
        # The available names cost nothing to say and are the whole content of the
        # answer to "then what should I have written?".
        known = ", ".join(registered_names(interface)) or "(none)"
        raise ValueError(
            f"unknown {_slot_name(interface)} {name!r}; registered: {known}"
        )

    built = _construct(
        _constructor_for(entry),
        _mapping_without_name(mapping),
        dependencies,
        nested_builders=entry.nested_builders,
    )
    return _validate(interface, name, built)


def _validate[T](interface: type[T], name: str, built: Any) -> T:
    if not isinstance(built, interface):
        raise TypeError(
            f"{_slot_name(interface)} factory {name!r} returned {type(built).__name__}"
        )
    return built


__all__ = [
    "Constructor",
    "NestedBuilder",
    "Registrar",
    "build",
    "construct_target",
    "register",
    "registered_interfaces",
    "registered_names",
    "registrable",
    "registrar",
]
