# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for working with references."""

from __future__ import annotations

import ast
import builtins
import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import itertools
import json
import math
import sys
import urllib.request
import webbrowser
from collections import defaultdict, deque
from functools import cached_property, lru_cache, partial
from types import ModuleType, MethodWrapperType

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import DefineMixin, define, get_type_and_kind, get_attr, get_attrs
from vectorbtpro.utils.config import Configured, merge_dicts, get
from vectorbtpro.utils.module_ import import_module, resolve_module, get_module, package_shortcut_config
from vectorbtpro.utils.warnings_ import warn

__all__ = [
    "get_refname",
    "get_obj",
    "imlucky",
    "get_api_ref",
    "open_api_ref",
    "RefGraph",
    "RefIndex",
]

if tp.TYPE_CHECKING:
    from networkx import DiGraph as DiGraphT
else:
    DiGraphT = "networkx.DiGraph"


class ReferenceResolutionError(LookupError):
    """Base class for refname resolution errors."""


class ReferenceNotFoundError(ReferenceResolutionError):
    """Raised when no reference matches the query."""


class AmbiguousReferenceError(ReferenceResolutionError):
    """Raised when multiple references match the query."""


def get_caller_qualname() -> tp.Optional[str]:
    """Return the qualified name of the calling function or method.

    Returns:
        Optional[str]: Qualified name of the function or method that invoked this function.
    """
    frame = inspect.currentframe()
    try:
        caller_frame = frame.f_back
        code = caller_frame.f_code
        func_name = code.co_name
        locals_ = caller_frame.f_locals
        if "self" in locals_:
            cls = locals_["self"].__class__
            return f"{cls.__qualname__}.{func_name}"
        elif "cls" in locals_:
            cls = locals_["cls"]
            return f"{cls.__qualname__}.{func_name}"
        else:
            module = inspect.getmodule(caller_frame)
            if module:
                func = module.__dict__.get(func_name, None)
                if func:
                    qualname = get_attr(func, "__qualname__", None)
                    if qualname is not None and isinstance(qualname, str):
                        return qualname
            return func_name
    finally:
        del frame


def get_method_class(meth: tp.Callable) -> tp.Optional[tp.Type]:
    """Return the class associated with the given method, if available.

    Args:
        meth (Callable): Method or function for which to determine the associated class.

    Returns:
        Optional[type]: Class object if found, otherwise None.
    """
    if inspect.ismethod(meth) or (
        inspect.isbuiltin(meth)
        and get_attr(meth, "__self__", None) is not None
        and get_attr(meth.__self__, "__class__", None) is not None
    ):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        meth = get_attr(meth, "__func__", meth)
    if inspect.isfunction(meth):
        cls = get_attr(get_module(meth), meth.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[0], None)
        if cls is not None and isinstance(cls, type):
            return cls
    return get_attr(meth, "__objclass__", None)


def get_obj_refname(obj: tp.Any, allow_type_fallback: bool = True) -> tp.Optional[str]:
    """Return the reference name for the provided object.

    Args:
        obj (Any): Target object.
        allow_type_fallback (bool): Whether to fallback to the type's reference name
            if no direct reference name is found.

    Returns:
        Optional[str]: Reference name, or None if not found.
    """
    from vectorbtpro.utils.decorators import class_property, custom_property, hybrid_property

    if inspect.ismodule(obj):
        name = get_attr(obj, "__name__", None)
        if name is not None and isinstance(name, str):
            return name
    if inspect.isclass(obj):
        module = get_attr(obj, "__module__", None)
        qualname = get_attr(obj, "__qualname__", None)
        if module is not None and qualname is not None and isinstance(module, str) and isinstance(qualname, str):
            return module + "." + qualname

    if isinstance(obj, staticmethod):
        func = get_attr(obj, "__func__", None)
        if func is not None:
            return get_obj_refname(func, allow_type_fallback=allow_type_fallback)
    if isinstance(obj, classmethod):
        func = get_attr(obj, "__func__", None)
        if func is not None:
            return get_obj_refname(func, allow_type_fallback=allow_type_fallback)

    if inspect.isbuiltin(obj) or isinstance(obj, MethodWrapperType):
        self_obj = get_attr(obj, "__self__", None)
        name = get_attr(obj, "__name__", None)
        if self_obj is not None and name is not None and not inspect.ismodule(self_obj) and isinstance(name, str):
            return get_obj_refname(type(self_obj), allow_type_fallback=allow_type_fallback) + "." + name
        module = get_module(obj)
        if module is not None and name is not None and isinstance(name, str):
            return get_obj_refname(module, allow_type_fallback=allow_type_fallback) + "." + name

    if (
        inspect.isdatadescriptor(obj)
        or inspect.ismethoddescriptor(obj)
        or inspect.isgetsetdescriptor(obj)
        or inspect.ismemberdescriptor(obj)
    ):
        cls = get_attr(obj, "__objclass__", None)
        name = get_attr(obj, "__name__", None)
        if cls is not None and name is not None and inspect.isclass(cls) and isinstance(name, str):
            return get_obj_refname(cls, allow_type_fallback=allow_type_fallback) + "." + name

    if inspect.ismethod(obj) or inspect.isfunction(obj):
        cls = get_method_class(obj)
        if cls is not None:
            name = get_attr(obj, "__name__", None)
            if name is not None and isinstance(name, str):
                return get_obj_refname(cls, allow_type_fallback=allow_type_fallback) + "." + name
        func = get_attr(obj, "func", None, resolve_descriptor=True)
        if func is not None:
            return get_obj_refname(func, allow_type_fallback=allow_type_fallback)
    if isinstance(obj, partial):
        func = get_attr(obj, "func", None, resolve_descriptor=True)
        if func is not None:
            return get_obj_refname(func, allow_type_fallback=allow_type_fallback)

    if isinstance(obj, (class_property, hybrid_property, custom_property)):
        return get_obj_refname(obj.func, allow_type_fallback=allow_type_fallback)
    if isinstance(obj, cached_property):
        func = get_attr(obj, "func", None, resolve_descriptor=True)
        if func is not None:
            return get_obj_refname(func, allow_type_fallback=allow_type_fallback)
    if isinstance(obj, property):
        return get_obj_refname(obj.fget, allow_type_fallback=allow_type_fallback)

    name = get_attr(obj, "__qualname__", None)
    if name is not None and isinstance(name, str):
        module = get_module(obj)
        if module is not None and name in module.__dict__:
            return get_obj_refname(module, allow_type_fallback=allow_type_fallback) + "." + name

    module = get_module(obj)
    if module is not None:
        for k, v in list(module.__dict__.items()):
            if obj is v:
                return get_obj_refname(module, allow_type_fallback=allow_type_fallback) + "." + k

    if allow_type_fallback:
        return get_obj_refname(type(obj), allow_type_fallback=allow_type_fallback)

    return None


def annotate_refname_parts(refname: str, allow_partial: bool = False, **kwargs) -> tp.Tuple[dict, ...]:
    """Annotate each part of a reference name with its corresponding object.

    Args:
        refname (str): Fully-qualified dotted name (e.g. "pkg.mod.Class.attr").
        allow_partial (bool): Whether to allow partial resolution of the reference name.
        **kwargs: Keyword arguments for `vectorbtpro.utils.attr_.get_attr`.

    Returns:
        Tuple[dict, ...]: Tuple of dictionaries, each containing:

            * `name`: Reference name part.
            * `obj`: Object corresponding to the reference name part.
    """
    refname_parts = refname.split(".")
    obj = None
    annotated_parts = []
    refname_so_far = None

    for i, name in enumerate(refname_parts):
        if obj is None:
            if i == 0:
                try:
                    obj = import_module(name)
                except ImportError:
                    if allow_partial:
                        obj = None
                    else:
                        raise
        else:
            try:
                obj = get_attr(obj, name, **kwargs)
            except AttributeError:
                try:
                    obj = import_module(refname_so_far + "." + name)
                except ImportError:
                    if refname_so_far.startswith("vectorbtpro.indicators.factory."):
                        from vectorbtpro.indicators.factory import IndicatorFactory

                        if inspect.isfunction(obj) and obj.__name__ in IndicatorFactory.list_builtin_locations():
                            obj = obj(name)
                        else:
                            if allow_partial:
                                obj = None
                            else:
                                raise
                    else:
                        if allow_partial:
                            obj = None
                        else:
                            raise
            else:
                shadow_modname = refname_so_far + "." + name
                try:
                    shadow_mod = import_module(shadow_modname)
                except ImportError:
                    pass
                else:
                    if shadow_mod is not obj:
                        obj = shadow_mod
        annotated_parts.append(dict(name=name, obj=obj))
        if refname_so_far is None:
            refname_so_far = name
        else:
            refname_so_far += "." + name
    return tuple(annotated_parts)


def get_refname_obj(refname: str, raise_error: bool = True, **kwargs) -> tp.Any:
    """Return the object corresponding to the provided reference name.

    Args:
        refname (str): Fully-qualified dotted name (e.g. "pkg.mod.Class.attr").
        raise_error (bool): Whether to raise an error if the object cannot be found.
        **kwargs: Keyword arguments for `annotate_refname_parts`.

    Returns:
        Any: Object obtained by importing modules and accessing attributes.
    """
    refname_parts = annotate_refname_parts(refname, allow_partial=not raise_error, **kwargs)
    if not refname_parts:
        return None
    return refname_parts[-1]["obj"]


def refname_exists(refname: str, static_only: bool = False) -> bool:
    """Check if the provided reference name corresponds to an existing object.

    Args:
        refname (str): Fully-qualified dotted name (e.g. "pkg.mod.Class.attr").
        static_only (bool): If True, only use static attribute retrieval without calling `getattr`.

    Returns:
        bool: True if the object exists, False otherwise.
    """
    if get_refname_obj(refname, raise_error=False, static_only=True) is not None:
        return True
    if not static_only and get_refname_obj(refname, raise_error=False, static_only=False) is not None:
        return True
    return False


def split_refname(
    refname: str,
    module: tp.Optional[tp.ModuleLike] = None,
    raise_error: bool = True,
) -> tp.Tuple[tp.Optional[ModuleType], tp.Optional[str]]:
    """Return the module and qualified name extracted from the given reference name.

    Args:
        refname (str): Fully-qualified dotted name (e.g. "pkg.mod.Class.attr").
        module (Optional[ModuleLike]): Module context for extraction.
        raise_error (bool): Whether to raise an error if the module cannot be found.

    Returns:
        Tuple[Optional[ModuleType], Optional[str]]: Tuple containing the module and qualified name.
    """
    if module is not None:
        module = resolve_module(module)
    refname_parts = refname.split(".")
    if module is None:
        try:
            module = import_module(refname_parts[0])
        except ImportError:
            if raise_error:
                raise
            return None, refname
        refname_parts = refname_parts[1:]
        if len(refname_parts) == 0:
            return module, None
        return split_refname(".".join(refname_parts), module=module)
    else:
        try:
            module = import_module(module.__name__ + "." + refname_parts[0])
        except ImportError:
            return module, ".".join(refname_parts)
        else:
            refname_parts = refname_parts[1:]
            if len(refname_parts) == 0:
                return module, None
            return split_refname(".".join(refname_parts), module=module)


def resolve_refname(
    refname: str,
    module: tp.Optional[tp.ModuleLike] = None,
    silence_warnings: bool = False,
) -> tp.Optional[tp.MaybeList[str]]:
    """Resolve a reference name into its fully qualified form using the provided module context.

    Uses static introspection to avoid executing code.

    !!! note
        This function attempts to resolve the reference name by checking the module context
        and its attributes. It may return multiple reference names if the reference name is ambiguous.

    Args:
        refname (str): Reference name to resolve.

            A reference name may be a fully qualified dotted path ("vectorbtpro.data.base.Data"),
            a library re-export ("vectorbtpro.Data"), a common alias ("vbt.Data"),
            or a simple name ("Data") that uniquely identifies an object.
        module (Optional[ModuleLike]): Module context used in reference resolution.
        silence_warnings (bool): Flag to suppress warning messages.

    Returns:
        Optional[MaybeList[str]]: Reference name(s), or None if resolution fails.
    """

    def _resolve_refname(refname, module=None, _refname_stack=None, _alias_stack=None, _test_no_module=True):
        from vectorbtpro.utils.source import get_import_alias_map, get_defined_names, resolve_name_origin

        if _refname_stack is None:
            _refname_stack = []
        else:
            _refname_stack = list(_refname_stack)
        if _alias_stack is None:
            _alias_stack = []
        else:
            _alias_stack = list(_alias_stack)

        if module is not None:
            module = resolve_module(module)

        def _make_key(module, name):
            return (module.__name__ if module is not None else None, name)

        def _key_to_str(key):
            return f"{key[0] + '.' if key[0] is not None else ''}{key[1]}"

        def _resolve(refname, module=None, _test_no_module=False):
            return _resolve_refname(
                refname,
                module=module,
                _refname_stack=_refname_stack,
                _test_no_module=_test_no_module,
                _alias_stack=_alias_stack,
            )

        if refname == "":
            if module is None:
                return None
            return module.__name__
        refname_parts = refname.split(".")

        if module is None:
            if refname_parts[0] in package_shortcut_config:
                refname_parts[0] = package_shortcut_config[refname_parts[0]]
                module = import_module(refname_parts[0])
                refname_parts = refname_parts[1:]
            else:
                try:
                    module = import_module(refname_parts[0])
                    refname_parts = refname_parts[1:]
                except ImportError:
                    module = import_module("vectorbtpro")
        elif _test_no_module:
            should_test = False
            if refname_parts[0] in package_shortcut_config:
                should_test = True
            else:
                try:
                    module = import_module(refname_parts[0])
                except ImportError:
                    pass
                else:
                    should_test = True
            if should_test:
                resolved_refname = _resolve(refname)
                if resolved_refname is not None:
                    if not isinstance(resolved_refname, list):
                        resolved_refnames = [resolved_refname]
                        made_list = True
                    else:
                        resolved_refnames = resolved_refname
                        made_list = False
                    for r in resolved_refnames:
                        if r != module.__name__ and not r.startswith(module.__name__ + "."):
                            return None
                    if made_list:
                        return resolved_refnames[0]
                    return resolved_refnames

        if len(refname_parts) == 0:
            return module.__name__
        if refname_parts[0] in package_shortcut_config:
            if package_shortcut_config[refname_parts[0]] == module.__name__:
                refname_parts[0] = package_shortcut_config[refname_parts[0]]
        if refname_parts[0] == module.__name__ and refname_parts[0] not in module.__dict__:
            refname_parts = refname_parts[1:]
            if len(refname_parts) == 0:
                return module.__name__

        refname_key = _make_key(module, ".".join(refname_parts))
        if refname_key in _refname_stack:
            call_stack_str = map(_key_to_str, _refname_stack + [refname_key])
            refname_key_str = _key_to_str(refname_key)
            if not silence_warnings:
                warn(f"Cyclic reference detected: {' -> '.join(call_stack_str)}. Using {refname_key_str!r}.")
            return refname_key_str
        _refname_stack.append(refname_key)

        obj = get_attr(module, refname_parts[0], None, static_only=True)
        if obj is not None:
            shadow_modname = module.__name__ + "." + refname_parts[0]
            try:
                shadow_mod = import_module(shadow_modname)
            except ImportError:
                pass
            else:
                if shadow_mod is not obj:
                    module = shadow_mod
                    refname_parts = refname_parts[1:]
                    if not refname_parts:
                        return module.__name__
                    return _resolve(".".join(refname_parts), module=module)

        if len(refname_parts) == 1:
            defined_names = get_defined_names(module, raise_error=False)
            if refname_parts[0] in defined_names:
                if obj is not None:
                    obj_refname = get_obj_refname(obj, allow_type_fallback=False)
                    if obj_refname is not None:
                        obj_root = obj_refname.split(".", 1)[0]
                        module_root = module.__name__.split(".", 1)[0]
                        if obj_root == module_root:
                            return obj_refname
                return f"{module.__name__}.{refname_parts[0]}"

            alias_map = get_import_alias_map(module, raise_error=False)
            if refname_parts[0] in alias_map:
                target = alias_map[refname_parts[0]]
                if target == refname or target == f"{module.__name__}.{refname_parts[0]}":
                    return target
                alias_key = _make_key(module, refname_parts[0])
                if alias_key in _alias_stack:
                    return None
                _alias_stack.append(alias_key)
                return _resolve(target)

            origin_mod = resolve_name_origin(module, refname_parts[0], raise_error=False)
            if origin_mod is not None:
                if obj is not None:
                    obj_refname = get_obj_refname(obj, allow_type_fallback=False)
                    if obj_refname is not None:
                        obj_root = obj_refname.split(".", 1)[0]
                        module_root = origin_mod.split(".", 1)[0]
                        if obj_root == module_root:
                            return obj_refname
                return f"{origin_mod}.{refname_parts[0]}"

        if obj is not None:
            obj_refname = get_obj_refname(obj, allow_type_fallback=False)
            if obj_refname is not None:
                obj_refname_root = obj_refname.split(".", 1)[0]
                module_root = module.__name__.split(".", 1)[0]
                if obj_refname_root == module_root:
                    full_path = module.__name__ + "." + ".".join(refname_parts)
                    if obj_refname == full_path:
                        return obj_refname
                    head_path = module.__name__ + "." + refname_parts[0]
                    if obj_refname != head_path and not obj_refname.startswith(head_path + "."):
                        if len(refname_parts) == 1:
                            return obj_refname
                        tail = ".".join(refname_parts[1:])
                        candidate = obj_refname + "." + tail
                        if not refname_exists(candidate, static_only=True):
                            return None
                        return _resolve(candidate)

            if inspect.ismodule(obj):
                parent_module = ".".join(obj.__name__.split(".")[:-1])
            else:
                parent_mod = get_module(obj)
                parent_module = None
                if parent_mod is not None:
                    if refname_parts[0] in parent_mod.__dict__:
                        if parent_mod.__dict__[refname_parts[0]] is obj:
                            parent_module = parent_mod.__name__

            if parent_module is None or parent_module == module.__name__:
                if inspect.ismodule(obj):
                    module = obj
                    refname_parts = refname_parts[1:]
                    if not refname_parts:
                        return module.__name__
                    return _resolve(".".join(refname_parts), module=module)
                name = get_attr(obj, "__name__", None)
                if name is not None and isinstance(name, str) and name in module.__dict__:
                    obj = module.__dict__[name]
                    refname_parts[0] = name
                if len(refname_parts) == 1:
                    return module.__name__ + "." + refname_parts[0]
                if not isinstance(obj, type):
                    cls = type(obj)
                else:
                    cls = obj
                k = refname_parts[1]
                owner = None
                for super_cls in inspect.getmro(cls):
                    d = get_attr(super_cls, "__dict__", {})
                    if k in d:
                        owner = super_cls
                        break
                if owner is None:
                    for super_cls in inspect.getmro(cls):
                        attrs_attrs = getattr(super_cls, "__attrs_attrs__", None)
                        if not attrs_attrs:
                            continue
                        for a in attrs_attrs:
                            if getattr(a, "name", None) == k:
                                owner = super_cls
                                break
                        if owner is not None:
                            break
                if owner is None:
                    return None
                owner_module = get_attr(owner, "__module__", None)
                owner_name = get_attr(owner, "__name__", None)
                if owner_module is None or owner_name is None:
                    return None
                if not isinstance(owner_module, str) or not isinstance(owner_name, str):
                    return None
                cls_path = owner_module + "." + owner_name
                tail = ".".join(refname_parts[1:])
                candidate = cls_path + "." + tail
                if not refname_exists(candidate, static_only=True):
                    return None
                return candidate

            if inspect.ismodule(obj):
                parent_module = obj
                refname_parts = refname_parts[1:]
            return _resolve(".".join(refname_parts), module=parent_module)

        if len(refname_parts) > 1:
            alias_map = get_import_alias_map(module, raise_error=False)
            if refname_parts[0] in alias_map:
                target = alias_map[refname_parts[0]]
                tail = ".".join(refname_parts[1:])
                new_refname = f"{target}.{tail}"
                if new_refname == refname:
                    return new_refname
                alias_key = _make_key(module, refname_parts[0])
                if alias_key in _alias_stack:
                    return None
                _alias_stack.append(alias_key)
                return _resolve(new_refname)

        refnames = []
        visited_modules = set()
        for k, v in list(module.__dict__.items()):
            if v is not module:
                if inspect.ismodule(v) and v.__name__.startswith(module.__name__) and v.__name__ not in visited_modules:
                    visited_modules.add(v.__name__)
                    refname = _resolve(".".join(refname_parts), module=v)
                    if refname is not None:
                        if isinstance(refname, str):
                            refname = [refname]
                        for r in refname:
                            if r not in refnames:
                                refnames.append(r)
        if len(refnames) > 1:
            pairs = [(r, get_refname_obj(r, raise_error=False, static_only=True)) for r in refnames]
            pairs = [(r, o) for (r, o) in pairs if o is not None]
            if not pairs:
                return refnames
            ids = {id(o) for _, o in pairs}
            if len(ids) > 1:
                return refnames
            obj = pairs[0][1]
            obj_refname = get_obj_refname(obj, allow_type_fallback=False)
            if obj_refname is not None:
                obj_refname_root = obj_refname.split(".", 1)[0]
                candidate_roots = {r.split(".", 1)[0] for (r, _) in pairs}
                if obj_refname_root in candidate_roots:
                    return obj_refname
            return refnames
        if len(refnames) == 1:
            return refnames[0]
        return None

    if module is not None:
        module = resolve_module(module)

    if not hasattr(resolve_refname, "_cache"):
        resolve_refname._cache = {}
    cache = resolve_refname._cache
    cache_key = (module.__name__ if module is not None else None, refname)
    if cache_key in cache:
        return cache[cache_key]

    resolved = _resolve_refname(refname, module=module)

    cache[cache_key] = resolved
    return cache[cache_key]


def get_refname(
    obj: tp.Any,
    module: tp.Optional[tp.ModuleLike] = None,
    resolve: bool = True,
    can_be_refname: bool = True,
    allow_type_fallback: bool = True,
    **kwargs,
) -> tp.Optional[tp.MaybeList[str]]:
    """Return the reference name(s) for the provided object.

    Args:
        obj (Any): Object from which to extract the reference name.

            If a string or tuple is provided, it is treated as a reference name.
        module (Optional[ModuleLike]): Module context used in reference resolution.
        resolve (bool): Whether to resolve the reference to an actual object.
        can_be_refname (bool): Whether the provided object can be a reference name itself.
        allow_type_fallback (bool): Whether to fallback to the type's reference name
            if no direct reference name is found.
        **kwargs: Keyword arguments for `resolve_refname`.

    Returns:
        Optional[MaybeList[str]]: Reference name as a string, a list of strings if multiple
            reference names are found, or None.
    """
    if can_be_refname and type(obj) is tuple:
        if len(obj) == 1:
            obj = obj[0]
        else:
            first_refname = get_obj_refname(obj[0], allow_type_fallback=allow_type_fallback)
            if first_refname is None:
                return None
            obj = first_refname + "." + ".".join(obj[1:])
    if can_be_refname and isinstance(obj, str):
        refname = obj
        if resolve:
            return resolve_refname(refname, module=module, **kwargs)
        return refname
    return get_obj_refname(obj, allow_type_fallback=allow_type_fallback)


def get_obj(
    obj: tp.Any,
    module: tp.Optional[tp.ModuleLike] = None,
    resolve: bool = True,
    allow_multiple: bool = False,
    raise_error: bool = True,
    **kwargs,
) -> tp.Optional[tp.MaybeList]:
    """Return the object by its reference name.

    Args:
        obj (Any): Object from which to extract the reference name.

            If a string or tuple is provided, it is treated as a reference name.
        module (Optional[ModuleLike]): Module context used in reference resolution.
        resolve (bool): Whether to resolve the reference to an actual object.
        allow_multiple (bool): Whether to allow returning multiple objects
            if more than one reference name is found.
        raise_error (bool): Whether to raise an error if the object cannot be found.
        **kwargs: Keyword arguments for `get_refname`.

    Returns:
        Optional[MaybeList]: Object or a list of objects if multiple reference names are found, or None.
    """
    refname = get_refname(obj, module=module, resolve=resolve, **kwargs)
    if refname is None:
        return None
    if isinstance(refname, list):
        obj = None
        for _refname in refname:
            _obj = get_refname_obj(_refname, raise_error=raise_error)
            if obj is None:
                obj = _obj
            elif not isinstance(obj, list):
                if _obj is not obj:
                    if not allow_multiple:
                        reflist = "\n* ".join(refname)
                        raise AmbiguousReferenceError(f"Multiple reference names found:\n\n* {reflist}")
                    obj = [obj, _obj]
            else:
                if _obj not in obj:
                    obj.append(_obj)
        return obj
    return get_refname_obj(refname, raise_error=raise_error)


def ensure_refname(
    obj: tp.Any,
    module: tp.Optional[tp.ModuleLike] = None,
    resolve: bool = True,
    vbt_only: bool = False,
    return_parts: bool = False,
    raise_error: bool = True,
    **kwargs,
) -> tp.Union[tp.Optional[str], tp.Tuple[tp.Optional[str], tp.Optional[ModuleType], tp.Optional[str]]]:
    """Return the reference name for an object and optionally its module and qualified name.

    Args:
        obj (Any): Object from which to extract the reference name.

            If a string or tuple is provided, it is treated as a reference name.
        module (Optional[ModuleLike]): Module context used in reference resolution.
        resolve (bool): Whether to resolve the reference to an actual object.
        vbt_only (bool): If True, limit resolution to objects within vectorbtpro.
        return_parts (bool): If True, return a tuple containing the reference name, module, and qualified name.
        raise_error (bool): Whether to raise an error if the reference name cannot be determined.
        **kwargs: Keyword arguments for `get_refname`.

    Returns:
        Union[Optional[str], Tuple[Optional[str], Optional[ModuleType], Optional[str]]]:
            Reference name as a string, or a tuple of (reference name, module, qualified name)
            if `return_parts` is True; or None if the reference name cannot be determined.
    """

    def _raise_error():
        raise ReferenceNotFoundError(
            f"Couldn't find the reference name for {obj!r}, or the object is external. "
            "If the object is internal, please decompose the object or provide a string instead."
        )

    refname = get_refname(obj, module=module, resolve=resolve, **kwargs)
    if refname is None:
        if raise_error:
            _raise_error()
        if return_parts:
            return None, None, None
        return None
    if isinstance(refname, list):
        if raise_error:
            reflist = "\n* ".join(refname)
            raise AmbiguousReferenceError(f"Multiple reference names found for {obj!r}:\n\n* {reflist}")
        if return_parts:
            return None, None, None
        return None
    if vbt_only or return_parts or resolve:
        module, qualname = split_refname(refname, raise_error=False)
        if module is not None:
            if vbt_only and module.__name__.split(".")[0] != "vectorbtpro":
                if raise_error:
                    _raise_error()
                if return_parts:
                    return None, None, None
                return None
            if return_parts:
                return refname, module, qualname
            if resolve:
                if qualname is None:
                    return module.__name__
                return module.__name__ + "." + qualname
        else:
            if vbt_only and refname.split(".")[0] != "vectorbtpro":
                if raise_error:
                    _raise_error()
                if return_parts:
                    return None, None, None
                return None
            if return_parts:
                return refname, module, qualname
    return refname


def get_imlucky_url(query: str) -> str:
    """Construct a DuckDuckGo "I'm lucky" URL for a query.

    Args:
        query (str): Search query.

    Returns:
        str: DuckDuckGo "I'm lucky" URL based on the query.
    """
    return "https://duckduckgo.com/?q=!ducky+" + urllib.request.pathname2url(query)


def imlucky(query: str) -> bool:
    """Open a DuckDuckGo "I'm lucky" URL for a query in the web browser.

    Args:
        query (str): Search query.

    Returns:
        bool: True if the browser was opened successfully, False otherwise.
    """
    return webbrowser.open(get_imlucky_url(query))


def get_api_ref(obj: tp.Any, module: tp.Optional[tp.ModuleLike] = None, resolve: bool = True, **kwargs) -> str:
    """Return the API reference URL for an object.

    Args:
        obj (Any): Object from which to extract the reference name.

            If a string or tuple is provided, it is treated as a reference name.
        module (Optional[ModuleLike]): Module context used in reference resolution.
        resolve (bool): Whether to resolve the reference to an actual object.
        **kwargs: Keyword arguments for `ensure_refname`.

    Returns:
        str: API reference URL for the given object.
    """
    refname, module, qualname = ensure_refname(obj, module=module, resolve=resolve, return_parts=True, **kwargs)
    if module.__name__.split(".")[0] == "vectorbtpro":
        api_url = "https://github.com/polakowo/vectorbt.pro/blob/pvt-links/api/"
        md_url = api_url + module.__name__ + ".md/"
        if qualname is None:
            return md_url + "#" + module.__name__.replace(".", "")
        return md_url + "#" + module.__name__.replace(".", "") + qualname.replace(".", "")
    if resolve:
        if qualname is None:
            search_query = module.__name__
        else:
            search_query = module.__name__ + "." + qualname
    else:
        search_query = refname
    return get_imlucky_url(search_query)


def open_api_ref(obj: tp.Any, module: tp.Optional[tp.ModuleLike] = None, resolve: bool = True, **kwargs) -> bool:
    """Open the API reference URL for an object in the web browser.

    Args:
        obj (Any): Object from which to extract the reference name.

            If a string or tuple is provided, it is treated as a reference name.
        module (Optional[ModuleLike]): Module context used in reference resolution.
        resolve (bool): Whether to resolve the reference to an actual object.
        **kwargs: Keyword arguments for `get_api_ref`.

    Returns:
        bool: True if the browser was opened successfully, False otherwise.
    """
    return webbrowser.open(get_api_ref(obj, module=module, resolve=resolve, **kwargs))


@define
class RefInfo(DefineMixin):
    """Class representing information about a reference."""

    refname: str = define.field()
    """Fully qualified reference name."""

    qualname: tp.Optional[str] = define.field(default=None)
    """Qualified name of the referenced object within its module."""

    module: tp.Optional[str] = define.field(default=None)
    """Module where the referenced object is defined."""

    type: tp.Optional[str] = define.field(default=None)
    """Type of the referenced object."""

    kind: tp.Optional[str] = define.field(default=None)
    """Kind of the referenced object."""

    container: tp.Optional[str] = define.field(default=None)
    """Reference name of the container."""

    direct_members: tp.List[str] = define.field(factory=list)
    """List of reference names of the direct members."""

    nested_members: tp.List[str] = define.field(factory=list)
    """List of reference names of the nested members."""

    direct_bases: tp.List[str] = define.field(factory=list)
    """List of reference names of the direct base classes."""

    nested_bases: tp.List[str] = define.field(factory=list)
    """List of reference names of the nested base classes."""

    direct_dependencies: tp.List[str] = define.field(factory=list)
    """List of reference names of the direct dependencies."""

    nested_dependencies: tp.List[str] = define.field(factory=list)
    """List of reference names of the nested dependencies."""

    is_shallow: bool = define.field(default=False)
    """Whether relations are not included."""


RefGraphT = tp.TypeVar("RefGraphT", bound="RefGraph")


class RefGraph(Configured):
    """Class representing a reference graph.

    Args:
        G (DiGraph): NetworkX directed graph representing the reference relationships.
        root (Optional[str]): Fully qualified reference name of the root node.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.
    """

    def __init__(self, G: DiGraphT, root: tp.Optional[str] = None, **kwargs) -> None:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("networkx")

        Configured.__init__(self, G=G, root=root, **kwargs)

        self._G = G
        self._root = root

    @property
    def G(self) -> DiGraphT:
        """NetworkX directed graph representing the reference relationships.

        Returns:
            DiGraph: NetworkX directed graph.
        """
        return self._G

    @property
    def root(self) -> tp.Optional[str]:
        """Fully qualified reference name of the root node.

        Returns:
            Optional[str]: Root node reference name, or None if not specified.
        """
        return self._root

    @property
    def is_multigraph(self) -> bool:
        """Check if the graph is a MultiGraph.

        Returns:
            bool: True if the graph is a MultiGraph, False otherwise.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("networkx")
        import networkx as nx

        return isinstance(self.G, nx.MultiGraph)

    @classmethod
    def from_ref_infos(
        cls: tp.Type[RefGraphT],
        ref_infos: tp.List[RefInfo],
        merge_edges: bool = True,
        **kwargs,
    ) -> RefGraphT:
        """Build a NetworkX directed graph from the provided reference information.

        Args:
            ref_infos (List[RefInfo]): List of `RefInfo` instances.
            merge_edges (bool): If True, merge multiple edges between the same nodes
                into a single edge with a set of kinds.
            **kwargs: Keyword arguments for `RefGraph`.

        Returns:
            RefGraph: Reference graph built from the reference information.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("networkx")
        import networkx as nx

        G = nx.MultiDiGraph()

        for ref_info in ref_infos:
            if ref_info.refname not in G:
                G.add_node(ref_info.refname)
            G.nodes[ref_info.refname].update(
                obj_qualname=ref_info.qualname,
                obj_module=ref_info.module,
                obj_type=ref_info.type,
                obj_kind=ref_info.kind,
            )

        for ref_info in ref_infos:
            if ref_info.container is not None and ref_info.container in G:
                G.add_edge(ref_info.container, ref_info.refname, rel_kind="container")
            for m in ref_info.direct_members:
                if m in G:
                    G.add_edge(ref_info.refname, m, rel_kind="direct_member")
            for m in ref_info.nested_members:
                if m in G:
                    G.add_edge(ref_info.refname, m, rel_kind="nested_member")
            for b in ref_info.direct_bases:
                if b in G:
                    G.add_edge(ref_info.refname, b, rel_kind="direct_base")
            for b in ref_info.nested_bases:
                if b in G:
                    G.add_edge(ref_info.refname, b, rel_kind="nested_base")
            for d in ref_info.direct_dependencies:
                if d in G:
                    G.add_edge(ref_info.refname, d, rel_kind="direct_dependency")
            for d in ref_info.nested_dependencies:
                if d in G:
                    G.add_edge(ref_info.refname, d, rel_kind="nested_dependency")

        return cls(G=G, **kwargs).merge_edges() if merge_edges else cls(G=G, **kwargs)

    def filter_nodes(self: RefGraphT, predicate: tp.Callable[[str, tp.Dict[str, tp.Any]], bool]) -> RefGraphT:
        """Filter nodes in the graph based on a predicate function.

        Args:
            predicate (Callable[[str, Dict[str, Any]], bool]): Function that takes a node ID
                and a data dictionary, and returns True if the node should be kept, False otherwise.

        Returns:
            RefGraph: Reference graph with filtered nodes.
        """
        new_G = type(self.G)()
        for n, d in self.G.nodes(data=True):
            if predicate(n, d):
                new_G.add_node(n, **d)
        if self.is_multigraph:
            for u, v, d in self.G.edges(data=True):
                if u in new_G and v in new_G:
                    new_G.add_edge(u, v, **d)
        else:
            for u, v, d in self.G.edges(data=True):
                if u in new_G and v in new_G:
                    new_G.add_edge(u, v, **d)
        return self.replace(G=new_G)

    def filter_edges(self: RefGraphT, predicate: tp.Callable[[str, str, tp.Dict[str, tp.Any]], bool]) -> RefGraphT:
        """Filter edges in the graph based on a predicate function.

        Args:
            predicate (Callable[[str, str, Dict[str, Any]], bool]): Function that takes a source node ID,
                a target node ID, and a data dictionary, and returns True if the edge should be kept,
                False otherwise.

        Returns:
            RefGraph: Reference graph with filtered edges.
        """
        new_G = type(self.G)()
        for n, d in self.G.nodes(data=True):
            new_G.add_node(n, **d)
        if self.is_multigraph:
            for u, v, d in self.G.edges(data=True):
                if predicate(u, v, d):
                    new_G.add_edge(u, v, **d)
        else:
            for u, v, d in self.G.edges(data=True):
                if predicate(u, v, d):
                    new_G.add_edge(u, v, **d)
        return self.replace(G=new_G)

    def merge_edges(self: RefGraphT) -> RefGraphT:
        """Merge multiple edges with a single relationship kind into a single edge with multiple relationship kinds.

        Returns:
            RefGraph: Reference graph with merged edges.
        """
        import networkx as nx

        if not self.is_multigraph:
            return self
        new_G = nx.DiGraph()
        for n, d in self.G.nodes(data=True):
            new_G.add_node(n, **d)
        for u, v, d in self.G.edges(data=True):
            if new_G.has_edge(u, v):
                new_G[u][v]["rel_kinds"].add(d.get("rel_kind", None))
            else:
                new_d = dict(d)
                rel_kinds = {new_d.pop("rel_kind")}
                new_G.add_edge(u, v, rel_kinds=rel_kinds, **new_d)
        return self.replace(G=new_G)

    def split_edges(self: RefGraphT) -> RefGraphT:
        """Split each edge with multiple relationship kinds into multiple edges with a single relationship kind.

        Returns:
            RefGraph: Reference graph with split edges.
        """
        import networkx as nx

        if self.is_multigraph:
            return self
        new_G = nx.MultiDiGraph()
        for n, d in self.G.nodes(data=True):
            new_G.add_node(n, **d)
        for u, v, d in self.G.edges(data=True):
            new_d = dict(d)
            rel_kinds = new_d.pop("rel_kinds", set())
            for rel_kind in rel_kinds:
                new_G.add_edge(u, v, rel_kind=rel_kind, **new_d)
        return self.replace(G=new_G)

    def get_container(self, *args, **kwargs) -> tp.Optional[str]:
        """Get the container of the specified member from the graph.

        Args:
            *args: Positional arguments for `ensure_refname`.
            **kwargs: Keyword arguments for `ensure_refname`.

        Returns:
            Optional[str]: Reference name of the container, or None if not found.
        """
        refname = ensure_refname(*args, **kwargs)
        if self.is_multigraph:
            self = self.merge_edges()

        for u, _, d in self.G.in_edges(refname, data=True):
            if "container" in d.get("rel_kinds", set()):
                return u
        return None

    def get_contents(self, *args, **kwargs) -> tp.List[str]:
        """Get all contents of the specified container from the graph.

        Args:
            *args: Positional arguments for `ensure_refname`.
            **kwargs: Keyword arguments for `ensure_refname`.

        Returns:
            List[str]: List of reference names of contents.
        """
        refname = ensure_refname(*args, **kwargs)
        if self.is_multigraph:
            self = self.merge_edges()

        contents = set()
        for _, v, d in self.G.out_edges(refname, data=True):
            if "container" in d.get("rel_kinds", set()):
                contents.add(v)
        return sorted(contents)

    def get_members(self, *args, relation: str = "all", **kwargs) -> tp.List[str]:
        """Get all members of the specified container from the graph.

        Args:
            *args: Positional arguments for `ensure_refname`.
            relation (str): Relation between references. One of:

                * "direct": References that are connected directly.
                * "nested": References that are connected through other references.
                * "all": Both direct and nested references.
            **kwargs: Keyword arguments for `ensure_refname`.

        Returns:
            List[str]: List of reference names of members.
        """
        refname = ensure_refname(*args, **kwargs)
        if self.is_multigraph:
            self = self.merge_edges()

        members = set()
        for _, v, d in self.G.out_edges(refname, data=True):
            if relation.lower() == "direct":
                if "direct_member" in d.get("rel_kinds", set()):
                    members.add(v)
            elif relation.lower() == "nested":
                if "nested_member" in d.get("rel_kinds", set()):
                    members.add(v)
            elif relation.lower() == "all":
                if "direct_member" in d.get("rel_kinds", set()) or "nested_member" in d.get("rel_kinds", set()):
                    members.add(v)
            else:
                raise ValueError(f"Invalid relation: {relation!r}")
        return sorted(members)

    def get_bases(self, *args, relation: str = "all", **kwargs) -> tp.List[str]:
        """Get all base classes of the specified derived class from the graph.

        Args:
            *args: Positional arguments for `ensure_refname`.
            relation (str): Relation between references. One of:

                * "direct": References that are connected directly.
                * "nested": References that are connected through other references.
                * "all": Both direct and nested references.
            **kwargs: Keyword arguments for `ensure_refname`.

        Returns:
            List[str]: List of reference names of base classes.
        """
        refname = ensure_refname(*args, **kwargs)
        if self.is_multigraph:
            self = self.merge_edges()

        bases = set()
        for _, v, d in self.G.out_edges(refname, data=True):
            if relation.lower() == "direct":
                if "direct_base" in d.get("rel_kinds", set()):
                    bases.add(v)
            elif relation.lower() == "nested":
                if "nested_base" in d.get("rel_kinds", set()):
                    bases.add(v)
            elif relation.lower() == "all":
                if "direct_base" in d.get("rel_kinds", set()) or "nested_base" in d.get("rel_kinds", set()):
                    bases.add(v)
            else:
                raise ValueError(f"Invalid relation: {relation!r}")
        return sorted(bases)

    def get_derived(self, *args, relation: str = "all", **kwargs) -> tp.List[str]:
        """Get all derived classes of the specified base class from the graph.

        Args:
            *args: Positional arguments for `ensure_refname`.
            relation (str): Relation between references. One of:

                * "direct": References that are connected directly.
                * "nested": References that are connected through other references.
                * "all": Both direct and nested references.
            **kwargs: Keyword arguments for `ensure_refname`.

        Returns:
            List[str]: List of reference names of derived classes.
        """
        refname = ensure_refname(*args, **kwargs)
        if self.is_multigraph:
            self = self.merge_edges()

        derived = set()
        for u, _, d in self.G.in_edges(refname, data=True):
            if relation.lower() == "direct":
                if "direct_base" in d.get("rel_kinds", set()):
                    derived.add(u)
            elif relation.lower() == "nested":
                if "nested_base" in d.get("rel_kinds", set()):
                    derived.add(u)
            elif relation.lower() == "all":
                if "direct_base" in d.get("rel_kinds", set()) or "nested_base" in d.get("rel_kinds", set()):
                    derived.add(u)
            else:
                raise ValueError(f"Invalid relation: {relation!r}")
        return sorted(derived)

    def get_dependencies(self, *args, relation: str = "all", **kwargs) -> tp.List[str]:
        """Get all dependencies of the specified reference name from the graph.

        Args:
            *args: Positional arguments for `ensure_refname`.
            relation (str): Relation between references. One of:

                * "direct": References that are connected directly.
                * "nested": References that are connected through other references.
                * "all": Both direct and nested references.
            **kwargs: Keyword arguments for `ensure_refname`.

        Returns:
            List[str]: List of reference names of dependencies.
        """
        refname = ensure_refname(*args, **kwargs)
        if self.is_multigraph:
            self = self.merge_edges()

        dependencies = set()
        for _, v, d in self.G.out_edges(refname, data=True):
            if relation.lower() == "direct":
                if "direct_dependency" in d.get("rel_kinds", set()):
                    dependencies.add(v)
            elif relation.lower() == "nested":
                if "nested_dependency" in d.get("rel_kinds", set()):
                    dependencies.add(v)
            elif relation.lower() == "all":
                if "direct_dependency" in d.get("rel_kinds", set()) or "nested_dependency" in d.get("rel_kinds", set()):
                    dependencies.add(v)
            else:
                raise ValueError(f"Invalid relation: {relation!r}")
        return sorted(dependencies)

    def get_dependents(self, *args, relation: str = "all", **kwargs) -> tp.List[str]:
        """Get all dependents of the specified reference name from the graph.

        Args:
            *args: Positional arguments for `ensure_refname`.
            relation (str): Relation between references. One of:

                * "direct": References that are connected directly.
                * "nested": References that are connected through other references.
                * "all": Both direct and nested references.
            **kwargs: Keyword arguments for `ensure_refname`.

        Returns:
            List[str]: List of reference names of dependents.
        """
        refname = ensure_refname(*args, **kwargs)
        if self.is_multigraph:
            self = self.merge_edges()

        dependents = set()
        for u, _, d in self.G.in_edges(refname, data=True):
            if relation.lower() == "direct":
                if "direct_dependency" in d.get("rel_kinds", set()):
                    dependents.add(u)
            elif relation.lower() == "nested":
                if "nested_dependency" in d.get("rel_kinds", set()):
                    dependents.add(u)
            elif relation.lower() == "all":
                if "direct_dependency" in d.get("rel_kinds", set()) or "nested_dependency" in d.get("rel_kinds", set()):
                    dependents.add(u)
            else:
                raise ValueError(f"Invalid relation: {relation!r}")
        return sorted(dependents)

    @classmethod
    def get_roots(cls, G: DiGraphT) -> tp.Set[tp.Hashable]:
        """Get root nodes in the graph.

        Args:
            G (DiGraph): NetworkX directed graph.

        Returns:
            Set[Hashable]: Set of root node IDs.
        """
        import networkx as nx

        if G.number_of_nodes() == 0:
            return set()
        cG = nx.condensation(G)
        roots = set()
        for comp in cG.nodes():
            if cG.in_degree(comp) == 0:
                roots.update(cG.nodes[comp]["members"])
        return roots

    @classmethod
    def radial_layout(
        cls,
        G: DiGraphT,
        root: tp.Optional[str] = None,
        layer_spacing: tp.Union[float, tp.Sequence[float]] = 1.0,
        ring_spacing: tp.Union[float, tp.Sequence[float]] = 0.15,
        distribute: tp.Union[bool, tp.Sequence[bool]] = False,
        start_angle: float = 0.0,
        clockwise: bool = True,
    ) -> tp.Dict[str, tp.Tuple[float, float]]:
        """Compute a radial layout for the graph.

        Args:
            G (DiGraph): NetworkX directed graph.
            root (Optional[str]): Root node to start the layout from.

                If None, the first root node in sorted order is chosen.
            layer_spacing (Union[float, Sequence[float]]): Spacing between layers (depths).

                If a sequence is provided, it specifies spacing for each layer.
            ring_spacing (Union[float, Sequence[float]]): Spacing between rings for different object kinds.

                If a sequence is provided, it specifies spacing for each layer.
            distribute (Union[bool, Sequence[bool]]): Whether to distribute nodes evenly at each layer.

                If a sequence is provided, it specifies the flag for each layer.
            start_angle (float): Starting angle in degrees for the first leaf node.

                0 degrees is at the top (12 o'clock position). Depends on the `clockwise` parameter.
            clockwise (bool): Whether to arrange nodes in a clockwise direction.

        Returns:
            Dict[str, Tuple[float, float]]: Dictionary mapping node IDs to (x, y) coordinates.
        """
        import networkx as nx

        if root is None:
            roots = cls.get_roots(G)
            if len(roots) == 1:
                root = next(iter(roots))
            else:
                G = G.copy()
                root = "__root__"
                G.add_node(root)
                for n in roots or G.nodes:
                    if n != root:
                        G.add_edge(root, n)
        if root not in G:
            raise ValueError(f"Root {root!r} is not in the graph")

        if isinstance(G, nx.DiGraph):
            G_tree = G.to_undirected()
        else:
            G_tree = G

        T = nx.dfs_tree(G_tree, root)
        if T.number_of_nodes() == 1:
            return {root: (0.0, 0.0)}

        leaves = [n for n in T.nodes if T.out_degree(n) == 0]
        if not leaves:
            return {root: (0.0, 0.0)}

        n_leaves = len(leaves)
        direction = -1.0 if clockwise else 1.0
        start_angle_rad = math.radians(start_angle)
        base_start = math.pi / 2.0 + direction * start_angle_rad
        leaf_step = 2 * math.pi / n_leaves

        angles = {}
        depths = {}
        parents = {}
        next_leaf_idx = 0

        def dfs(node, depth, parent):
            nonlocal next_leaf_idx
            depths[node] = depth
            parents[node] = parent
            children = list(T.successors(node))
            try:
                children.sort(key=str)
            except Exception:
                pass
            if not children:
                angles[node] = base_start + direction * next_leaf_idx * leaf_step
                next_leaf_idx += 1
            else:
                for child in children:
                    dfs(child, depth + 1, node)
                child_angles = [angles[ch] for ch in children]
                angles[node] = sum(child_angles) / len(child_angles)

        dfs(root, 0, None)
        max_depth = max(depths.values())

        if isinstance(layer_spacing, (int, float)):
            layer_spacings = [float(layer_spacing)] * max_depth
        else:
            layer_spacings = []
            for d in range(max_depth):
                if d < len(layer_spacing):
                    layer_spacings.append(float(layer_spacing[d]))
                else:
                    layer_spacings.append(float(layer_spacing[-1]))

        if isinstance(ring_spacing, (int, float)):
            ring_spacings = [float(ring_spacing)] * max_depth
        else:
            ring_spacings = []
            for d in range(max_depth):
                if d < len(ring_spacing):
                    ring_spacings.append(float(ring_spacing[d]))
                else:
                    ring_spacings.append(float(ring_spacing[-1]))

        if isinstance(distribute, bool):
            distribute_flags = [distribute] * max_depth
        else:
            distribute_flags = []
            for d in range(max_depth):
                if d < len(distribute):
                    distribute_flags.append(bool(distribute[d]))
                else:
                    distribute_flags.append(bool(distribute[-1]))

        depth_nodes = {d: [] for d in range(max_depth + 1)}
        for n, d in depths.items():
            depth_nodes[d].append(n)

        depth_radius = {0: 0.0}
        acc = 0.0
        for d in range(1, max_depth + 1):
            acc += layer_spacings[d - 1]
            depth_radius[d] = acc

        kind_to_index = {
            "module": 0,
            "class": 1,
            "callable": 2,
            "data": 3,
            "unknown": 4,
        }

        r = {}
        for n, d in depths.items():
            if d == 0:
                r[n] = 0.0
                continue
            base_r = depth_radius[d]
            rs = ring_spacings[d - 1]
            raw_kind = G.nodes[n].get("obj_kind", None)
            kind = raw_kind if raw_kind is not None and raw_kind in kind_to_index else "unknown"
            idx = kind_to_index[kind]
            offset = (idx - 2.0) * rs
            r[n] = base_r + offset

        for d in range(1, max_depth + 1):
            if not distribute_flags[d - 1]:
                continue
            nodes_d = depth_nodes[d]
            if len(nodes_d) <= 1:
                continue
            nodes_d = sorted(nodes_d, key=str)
            k = len(nodes_d)
            if d == 1:
                base = base_start
            else:
                base = 0.0
            step = direction * 2 * math.pi / k
            for i, n in enumerate(nodes_d):
                angles[n] = base + i * step

        for d in range(2, max_depth + 1):
            nodes_d = depth_nodes[d]
            if not nodes_d:
                continue
            C = 0.0
            S = 0.0
            has_weight = False
            for n in nodes_d:
                p = parents[n]
                if p is None:
                    continue
                w = r[n] * r.get(p, 0.0)
                if w == 0.0:
                    continue
                has_weight = True
                delta = angles[n] - angles[p]
                C += w * math.cos(delta)
                S += w * math.sin(delta)
            if not has_weight:
                continue
            phi = -math.atan2(S, C)
            for n in nodes_d:
                angles[n] += phi

        pos = {}
        for n in G.nodes:
            if n in depths:
                a = angles[n]
                rr = r.get(n, 0.0)
                pos[n] = (rr * math.cos(a), rr * math.sin(a))
            else:
                pos[n] = (0.0, 0.0)

        return pos

    def generate_node_positions(
        self,
        layout: tp.Union[None, str, tp.Callable] = None,
        root: tp.Optional[str] = None,
        add_root: bool = False,
        add_sibling_edges: bool = False,
        **kwargs,
    ) -> tp.Dict[str, tp.Tuple[float, float]]:
        """Generate positions for the nodes in the graph.

        Args:
            layout (Union[None, str, Callable]): Layout algorithm to use.

                If a string, will be first searched as `RefGraph.<layout>`; if not found,
                it will be searched as `networkx.layout.<layout>_layout`; if still not found,
                it will be used as `prog` in `graphviz_layout` if `graphviz` is installed.
                If a callable, it will be called with the graph and `**kwargs`.
                If None, defaults to `RefGraph.radial_layout`.
            root (Optional[str]): Root node for the layout computation.
            add_root (bool): If True, add a root node to the layout computation.

                Added only if there are multiple root nodes in the graph.
            add_sibling_edges (bool): If True, add sibling edges to the layout computation.
            **kwargs: Keyword arguments for layout computation.

        Returns:
            Dict[str, Tuple[float, float]]: Dictionary mapping node IDs to their (x, y) positions.
        """
        import networkx as nx

        if self.is_multigraph:
            self = self.merge_edges()
        H = self.filter_edges(lambda u, v, d: "container" in d.get("rel_kinds", set())).G
        roots = self.get_roots(H)

        if add_root and len(roots) != 1:
            root = "__root__"
            H.add_node(root)
            for n in roots or H.nodes:
                if n != root:
                    H.add_edge(root, n)

        if add_sibling_edges:
            parent_to_children = {p: list(H.successors(p)) for p in H.nodes()}
            for children in parent_to_children.values():
                if len(children) < 2:
                    continue
                for i in range(len(children)):
                    for j in range(i + 1, len(children)):
                        u, v = children[i], children[j]
                        if not H.has_edge(u, v):
                            H.add_edge(u, v)
                        if not H.has_edge(v, u):
                            H.add_edge(v, u)

        if layout is None:
            return self.radial_layout(H, root=root, **kwargs)
        if isinstance(layout, str):
            layout_func = getattr(nx.layout, f"{layout}_layout", None)
            if layout_func is not None:
                return layout_func(H, **kwargs)
            try:
                from networkx.drawing.nx_agraph import graphviz_layout

                if root is None:
                    if "__root__" in H.nodes:
                        root = "__root__"
                    else:
                        root = self.root
                return graphviz_layout(H, prog=layout, root=root, **kwargs)
            except ImportError:
                raise ValueError(f"Layout {layout!r} not found in networkx and graphviz is not installed")
        else:
            return layout(H, **kwargs)

    def generate_node_colors(
        self,
        partition: bool = False,
        alphabetical: bool = True,
        max_depth: tp.Optional[int] = None,
        cmap: tp.Any = "rainbow",
        predicate: tp.Optional[tp.Callable[[str, tp.Dict[str, tp.Any]], bool]] = None,
    ) -> tp.Dict[str, str]:
        """Generate colors for the nodes in the graph.

        Args:
            partition (bool): If True, assign colors hierarchically by recursively
                splitting intervals; otherwise, map nodes by their position in the global order.
            alphabetical (bool): If True, order nodes alphabetically; otherwise, use a hash-based order.
            max_depth (Optional[int]): Limit the effective reference name depth; nodes deeper than this
                inherit the color of their ancestor at that depth.
            cmap (Any): Colormap identifier provided as a string name or a collection (list/tuple) of colors.
            predicate (Optional[Callable[[str, Dict[str, Any]], bool]]):
                If provided, only nodes for which this predicate returns True
                participate in color distribution; all other nodes inherit the
                color of the closest matching ancestor in their dotted name.

        Returns:
            Dict[str, str]: Dictionary mapping node IDs to their colors.
        """
        from vectorbtpro.utils.colors import map_value_to_cmap

        graph_nodes = list(self.G.nodes())
        if not graph_nodes:
            return {}
        if max_depth is not None and max_depth <= 0:
            raise ValueError("max_depth must be None or a positive integer")

        node_set = set(graph_nodes)

        def _hash_key(s):
            h = hashlib.sha1(s.encode("utf-8")).digest()
            return int.from_bytes(h[:4], "big")

        def _get_effective_depth(parts):
            if max_depth is None:
                return len(parts)
            return min(max_depth, len(parts))

        selected_nodes = None
        if predicate is not None:
            selected_nodes = {n for n in node_set if predicate(n, self.G.nodes[n])}
            if not selected_nodes:
                selected_nodes = None
                predicate = None

        def _lookup_by_prefix(name, values):
            if name in values:
                return values[name]
            parts = name.split(".")
            for d in range(len(parts), 0, -1):
                prefix = ".".join(parts[:d])
                if prefix in values:
                    return values[prefix]
            return 0.0

        if partition:
            root = {
                "name": "",
                "full_path": "",
                "children": {},
            }

            sources = selected_nodes if predicate is not None else node_set

            for fqn in sources:
                parts = fqn.split(".")
                current = root
                full_path = ""
                for part in parts:
                    full_path = part if not full_path else full_path + "." + part
                    children = current["children"]
                    if part not in children:
                        children[part] = {
                            "name": part,
                            "full_path": full_path,
                            "children": {},
                        }
                    current = children[part]

            values = {}

            def _order_child_keys(children):
                keys = list(children.keys())
                if alphabetical:
                    keys.sort()
                else:
                    keys.sort(key=_hash_key)
                return keys

            def _propagate(node, scalar):
                if node["full_path"]:
                    values[node["full_path"]] = scalar
                for child in node["children"].values():
                    _propagate(child, scalar)

            def _compute_weight(node):
                children = node["children"]
                if not children:
                    w = 1
                else:
                    w = 0
                    for child in children.values():
                        w += _compute_weight(child)
                node["weight"] = w
                return w

            _compute_weight(root)

            def _assign_interval(node, start, end, depth):
                if node["full_path"]:
                    values[node["full_path"]] = start

                children = node["children"]
                if not children:
                    return

                if max_depth is not None and depth >= max_depth:
                    scalar = values.get(node["full_path"], start)
                    _propagate(node, scalar)
                    return

                keys = _order_child_keys(children)
                total_weight = sum(children[k]["weight"] for k in keys)
                if total_weight <= 0:
                    n = len(keys)
                    if n == 0:
                        return
                    width = (end - start) / n
                    for i, key in enumerate(keys):
                        child_start = start + i * width
                        child_end = child_start + width
                        _assign_interval(children[key], child_start, child_end, depth + 1)
                    return

                cur = start
                length = end - start
                for key in keys:
                    child = children[key]
                    child_width = length * (child["weight"] / float(total_weight))
                    child_start = cur
                    child_end = child_start + child_width
                    _assign_interval(child, child_start, child_end, depth + 1)
                    cur = child_end

            _assign_interval(root, 0.0, 1.0, 0)

            if predicate is None:
                scalars = [values[node] for node in graph_nodes]
            else:
                scalars = [_lookup_by_prefix(node, values) for node in graph_nodes]

        else:
            if predicate is None:
                color_keys = set()
                for fqn in node_set:
                    parts = fqn.split(".")
                    depth = _get_effective_depth(parts)
                    for d in range(1, depth + 1):
                        color_keys.add(".".join(parts[:d]))
            else:
                color_keys = set()
                for fqn in selected_nodes:
                    parts = fqn.split(".")
                    depth = _get_effective_depth(parts)
                    for d in range(1, depth + 1):
                        color_keys.add(".".join(parts[:d]))

            if alphabetical:
                color_keys = sorted(color_keys)
            else:
                color_keys = sorted(color_keys, key=_hash_key)

            n_keys = len(color_keys)
            if n_keys == 0:
                return {}

            key_to_scalar = {}
            for i, key in enumerate(color_keys):
                key_to_scalar[key] = 0.0 if n_keys == 1 else i / float(n_keys - 1)

            scalars = []

            if predicate is None:
                for fqn in graph_nodes:
                    parts = fqn.split(".")
                    depth = _get_effective_depth(parts)
                    color_key = ".".join(parts[:depth])
                    scalars.append(key_to_scalar[color_key])
            else:
                color_key_set = set(color_keys)

                def _lookup_scalar_for_node(fqn):
                    parts = fqn.split(".")
                    depth = _get_effective_depth(parts)
                    for d in range(depth, 0, -1):
                        key = ".".join(parts[:d])
                        if key in color_key_set:
                            return key_to_scalar[key]
                    return 0.0

                for fqn in graph_nodes:
                    scalars.append(_lookup_scalar_for_node(fqn))

        color_list = map_value_to_cmap(
            scalars,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            as_hex=True,
        )
        return {node: color for node, color in zip(graph_nodes, color_list)}

    def generate_data(
        self,
        incl_node_positions: bool = True,
        incl_node_colors: bool = True,
        node_position_kwargs: tp.KwargsLike = None,
        node_color_kwargs: tp.KwargsLike = None,
    ) -> dict:
        """Generate data dictionary representing the graph.

        Args:
            incl_node_positions (bool): If True, include node positions in the data.
            incl_node_colors (bool): If True, include node colors in the data.
            node_position_kwargs (KwargsLike): Keyword arguments for `RefGraph.generate_node_positions`.
            node_color_kwargs (KwargsLike): Keyword arguments for `RefGraph.generate_node_colors`.

        Returns:
            dict: Data dictionary representing the graph.
        """
        if incl_node_positions:
            if node_position_kwargs is None:
                node_position_kwargs = {}
            node_positions = self.generate_node_positions(**node_position_kwargs)
        else:
            node_positions = None
        if incl_node_colors:
            if node_color_kwargs is None:
                node_color_kwargs = {}
            node_colors = self.generate_node_colors(**node_color_kwargs)
        else:
            node_colors = None

        nodes = []
        for n, d in self.G.nodes(data=True):
            node = dict(d)
            node["id"] = n
            if node_positions:
                node["x"] = float(node_positions[n][0])
                node["y"] = float(node_positions[n][1])
            if node_colors:
                node["color"] = node_colors[n]
            nodes.append(node)

        edges = []
        for u, v, d in self.G.edges(data=True):
            edge = dict(d)
            edge["source"] = u
            edge["target"] = v
            edges.append(edge)

        return {
            "is_multigraph": self.is_multigraph,
            "nodes": nodes,
            "edges": edges,
        }

    def export(
        self,
        *args,
        path: tp.Optional[tp.PathLike] = None,
        node_fields: tp.Optional[tp.List[str]] = None,
        edges_fields: tp.Optional[tp.List[str]] = None,
        **kwargs,
    ) -> None:
        """Export the graph to a JSON file.

        Args:
            *args: Positional arguments for `RefGraph.generate_data`.
            path (Optional[PathLike]): Path to the output JSON file.

                Defaults to "<ClassName>.json".
            node_fields (Optional[List[str]]): List of node fields to include in the export.

                If None, include all node fields.
            edges_fields (Optional[List[str]]): List of edge fields to include in the export.

                If None, include all edge fields.
            **kwargs: Keyword arguments for `RefGraph.generate_data`.

        Returns:
            None
        """
        if path is None:
            path = f"{type(self).__name__}.json"
        data = self.generate_data(*args, **kwargs)
        for node in data["nodes"]:
            for k, v in node.items():
                if node_fields is not None and k not in node_fields:
                    continue
                if isinstance(v, set):
                    node[k] = sorted(v)
        for edge in data["edges"]:
            for k, v in edge.items():
                if edges_fields is not None and k not in edges_fields:
                    continue
                if isinstance(v, set):
                    edge[k] = sorted(v)
        with open(path, "w") as f:
            json.dump(data, f)

    def plot(
        self,
        highlight_nodes: tp.Optional[tp.MaybeList] = None,
        module: tp.Optional[tp.ModuleLike] = None,
        resolve: bool = True,
        show_dimmed: bool = True,
        highlight_neighbors: tp.Union[bool, str] = True,
        node_position_kwargs: tp.KwargsLike = None,
        node_color_kwargs: tp.KwargsLike = None,
        node_trace_kwargs: tp.KwargsLike = None,
        edge_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        axes_equal: bool = True,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        make_figure_kwargs: tp.KwargsLike = None,
        interactive: tp.Union[bool, str] = True,
        to_dash_kwargs: tp.KwargsLike = None,
        dash_run_kwargs: tp.KwargsLike = None,
        use_webgl: tp.Optional[bool] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot the reference graph using Plotly.

        Args:
            highlight_nodes (Optional[MaybeList]): Reference name(s) of nodes to highlight.
            module (Optional[ModuleLike]): Module context used in reference resolution.
            resolve (bool): Whether to resolve the reference names to actual objects.
            show_dimmed (bool): If True, show dimmed nodes and edges when some nodes
                are highlighted; otherwise, hide them.
            highlight_neighbors (Union[bool, str]): If True, highlight neighbors of highlighted nodes.

                Supports "none" (False), "incoming", "outgoing", and "both" (True) modes.
            node_position_kwargs (KwargsLike): Keyword arguments for `RefGraph.generate_node_positions`.
            node_color_kwargs (KwargsLike): Keyword arguments for `RefGraph.generate_node_colors`.
            node_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the node trace.
            edge_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the edge trace.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            axes_equal (bool): If True, set equal scaling for x and y axes.
            xref (str): Reference for the x-axis (e.g., "x", "x2").
            yref (str): Reference for the y-axis (e.g., "y", "y2").
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            make_figure_kwargs (KwargsLike): Keyword arguments for making the figure.

                See `vectorbtpro.utils.figure.make_figure`.
            interactive (Union[bool, str]): If True, create an interactive figure.

                Supports "notebook" (with ipywidgets) and "dash" (with Dash) modes.
                If True, picks the appropriate mode based on the environment.
            to_dash_kwargs (KwargsLike): Keyword arguments for converting to a Dash app.

                See `vectorbtpro.utils.figure.FigureMixin.to_dash_app`.
            dash_run_kwargs (KwargsLike): Keyword arguments for `dash.Dash.run`.
            use_webgl (Optional[bool]): Flag to use `plotly.graph_objects.Scattergl`.

                If the global configuration is True and the data has more than 10,000 points,
                this flag becomes True.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Plotly figure representing the reference graph.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go
        from vectorbtpro.utils.colors import adjust_opacity, flatten_opacity, get_contrast_color
        from vectorbtpro.utils.figure import make_figure, get_bgcolor
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]
        layout_cfg = plotting_cfg["layout"]

        data = self.generate_data(
            node_position_kwargs=node_position_kwargs,
            node_color_kwargs=node_color_kwargs,
        )
        nodes = data["nodes"]
        edges = data["edges"]
        pos = {n["id"]: (n["x"], n["y"]) for n in nodes}
        incoming_adj = defaultdict(set)
        outgoing_adj = defaultdict(set)
        for e in edges:
            u = e["source"]
            v = e["target"]
            outgoing_adj[u].add(v)
            incoming_adj[v].add(u)

        if isinstance(highlight_neighbors, bool):
            if highlight_neighbors:
                highlight_neighbors = "both"
            else:
                highlight_neighbors = "none"

        if node_trace_kwargs is None:
            node_trace_kwargs = {}
        if edge_trace_kwargs is None:
            edge_trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if make_figure_kwargs is None:
            make_figure_kwargs = {}

        if isinstance(interactive, bool):
            if interactive:
                interactive = "notebook" if checks.in_notebook() else "dash"
            else:
                interactive = None
        if isinstance(interactive, str) and interactive.lower() == "notebook":
            make_figure_kwargs = dict(make_figure_kwargs or {})
            make_figure_kwargs["use_widgets"] = True

        if use_webgl is None:
            use_webgl = plotting_cfg["use_webgl"] and len(nodes) + len(edges) >= 10_000
        if use_webgl:
            Scatter = go.Scattergl
        else:
            Scatter = go.Scatter

        xaxis = "xaxis" + xref[1:]
        yaxis = "yaxis" + yref[1:]

        def _resolve_fig(fig):
            if fig is None:
                fig = make_figure(**make_figure_kwargs)
                def_width = layout_cfg.get("width", None)
                def_height = layout_cfg.get("height", None)
                width = layout_kwargs.get("width", None)
                height = layout_kwargs.get("height", None)
                if width is None or height is None:
                    if width is not None:
                        height = width
                    elif height is not None:
                        width = height
                    else:
                        if def_width is not None and def_height is not None:
                            width = height = max(def_width, def_height)
                        elif def_width is not None:
                            width = height = def_width
                        elif def_height is not None:
                            width = height = def_height
                def_layout_kwargs = {
                    "width": width,
                    "height": height,
                    "hovermode": "closest",
                    xaxis: dict(showgrid=False, zeroline=False, visible=False),
                    yaxis: dict(showgrid=False, zeroline=False, visible=False),
                }
                if axes_equal:
                    def_layout_kwargs[yaxis]["scaleanchor"] = xref
                    def_layout_kwargs[yaxis]["scaleratio"] = 1
                _layout_kwargs = merge_dicts(def_layout_kwargs, layout_kwargs)
            else:
                _layout_kwargs = layout_kwargs
            fig.update_layout(**_layout_kwargs)
            return fig

        fig = _resolve_fig(fig)

        bg = get_bgcolor(fig)
        contrast_color = get_contrast_color(bg)

        node_x = [n["x"] for n in nodes]
        node_y = [n["y"] for n in nodes]
        base_node_colors = [n["color"] for n in nodes]
        node_style_by_kind = {
            "module": dict(symbol="square", size=6.0, line=dict(width=0.5)),
            "class": dict(symbol="diamond", size=5.5, line=dict(width=0.5)),
            "callable": dict(symbol="circle", size=5.0, line=dict(width=0.5)),
            "data": dict(symbol="triangle-up", size=4.5, line=dict(width=0.5)),
            "unknown": dict(symbol="triangle-down", size=4.0, line=dict(width=0.5)),
        }
        node_idxs_by_kind = defaultdict(list)
        for i, n in enumerate(nodes):
            k = n.get("obj_kind", None)
            if k is None or k not in node_style_by_kind:
                k = "unknown"
            node_idxs_by_kind[k].append(i)
        node_customdata = [
            [
                n["id"],
                n.get("obj_qualname", None) or "",
                n.get("obj_module", None) or "",
                n.get("obj_type", None) or "",
                n.get("obj_kind", None) or "",
            ]
            for n in nodes
        ]
        node_hovertemplate = (
            "qualname: %{customdata[1]}<br>"
            "module: %{customdata[2]}<br>"
            "type: %{customdata[3]}<br>"
            "kind: %{customdata[4]}<extra></extra>"
        )

        edge_style_by_kind = {
            "container": dict(color=plotting_cfg["color_schema"]["gray"], width=2, dash="solid"),
            "direct_base": dict(color=plotting_cfg["color_schema"]["blue"], width=2, dash="solid"),
            "nested_base": dict(color=plotting_cfg["color_schema"]["cyan"], width=2, dash="dot"),
            "direct_member": dict(color=plotting_cfg["color_schema"]["green"], width=1, dash="solid"),
            "nested_member": dict(color=plotting_cfg["color_schema"]["orange"], width=1, dash="dot"),
            "direct_dependency": dict(color=plotting_cfg["color_schema"]["purple"], width=1, dash="solid"),
            "nested_dependency": dict(color=plotting_cfg["color_schema"]["pink"], width=1, dash="dot"),
            "unknown": dict(color=plotting_cfg["color_schema"]["brown"], width=1, dash="dash"),
        }
        edges_by_kind = defaultdict(list)
        for e in edges:
            if self.is_multigraph:
                kind = e.get("rel_kind", None)
                if kind is None or kind not in edge_style_by_kind:
                    kind = "unknown"
                edges_by_kind[kind].append(e)
            else:
                kinds = e.get("rel_kinds", set())
                if kinds:
                    for kind in kinds:
                        if kind is None or kind not in edge_style_by_kind:
                            kind = "unknown"
                        edges_by_kind[kind].append(e)
                else:
                    edges_by_kind["unknown"].append(e)
        kind_edge_data = {}
        for kind in edge_style_by_kind:
            kd = []
            for e in edges_by_kind.get(kind, []):
                u = e["source"]
                v = e["target"]
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                kd.append((u, v, x0, y0, x1, y1))
            kind_edge_data[kind] = kd

        adj_color_cache = {}
        dim_color_cache = {}

        def _create_trace_specs(highlight_nodes):
            if isinstance(highlight_nodes, str):
                highlight_nodes = [highlight_nodes]
            highlight_set = set(highlight_nodes or [])
            highlight_set = {ensure_refname(n, module=module, resolve=resolve) for n in highlight_set}
            highlight_set &= set(pos.keys())
            neighbor_set = dict(incoming=set(), outgoing=set())
            if highlight_set and highlight_neighbors.lower() != "none":
                mode = highlight_neighbors.lower()
                if mode in ("both", "outgoing"):
                    neighbor_set["outgoing"] = set().union(*(outgoing_adj[n] for n in highlight_set))
                    neighbor_set["outgoing"] -= highlight_set
                if mode in ("both", "incoming"):
                    neighbor_set["incoming"] = set().union(*(incoming_adj[n] for n in highlight_set))
                    neighbor_set["incoming"] -= highlight_set

            def _adj_color(c, alpha):
                key = (c, alpha)
                out = adj_color_cache.get(key)
                if out is None:
                    out = adjust_opacity(c, alpha)
                    adj_color_cache[key] = out
                return out

            def _dim_color(c, alpha):
                key = (c, alpha)
                out = dim_color_cache.get(key)
                if out is None:
                    out = flatten_opacity(_adj_color(c, alpha), background_color=bg)
                    dim_color_cache[key] = out
                return out

            node_colors = list(base_node_colors)
            if highlight_set:
                marker_opacity = []
                for i, n in enumerate(nodes):
                    nid = n["id"]
                    if nid in highlight_set:
                        marker_opacity.append(1.0)
                    elif nid in neighbor_set["incoming"] or nid in neighbor_set["outgoing"]:
                        marker_opacity.append(0.6)
                    elif highlight_set:
                        marker_opacity.append(1.0)
                        node_colors[i] = _dim_color(node_colors[i], 0.2)
                    else:
                        marker_opacity.append(0.2)
            else:
                marker_opacity = [1.0] * len(nodes)

            gl_node_trace_spec = None
            dim_node_trace_specs = []
            hl_node_trace_specs = []
            legend_node_trace_specs = []

            if use_webgl:
                hl_idxs = []
                for i in range(len(nodes)):
                    nid = nodes[i]["id"]
                    if (
                        not highlight_set
                        or nid in highlight_set
                        or nid in neighbor_set["incoming"]
                        or nid in neighbor_set["outgoing"]
                    ):
                        hl_idxs.append(i)

                hit_marker = dict(
                    symbol="circle",
                    size=8,
                    line=dict(width=0),
                    color="rgba(0,0,0,0)",
                )
                gl_node_trace_spec = merge_dicts(
                    dict(
                        x=[node_x[i] for i in hl_idxs],
                        y=[node_y[i] for i in hl_idxs],
                        mode="markers",
                        marker=hit_marker,
                        customdata=[node_customdata[i] for i in hl_idxs],
                        name="nodes_hit",
                        showlegend=False,
                        hovertemplate=node_hovertemplate,
                    ),
                    node_trace_kwargs,
                )

            for kind in node_style_by_kind:
                node_idxs = node_idxs_by_kind.get(kind, [])

                hl_idxs = []
                dim_idxs = []
                for i in node_idxs:
                    nid = nodes[i]["id"]
                    if (
                        not highlight_set
                        or nid in highlight_set
                        or nid in neighbor_set["incoming"]
                        or nid in neighbor_set["outgoing"]
                    ):
                        hl_idxs.append(i)
                    elif show_dimmed:
                        dim_idxs.append(i)

                marker = dict(node_style_by_kind.get(kind, node_style_by_kind["unknown"]))

                dim_node_trace_specs.append(
                    merge_dicts(
                        dict(
                            x=[node_x[i] for i in dim_idxs],
                            y=[node_y[i] for i in dim_idxs],
                            mode="markers",
                            marker={
                                **marker,
                                "color": [node_colors[i] for i in dim_idxs],
                                "opacity": [marker_opacity[i] for i in dim_idxs],
                            },
                            customdata=[node_customdata[i] for i in dim_idxs],
                            name=f"{kind}_nodes_dim",
                            legendgroup=f"{kind}_nodes",
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        node_trace_kwargs,
                    )
                )

                hl_node_trace_specs.append(
                    merge_dicts(
                        dict(
                            x=[node_x[i] for i in hl_idxs],
                            y=[node_y[i] for i in hl_idxs],
                            mode="markers",
                            marker={
                                **marker,
                                "color": [node_colors[i] for i in hl_idxs],
                                "opacity": [marker_opacity[i] for i in hl_idxs],
                            },
                            customdata=[node_customdata[i] for i in hl_idxs],
                            name=f"{kind}_nodes_hl",
                            legendgroup=f"{kind}_nodes",
                            showlegend=False,
                            hovertemplate=node_hovertemplate,
                        ),
                        node_trace_kwargs,
                    )
                )

                legend_marker = dict(marker)
                legend_marker["size"] = 8
                legend_marker["color"] = "rgba(0,0,0,0)"
                legend_marker["line"] = dict(legend_marker["line"])
                legend_marker["line"]["color"] = contrast_color
                legend_node_trace_specs.append(
                    merge_dicts(
                        dict(
                            x=[None],
                            y=[None],
                            mode="markers",
                            marker=legend_marker,
                            name=f"{kind}_nodes",
                            legendgroup=f"{kind}_nodes",
                            showlegend=True,
                            hoverinfo="skip",
                        ),
                        node_trace_kwargs,
                    )
                )

            dim_edge_trace_specs = []
            hl_edge_trace_specs = []
            legend_edge_trace_specs = []

            for kind in edge_style_by_kind:
                default_edge_visible = True if kind == "container" else "legendonly"

                hl_edge_x, hl_edge_y = [], []
                dim_edge_x, dim_edge_y = [], []
                hl_dir_x, hl_dir_y = [], []
                for u, v, x0, y0, x1, y1 in kind_edge_data.get(kind, []):
                    highlight_edge = (
                        (u in highlight_set and v in highlight_set)
                        or (u in highlight_set and v in neighbor_set["outgoing"])
                        or (u in neighbor_set["incoming"] and v in highlight_set)
                    )
                    if highlight_set and highlight_edge:
                        hl_edge_x += [x0, x1, None]
                        hl_edge_y += [y0, y1, None]

                        sx = x0 + (x1 - x0) * 0.9
                        sy = y0 + (y1 - y0) * 0.9
                        ex, ey = x1, y1
                        hl_dir_x += [sx, ex, None]
                        hl_dir_y += [sy, ey, None]
                    else:
                        dim_edge_x += [x0, x1, None]
                        dim_edge_y += [y0, y1, None]

                line = dict(edge_style_by_kind.get(kind, edge_style_by_kind["unknown"]))

                dim_line = dict(line)
                if highlight_set:
                    dim_line["color"] = _dim_color(dim_line["color"], 0.2)
                else:
                    dim_line["color"] = _adj_color(dim_line["color"], 0.2)
                dim_x = dim_edge_x if show_dimmed or not highlight_set else []
                dim_y = dim_edge_y if show_dimmed or not highlight_set else []
                dim_edge_trace_specs.append(
                    merge_dicts(
                        dict(
                            x=dim_x,
                            y=dim_y,
                            mode="lines",
                            line=dim_line,
                            name=f"{kind}_edges_dim",
                            legendgroup=f"{kind}_edges",
                            showlegend=False,
                            hoverinfo="none",
                            visible=default_edge_visible,
                        ),
                        edge_trace_kwargs,
                    )
                )

                hl_line = dict(line)
                hl_line["width"] = hl_line["width"] + 1
                hl_line["color"] = _adj_color(hl_line["color"], 0.8)
                hl_x = hl_edge_x if highlight_set else []
                hl_y = hl_edge_y if highlight_set else []
                hl_edge_trace_specs.append(
                    merge_dicts(
                        dict(
                            x=hl_x,
                            y=hl_y,
                            mode="lines",
                            line=hl_line,
                            name=f"{kind}_edges_hl",
                            legendgroup=f"{kind}_edges",
                            showlegend=False,
                            hoverinfo="none",
                            visible=default_edge_visible,
                        ),
                        edge_trace_kwargs,
                    )
                )

                dir_line = dict(hl_line)
                dir_line["width"] = hl_line["width"] * 2
                dir_x = hl_dir_x if highlight_set else []
                dir_y = hl_dir_y if highlight_set else []
                hl_edge_trace_specs.append(
                    merge_dicts(
                        dict(
                            x=dir_x,
                            y=dir_y,
                            mode="lines",
                            line=dir_line,
                            name=f"{kind}_edges_dir",
                            legendgroup=f"{kind}_edges",
                            showlegend=False,
                            hoverinfo="skip",
                            visible=default_edge_visible,
                        ),
                        edge_trace_kwargs,
                    )
                )

                legend_edge_trace_specs.append(
                    merge_dicts(
                        dict(
                            x=[None],
                            y=[None],
                            mode="lines",
                            line=line,
                            name=f"{kind}_edges",
                            legendgroup=f"{kind}_edges",
                            showlegend=True,
                            hoverinfo="skip",
                            visible=default_edge_visible,
                        ),
                        edge_trace_kwargs,
                    )
                )

            trace_specs = []
            if gl_node_trace_spec is not None:
                trace_specs.append(gl_node_trace_spec)
            trace_specs.extend(dim_edge_trace_specs)
            trace_specs.extend(dim_node_trace_specs)
            trace_specs.extend(hl_edge_trace_specs)
            trace_specs.extend(hl_node_trace_specs)
            trace_specs.extend(legend_edge_trace_specs)
            trace_specs.extend(legend_node_trace_specs)
            return trace_specs

        for trace_spec in _create_trace_specs(highlight_nodes):
            fig.add_trace(Scatter(**trace_spec), **add_trace_kwargs)

        if interactive is None:
            return fig

        def _norm_visible(v):
            return True if v is None else v

        def _extract_vis_by_group(data):
            out = {}
            for trace in data:
                lg = get(trace, "legendgroup", None)
                if not lg:
                    continue
                if get(trace, "showlegend", False):
                    out[lg] = _norm_visible(get(trace, "visible", True))
            for trace in data:
                lg = get(trace, "legendgroup", None)
                if lg and lg not in out:
                    out[lg] = _norm_visible(get(trace, "visible", True))
            return out

        def _apply_vis_by_group(data, vis_by_group, default=None):
            for trace in data:
                lg = get(trace, "legendgroup", None)
                if not lg:
                    continue
                v = get(vis_by_group, lg, default)
                if v is None:
                    continue
                trace["visible"] = v

        def _make_all_groups_visible(data):
            for trace in data:
                if get(trace, "legendgroup"):
                    trace["visible"] = True

        if isinstance(interactive, str) and interactive.lower() == "notebook":
            active_node = None
            base_vis_by_group = {}
            hl_vis_by_group = None
            cb_attached = set()

            def _patch_trace(dst, trace_spec):
                dst.update(trace_spec, overwrite=True)

            base_vis_by_group = _extract_vis_by_group(fig.data)

            def _update_highlight(node_id):
                nonlocal active_node, highlight_nodes, base_vis_by_group, hl_vis_by_group

                current_map = _extract_vis_by_group(fig.data)
                if active_node is None:
                    base_vis_by_group.update(current_map)
                else:
                    if hl_vis_by_group is None:
                        hl_vis_by_group = {}
                    hl_vis_by_group.update(current_map)

                if active_node == node_id:
                    new_active = None
                    new_trace_specs = _create_trace_specs(highlight_nodes)
                else:
                    new_active = node_id
                    new_trace_specs = _create_trace_specs([node_id])

                if new_active is None:
                    desired = dict(base_vis_by_group)
                else:
                    if hl_vis_by_group is None:
                        tmp_trace_specs = [dict(trace_spec) for trace_spec in new_trace_specs]
                        _make_all_groups_visible(tmp_trace_specs)
                        hl_vis_by_group = _extract_vis_by_group(tmp_trace_specs)
                    desired = dict(hl_vis_by_group)

                with fig.batch_update():
                    for old_trace, new_trace_spec in zip(fig.data, new_trace_specs):
                        _patch_trace(old_trace, new_trace_spec)
                    _apply_vis_by_group(fig.data, desired, default=None)

                active_node = new_active

            def _handle_click(trace, points, state):
                if not points.point_inds:
                    return
                idx = points.point_inds[0]
                customdata = getattr(trace, "customdata", None)
                if customdata is None or idx >= len(customdata):
                    return
                node_id = customdata[idx][0]
                _update_highlight(node_id)

            def _attach_callbacks(widget):
                for trace in widget.data:
                    k = id(trace)
                    if k in cb_attached:
                        continue
                    cb_attached.add(k)
                    mode = getattr(trace, "mode", None)
                    customdata = getattr(trace, "customdata", None)
                    if mode and "markers" in mode and customdata is not None:
                        trace.on_click(_handle_click)

            _attach_callbacks(fig)
            return fig

        elif isinstance(interactive, str) and interactive.lower() == "dash":
            assert_can_import("dash")
            import dash

            if to_dash_kwargs is None:
                to_dash_kwargs = {}
            if dash_run_kwargs is None:
                dash_run_kwargs = {}

            def _plot(highlight_nodes):
                fig = _resolve_fig(None)
                for trace_spec in _create_trace_specs(highlight_nodes):
                    fig.add_trace(Scatter(**trace_spec), **add_trace_kwargs)
                return fig

            def _update_vis_store_from_restyle(restyle_data, fig_json, vis_store):
                if not restyle_data or len(restyle_data) != 2:
                    return vis_store
                edits, idxs = restyle_data
                if not isinstance(edits, dict) or "visible" not in edits:
                    return vis_store
                vis_store = dict(vis_store or {})
                vis_vals = edits["visible"]
                if not isinstance(vis_vals, (list, tuple)):
                    vis_vals = [vis_vals] * len(idxs)
                data = get(fig_json, "data", [])
                for i, tr_idx in enumerate(idxs):
                    if not isinstance(tr_idx, int) or tr_idx < 0 or tr_idx >= len(data):
                        continue
                    lg = get(data[tr_idx], "legendgroup")
                    if not lg:
                        continue
                    vis_store[lg] = _norm_visible(vis_vals[i])
                return vis_store

            fig.update_layout(legend=dict(groupclick="togglegroup"))

            base_vis_init = _extract_vis_by_group(fig.data)

            app = fig.to_dash_app(**to_dash_kwargs)
            app.layout.children.insert(0, dash.dcc.Store(id="active-node", data=None))
            app.layout.children.insert(0, dash.dcc.Store(id="base-vis", data=base_vis_init))
            app.layout.children.insert(0, dash.dcc.Store(id="hl-vis", data=None))

            @app.callback(
                dash.Output("graph", "figure"),
                dash.Output("active-node", "data"),
                dash.Output("base-vis", "data"),
                dash.Output("hl-vis", "data"),
                dash.Input("graph", "clickData"),
                dash.Input("graph", "restyleData"),
                dash.State("graph", "figure"),
                dash.State("active-node", "data"),
                dash.State("base-vis", "data"),
                dash.State("hl-vis", "data"),
                prevent_initial_call=True,
            )
            def _update(click_data, restyle_data, fig_json, active_node, base_vis, hl_vis):
                trig = dash.callback_context.triggered
                prop_id = trig[0]["prop_id"] if trig else ""
                prop = prop_id.split(".", 1)[1] if "." in prop_id else ""

                if prop == "restyleData" and restyle_data is not None:
                    if fig_json is None:
                        raise dash.exceptions.PreventUpdate
                    if active_node is None:
                        base_vis = _update_vis_store_from_restyle(restyle_data, fig_json, base_vis)
                    else:
                        hl_vis = _update_vis_store_from_restyle(restyle_data, fig_json, hl_vis)
                    return dash.no_update, dash.no_update, base_vis, hl_vis

                if click_data is None or "points" not in click_data or not click_data["points"]:
                    raise dash.exceptions.PreventUpdate

                pt = click_data["points"][0]
                customdata = pt.get("customdata", None)
                if not customdata:
                    raise dash.exceptions.PreventUpdate
                node_id = customdata[0]

                if active_node == node_id:
                    new_active = None
                    new_fig = _plot(highlight_nodes)
                    new_fig.update_layout(legend=dict(groupclick="togglegroup"))
                    if base_vis is None:
                        base_vis = _extract_vis_by_group(new_fig.data)
                    _apply_vis_by_group(new_fig.data, base_vis, default=None)
                else:
                    new_active = node_id
                    new_fig = _plot([node_id])
                    new_fig.update_layout(legend=dict(groupclick="togglegroup"))
                    if hl_vis is None:
                        _make_all_groups_visible(new_fig.data)
                        hl_vis = _extract_vis_by_group(new_fig.data)
                    _apply_vis_by_group(new_fig.data, hl_vis, default=None)

                if to_dash_kwargs.get("fit_to_window", False):
                    new_fig.update_layout(autosize=True, width=None, height=None)
                return new_fig, new_active, base_vis, hl_vis

            app.run(**dash_run_kwargs)
        else:
            raise ValueError(f"Invalid interactive: {interactive!r}")


DBlock = tp.Literal["decorator", "head", "body"]
"""Literal type representing different parts of a scope where a dependency can occur."""

DRole = tp.Literal["expr", "annotation", "default", "returns", "base", "metaclass", "keyword"]
"""Literal type representing different syntactic roles of dependencies."""


@define
class DHitMeta(DefineMixin):
    """Class representing metadata for a single dependency hit in the source code."""

    name: str = define.field()
    """Name used in the source code."""

    refname: str = define.field()
    """Fully qualified reference name of the used name."""

    lineno: int = define.field()
    """Line number of the dependency hit (1-indexed)."""

    col_offset: int = define.field()
    """Column offset of the dependency hit (0-indexed)."""

    end_lineno: tp.Optional[int] = define.field(default=None)
    """End line number of the dependency hit (1-indexed)."""

    end_col_offset: tp.Optional[int] = define.field(default=None)
    """End column offset of the dependency hit (0-indexed)."""

    block: DBlock = define.field(default="body")
    """Part of the scope where the dependency occurs."""

    role: DRole = define.field(default="expr")
    """Syntactic role of the dependency."""

    scope_refname: tp.Optional[str] = define.field(default=None)
    """Reference name of the scope where the dependency occurs."""

    source_line: tp.Optional[str] = define.field(default=None)
    """Source line of the dependency hit."""

    @property
    def is_builtin(self) -> bool:
        """Check if the dependency hit refers to a builtin object.

        Returns:
            bool: True if the reference name starts with "builtins.", False otherwise.
        """
        return self.refname == "builtins" or self.refname.startswith("builtins.")

    @property
    def is_unreachable(self) -> bool:
        """Check if the dependency hit refers to an unreachable scope.

        Returns:
            bool: True if the reference name or scope reference name contains "::", False otherwise.
        """
        return "::" in self.refname or "::" in self.scope_refname

    @property
    def is_private(self) -> bool:
        """Check if the dependency hit refers to a private object.

        Returns:
            bool: True if the last part of the reference name starts with "_", False otherwise.
        """
        last_part = self.refname.split(".")[-1]
        return last_part.startswith("_")


class RefIndex(Configured):
    """Class representing a lazy reference index across modules.

    Args:
        expand_star_imports (bool): If True, attempts to resolve `from x import *` by importing the module
            and expanding its public names.
        container_kinds (Optional[List[str]]): Container kinds to visit regardless of checks.
        incl_modules (Optional[MaybeList[ModuleLike]]): Module names or objects to include for reference names.
        excl_modules (Optional[MaybeList[ModuleLike]]): Module names or objects to exclude for reference names.
        incl_unreachable (bool): Whether to allow visiting reference names from unreachable scopes.
        incl_builtins (bool): Whether to allow visiting reference names from builtins.
        incl_private (bool): Whether to allow visiting private reference names.
        incl_predicate (Optional[Callable[[str], bool]]): Predicate function to include reference names.

            Called before other checks; if it returns True, the name is included.
        excl_predicate (Optional[Callable[[str], bool]]): Predicate function to exclude reference names.

            Called after other checks; if it returns True, the name is excluded.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.
    """

    def __init__(
        self,
        expand_star_imports: bool = False,
        container_kinds: tp.Optional[tp.List[str]] = None,
        incl_modules: tp.Optional[tp.MaybeList[tp.ModuleLike]] = None,
        excl_modules: tp.Optional[tp.MaybeList[tp.ModuleLike]] = None,
        incl_unreachable: bool = False,
        incl_builtins: bool = False,
        incl_private: bool = False,
        incl_predicate: tp.Optional[tp.Callable[[str], bool]] = None,
        excl_predicate: tp.Optional[tp.Callable[[str], bool]] = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            expand_star_imports=expand_star_imports,
            container_kinds=container_kinds,
            incl_modules=incl_modules,
            excl_modules=excl_modules,
            incl_unreachable=incl_unreachable,
            incl_builtins=incl_builtins,
            incl_private=incl_private,
            incl_predicate=incl_predicate,
            excl_predicate=excl_predicate,
            **kwargs,
        )

        if incl_modules is None:
            incl_modules = []
        elif not isinstance(incl_modules, list):
            incl_modules = [incl_modules]
        incl_modules = [resolve_module(module) for module in incl_modules]
        if excl_modules is None:
            excl_modules = []
        elif not isinstance(excl_modules, list):
            excl_modules = [excl_modules]
        excl_modules = [resolve_module(module) for module in excl_modules]

        self._expand_star_imports = expand_star_imports
        self._container_kinds = container_kinds
        self._incl_modules = incl_modules
        self._excl_modules = excl_modules
        self._incl_unreachable = incl_unreachable
        self._incl_builtins = incl_builtins
        self._incl_private = incl_private
        self._incl_predicate = incl_predicate
        self._excl_predicate = excl_predicate

        self._dependencies = {}

    @property
    def expand_star_imports(self) -> bool:
        """If True, attempts to resolve `from x import *` by importing the module
        and expanding its public names.

        Returns:
            bool: Whether to expand star imports.
        """
        return self._expand_star_imports

    @property
    def container_kinds(self) -> tp.Optional[tp.List[str]]:
        """Container kinds to visit regardless of checks.

        Returns:
            Optional[List[str]]: List of container kinds, or None if all kinds are considered.
        """
        return self._container_kinds

    @property
    def incl_modules(self) -> tp.List[tp.ModuleLike]:
        """Module names or objects to include for reference names.

        Returns:
            List[ModuleLike]: List of included modules.
        """
        return self._incl_modules

    @property
    def excl_modules(self) -> tp.List[tp.ModuleLike]:
        """Module names or objects to exclude for reference names.

        Returns:
            List[ModuleLike]: List of excluded modules.
        """
        return self._excl_modules

    @property
    def incl_unreachable(self) -> bool:
        """Whether to allow visiting reference names from unreachable scopes.

        Returns:
            bool: Whether to include unreachable scopes.
        """
        return self._incl_unreachable

    @property
    def incl_builtins(self) -> bool:
        """Whether to allow visiting reference names from builtins.

        Returns:
            bool: Whether to include builtins.
        """
        return self._incl_builtins

    @property
    def incl_private(self) -> bool:
        """Whether to allow visiting private reference names.

        Returns:
            bool: Whether to include private names.
        """
        return self._incl_private

    @property
    def incl_predicate(self) -> tp.Optional[tp.Callable[[str], bool]]:
        """Predicate function to include reference names.

        Called before other checks; if it returns True, the name is included.

        Returns:
            Optional[Callable[[str], bool]]: Predicate function to include reference names.
        """
        return self._incl_predicate

    @property
    def excl_predicate(self) -> tp.Optional[tp.Callable[[str], bool]]:
        """Predicate function to exclude reference names.

        Called after other checks; if it returns True, the name is excluded.

        Returns:
            Optional[Callable[[str], bool]]: Predicate function to exclude reference names.
        """
        return self._excl_predicate

    @property
    def dependencies(self) -> tp.Dict[str, tp.List[DHitMeta]]:
        """Dependencies for all modules in the index.

        Returns:
            Dict[str, List[DHitMeta]]: Dictionary mapping module names to dependency hit metadata
                as lists of `DHitMeta` instances.
        """
        return self._dependencies

    @classmethod
    def get_dependencies(
        cls,
        module: tp.ModuleLike,
        expand_star_imports: bool = False,
        return_matrix: bool = False,
        return_meta: bool = False,
        unique_only: bool = True,
    ) -> tp.List[tp.MaybeList[tp.Union[str, DHitMeta]]]:
        """Get dependencies (i.e., non-local name usages) in the specified module.

        This analyzes name loads inside each scope (modules, functions, classes,
        lambdas, and comprehensions) and resolves them to fully qualified reference names.
        Locals of the current scope are excluded; imports and names from enclosing scopes,
        the module scope, or builtins may be included depending on the settings.

        Generated reference names may include unreachable scopes (e.g., `<lambda>`, `<listcomp>`)
        that are not accessible via attribute access. These are rendered with `::`, e.g., `pkg.mod.func::<lambda>`.

        Args:
            module (ModuleLike): Module reference name or object.
            expand_star_imports (bool): If True, attempts to resolve `from x import *` by importing the module
                and expanding its public names.
            return_matrix (bool): If True, returns a list whose length equals the number of lines
                in the module's source; each entry contains results for that line.

                If False, returns a flat list across the whole file.
            return_meta (bool): If True, returns detailed dependency hit metadata of type `DHitMeta`.

                If False, returns only reference name strings.
            unique_only (bool): If True, return only unique reference names (applies only when `return_meta` is False).

        Returns:
            list: When `return_matrix` is True, returns a list of per-line results
                (where lines with no hits are empty lists), otherwise a flat list.

                Also, when `return_meta` is True, each result is a list of `DHitMeta` instances,
                otherwise a list of reference name strings.

        """
        from vectorbtpro.utils.source import get_source, absolutize_import

        module = resolve_module(module)
        source = get_source(module)
        source_lines = source.splitlines()
        n_lines = len(source_lines)
        tree = ast.parse(source, type_comments=True)

        def _is_scope(n):
            return isinstance(
                n,
                (
                    ast.Module,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.Lambda,
                    ast.ListComp,
                    ast.SetComp,
                    ast.DictComp,
                    ast.GeneratorExp,
                ),
            )

        def _add_target_names(target, out):
            if isinstance(target, ast.Name):
                out.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    _add_target_names(elt, out)
            elif isinstance(target, ast.Starred):
                _add_target_names(target.value, out)

        def _add_pattern_binds(pat, out):
            if not sys.version_info >= (3, 10):
                return
            T = ast
            if isinstance(pat, getattr(T, "MatchAs", ())):
                if pat.name:
                    out.add(pat.name)
                if pat.pattern:
                    _add_pattern_binds(pat.pattern, out)
            elif isinstance(pat, getattr(T, "MatchStar", ())):
                if pat.name:
                    out.add(pat.name)
            elif isinstance(pat, getattr(T, "MatchOr", ())):
                for p in pat.patterns:
                    _add_pattern_binds(p, out)
            elif isinstance(pat, getattr(T, "MatchSequence", ())):
                for p in pat.patterns:
                    _add_pattern_binds(p, out)
            elif isinstance(pat, getattr(T, "MatchMapping", ())):
                for p in pat.patterns:
                    _add_pattern_binds(p, out)
                if pat.rest:
                    out.add(pat.rest)
            elif isinstance(pat, getattr(T, "MatchClass", ())):
                for p in pat.patterns:
                    _add_pattern_binds(p, out)
                for kp in getattr(pat, "kwd_patterns", []):
                    _add_pattern_binds(kp, out)

        class ScopeInfo:
            def __init__(self, node, parent, refname):
                self._final = None

                self.node = node
                self.parent = parent
                self.locals = set()
                self.globals = set()
                self.nonlocals = set()
                self.imports = {}
                self.star_imports = []
                self.refname = refname

            def final_locals(self):
                if self._final is None:
                    self._final = self.locals - self.globals - self.nonlocals
                return self._final

        scope_infos = {}

        def _scope_refname_for(node, parent):
            base = parent.refname if parent else module.__name__
            if isinstance(node, ast.Module):
                return module.__name__
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return f"{base}.{node.name}"
            tag = type(node).__name__.lower()
            return f"{base}::<{tag}>"

        class LocalCollector(ast.NodeVisitor):
            def __init__(self):
                self._stack = []

            def visit(self, node):
                if _is_scope(node):
                    parent = self._stack[-1] if self._stack else None
                    sc = ScopeInfo(node, parent, _scope_refname_for(node, parent))
                    scope_infos[node] = sc
                    self._stack.append(sc)
                    self._collect_within(node, sc)
                    super().visit(node)
                    self._stack.pop()
                else:
                    super().visit(node)

            def _collect_within(self, root, sc):
                for child in ast.iter_child_nodes(root):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        sc.locals.add(child.name)
                    if hasattr(ast, "TypeAlias") and isinstance(child, getattr(ast, "TypeAlias")):
                        try:
                            sc.locals.add(child.name.id)
                        except Exception:
                            pass

                if isinstance(root, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    for gen in root.generators:
                        _add_target_names(gen.target, sc.locals)

                def _add_args_as_locals(args):
                    if not args:
                        return
                    for a in itertools.chain(getattr(args, "posonlyargs", ()), args.args, args.kwonlyargs):
                        sc.locals.add(a.arg)
                    if getattr(args, "vararg", None):
                        sc.locals.add(args.vararg.arg)
                    if getattr(args, "kwarg", None):
                        sc.locals.add(args.kwarg.arg)

                if isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    _add_args_as_locals(getattr(root, "args", None))

                def _bind_type_params_if_any(node, sc):
                    tp = getattr(node, "type_params", None)
                    if not tp:
                        return
                    for p in getattr(tp, "params", []):
                        name = getattr(p, "name", None)
                        if isinstance(name, ast.Name):
                            sc.locals.add(name.id)
                        elif isinstance(name, str):
                            sc.locals.add(name)

                if isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    _bind_type_params_if_any(root, sc)

                def _walk_here(n):
                    for child in ast.iter_child_nodes(n):
                        if _is_scope(child):
                            continue
                        if isinstance(child, ast.Global):
                            sc.globals.update(child.names)
                            continue
                        if isinstance(child, ast.Nonlocal):
                            sc.nonlocals.update(child.names)
                            continue
                        if isinstance(child, ast.Import):
                            for alias in child.names:
                                local = alias.asname or alias.name.split(".", 1)[0]
                                target = alias.name if alias.asname else alias.name.split(".", 1)[0]
                                sc.locals.add(local)
                                sc.imports[local] = target
                            continue
                        if isinstance(child, ast.ImportFrom):
                            abs_mod = absolutize_import(module, child.level or 0, child.module)
                            for alias in child.names:
                                if alias.name == "*":
                                    if abs_mod:
                                        sc.star_imports.append(abs_mod)
                                    continue
                                local = alias.asname or alias.name
                                if abs_mod:
                                    sc.locals.add(local)
                                    sc.imports[local] = f"{abs_mod}.{alias.name}"
                                else:
                                    sc.locals.add(local)
                                    sc.imports[local] = alias.name
                            continue
                        if isinstance(child, ast.Assign):
                            for t in child.targets:
                                _add_target_names(t, sc.locals)
                        elif isinstance(child, ast.AugAssign):
                            _add_target_names(child.target, sc.locals)
                        elif isinstance(child, ast.AnnAssign):
                            if child.value is not None:
                                _add_target_names(child.target, sc.locals)
                        elif isinstance(child, (ast.For, ast.AsyncFor)):
                            _add_target_names(child.target, sc.locals)
                        elif isinstance(child, (ast.With, ast.AsyncWith)):
                            for item in child.items:
                                if item.optional_vars:
                                    _add_target_names(item.optional_vars, sc.locals)
                        elif isinstance(child, ast.NamedExpr):
                            _add_target_names(child.target, sc.locals)
                        elif isinstance(child, ast.ExceptHandler):
                            nm = getattr(child, "name", None)
                            if sys.version_info >= (3, 11) and isinstance(nm, ast.Name):
                                sc.locals.add(nm.id)
                            elif isinstance(nm, str):
                                sc.locals.add(nm)
                        if sys.version_info >= (3, 10) and isinstance(child, getattr(ast, "Match", ())):
                            for case in child.cases:
                                _add_pattern_binds(case.pattern, sc.locals)
                        _walk_here(child)

                _walk_here(root)

        LocalCollector().visit(tree)

        if expand_star_imports:
            for sc in scope_infos.values():
                for modname in sc.star_imports:
                    spec = importlib.util.find_spec(modname)
                    if not spec or not (spec.origin or "").endswith((".py", ".pyc")):
                        continue
                    try:
                        mod = import_module(modname)
                    except Exception:
                        continue
                    public = getattr(mod, "__all__", None)
                    if public is None:
                        public = [n for n in dir(mod) if not n.startswith("_")]
                    for n in public:
                        if n not in sc.locals:
                            sc.locals.add(n)
                            sc.imports[n] = f"{modname}.{n}"
                sc.star_imports.clear()

        builtin_names = set(dir(builtins))
        module_scope = scope_infos.get(tree, None)

        def _is_plain_segment(seg):
            return seg and not seg.startswith("<") and seg.isidentifier()

        @lru_cache(maxsize=None)
        def _resolve_scope_object(scope_refname):
            if not isinstance(scope_refname, str):
                return None, None
            mod_name = module.__name__
            if not scope_refname.startswith(mod_name):
                return None, None
            mod_parts = mod_name.split(".")
            scope_parts = scope_refname.split(".")
            if scope_parts[: len(mod_parts)] != mod_parts:
                return None, None

            obj = module
            walked = mod_parts[:]
            for seg in scope_parts[len(mod_parts) :]:
                if not _is_plain_segment(seg):
                    return None, None
                try:
                    obj = get_attr(obj, seg)
                except AttributeError:
                    return None, None
                walked.append(seg)
            return obj, ".".join(walked)

        @lru_cache(maxsize=None)
        def _attr_path_if_accessible(scope_refname, name):
            obj, dotted = _resolve_scope_object(scope_refname)
            if obj is None:
                return None if name is not None else None
            if name is None:
                return dotted
            if not _is_plain_segment(name):
                return None
            try:
                get_attr(obj, name)
                return f"{dotted}.{name}"
            except AttributeError:
                return None

        def _resolve_refname(name, cur):
            if name in getattr(cur, "globals", ()):
                if module_scope is not None:
                    if name in module_scope.imports:
                        return module_scope.imports[name]
                    if name in module_scope.final_locals():
                        dotp = _attr_path_if_accessible(module_scope.refname, name)
                        return dotp or f"{module_scope.refname}::{name}"
                dotp = _attr_path_if_accessible(module_scope.refname, name)
                return dotp or f"{module_scope.refname}::{name}"
            sc = cur
            while sc is not None:
                if name in sc.imports:
                    return sc.imports[name]
                if name in sc.final_locals():
                    dotp = _attr_path_if_accessible(sc.refname, name)
                    return dotp or f"{sc.refname}::{name}"
                sc = sc.parent
            if name in builtin_names:
                return f"builtins.{name}"
            return None

        def _scope_leaf_label(node) -> str:
            if isinstance(node, ast.Module):
                return module.__name__
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return node.name
            return f"<{type(node).__name__.lower()}>"

        @lru_cache(maxsize=None)
        def _canonicalize_refname(refname):
            if "::" not in refname:
                new_refname = ensure_refname(refname, raise_error=False)
                if new_refname is not None:
                    refname = new_refname
                return refname
            base, suffix = refname.split("::", 1)
            base_refname = ensure_refname(base, raise_error=False)
            if base_refname is not None:
                base = base_refname
            return base + "::" + suffix

        def _format_scope_refname(sc):
            if sc is None:
                return module.__name__
            dotted = _attr_path_if_accessible(sc.refname, None)
            if dotted:
                return _canonicalize_refname(dotted)
            parent_disp = _format_scope_refname(sc.parent) if sc.parent else module.__name__
            return _canonicalize_refname(f"{parent_disp}::{_scope_leaf_label(sc.node)}")

        class UseCollector(ast.NodeVisitor):
            def __init__(self):
                self._stack = []
                self._block_stack = ["body"]
                self._role_stack = ["expr"]
                self._name_nodes_emitted = set()

                self.hit_meta_by_line = defaultdict(list)

            @contextlib.contextmanager
            def _ctx(self, *, block=None, role=None):
                if block is not None:
                    self._block_stack.append(block)
                if role is not None:
                    self._role_stack.append(role)
                try:
                    yield
                finally:
                    if role is not None:
                        self._role_stack.pop()
                    if block is not None:
                        self._block_stack.pop()

            def _visit_in(self, node, *, block=None, role=None):
                if node is None:
                    return
                with self._ctx(block=block, role=role):
                    self.visit(node)

            def _emit(self, node, refname):
                hit_meta = DHitMeta(
                    name=node.id,
                    refname=_canonicalize_refname(refname),
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    end_lineno=getattr(node, "end_lineno", None),
                    end_col_offset=getattr(node, "end_col_offset", None),
                    block=self._block_stack[-1],
                    role=self._role_stack[-1],
                    scope_refname=_format_scope_refname(self._stack[-1] if self._stack else None),
                    source_line=source_lines[node.lineno - 1] if 1 <= node.lineno <= n_lines else None,
                )
                self.hit_meta_by_line[node.lineno].append(hit_meta)

            def visit(self, node):
                if _is_scope(node):
                    self._stack.append(scope_infos[node])
                    super().visit(node)
                    self._stack.pop()
                else:
                    super().visit(node)

            def _visit_function_like(self, node):
                for dec in getattr(node, "decorator_list", ()):
                    self._visit_in(dec, block="decorator", role="expr")
                args = node.args
                for arg in itertools.chain(getattr(args, "posonlyargs", ()), args.args, args.kwonlyargs):
                    if getattr(arg, "annotation", None):
                        self._visit_in(arg.annotation, block="head", role="annotation")
                if getattr(args, "vararg", None) and args.vararg.annotation:
                    self._visit_in(args.vararg.annotation, block="head", role="annotation")
                if getattr(args, "kwarg", None) and args.kwarg.annotation:
                    self._visit_in(args.kwarg.annotation, block="head", role="annotation")
                for default in getattr(args, "defaults", ()):
                    self._visit_in(default, block="head", role="default")
                for default in getattr(args, "kw_defaults", ()):
                    if default is not None:
                        self._visit_in(default, block="head", role="default")
                if getattr(node, "returns", None):
                    self._visit_in(node.returns, block="head", role="returns")
                for stmt in node.body:
                    self._visit_in(stmt, block="body", role="expr")

            def visit_FunctionDef(self, node):
                self._visit_function_like(node)

            def visit_AsyncFunctionDef(self, node):
                self._visit_function_like(node)

            def visit_ClassDef(self, node):
                for dec in node.decorator_list:
                    self._visit_in(dec, block="decorator", role="expr")
                for base in node.bases:
                    self._visit_in(base, block="head", role="base")
                for kw in node.keywords:
                    role = "metaclass" if kw.arg == "metaclass" else "keyword"
                    self._visit_in(kw.value, block="head", role=role)
                for stmt in node.body:
                    self._visit_in(stmt, block="body", role="expr")

            def visit_TypeAlias(self, node):
                self._visit_in(node.value, block="head", role="annotation")

            def visit_AnnAssign(self, node):
                if node.annotation:
                    self._visit_in(node.annotation, role="annotation")
                if node.value:
                    self._visit_in(node.value)

            def visit_Attribute(self, node):
                attrs = []
                base = node
                while isinstance(base, ast.Attribute):
                    attrs.append(base.attr)
                    base = base.value
                if isinstance(base, ast.Name) and isinstance(base.ctx, ast.Load) and self._stack:
                    cur = self._stack[-1]
                    if base.id in cur.final_locals():
                        return
                    refname = _resolve_refname(base.id, cur)
                    if refname:
                        tail = ".".join(reversed(attrs))
                        self._emit(base, f"{refname}.{tail}")
                        self._name_nodes_emitted.add(id(base))
                    return
                self.visit(node.value)

            def visit_Name(self, node):
                if id(node) in getattr(self, "_name_nodes_emitted", ()):
                    return
                if self._stack and isinstance(node.ctx, ast.Load):
                    cur = self._stack[-1]
                    locals_final = cur.final_locals()
                    if node.id not in locals_final:
                        refname = _resolve_refname(node.id, cur)
                        if refname:
                            self._emit(node, refname)

        collector = UseCollector()
        collector.visit(tree)

        if return_matrix:
            matrix = [[] for _ in range(n_lines)]
            if return_meta:
                for ln, items in collector.hit_meta_by_line.items():
                    if 1 <= ln <= n_lines:
                        matrix[ln - 1] = sorted(items, key=lambda d: d.col_offset)
            else:
                for ln, items in collector.hit_meta_by_line.items():
                    if 1 <= ln <= n_lines:
                        items_sorted = sorted(items, key=lambda d: d.col_offset)
                        refnames = []
                        for d in items_sorted:
                            refnames.append(d.refname)
                        if unique_only:
                            refnames = list(dict.fromkeys(refnames))
                        matrix[ln - 1] = refnames
            return matrix
        else:
            if return_meta:
                meta_items = []
                for ln, items in collector.hit_meta_by_line.items():
                    if 1 <= ln <= n_lines:
                        meta_items.extend(items)
                return sorted(meta_items, key=lambda d: (d.lineno, d.col_offset))
            else:
                refnames = []
                for ln, items in collector.hit_meta_by_line.items():
                    if 1 <= ln <= n_lines:
                        items_sorted = sorted(items, key=lambda d: d.col_offset)
                        for d in items_sorted:
                            refnames.append(d.refname)
                if unique_only:
                    refnames = list(dict.fromkeys(refnames))
                return refnames

    def index_module(self, module: tp.ModuleLike) -> None:
        """Index the specified module if not already indexed.

        Args:
            module (ModuleLike): Module reference name or object.

        Returns:
            None
        """
        if module.__name__ in self.dependencies:
            return
        module = resolve_module(module)
        dependencies = self.get_dependencies(
            module,
            expand_star_imports=self.expand_star_imports,
            return_meta=True,
        )
        self.dependencies[module.__name__] = dependencies

    def get_dependency_scopes(
        self,
        module: tp.ModuleType,
        incl_unreachable: tp.Optional[bool] = None,
        unique_only: bool = True,
    ) -> tp.List[str]:
        """Return dependency scopes in the specified module.

        Args:
            module (ModuleLike): Module reference name or object.
            incl_unreachable (Optional[bool]): Whether to allow visiting reference names from unreachable scopes.

                Defaults to `RefIndex.incl_unreachable`.
            unique_only (bool): If True, return only unique reference names.

        Returns:
            List[str]: List of scope reference names.
        """
        if incl_unreachable is None:
            incl_unreachable = self.incl_unreachable
        self.index_module(module)
        scopes = []
        for dependency in self.dependencies[module.__name__]:
            if not incl_unreachable and "::" in dependency.scope_refname:
                continue
            scopes.append(dependency.scope_refname)
        if unique_only:
            return list(dict.fromkeys(scopes))
        return scopes

    def get_scope_dependencies(
        self,
        scope_refname: str,
        module: tp.Optional[tp.ModuleLike] = None,
        resolve: bool = True,
        relation: str = "all",
        block: tp.Optional[str] = None,
        role: tp.Optional[str] = None,
        return_meta: bool = True,
        unique_only: bool = True,
    ) -> tp.List[tp.MaybeList[tp.Union[str, DHitMeta]]]:
        """Return dependencies in the specified scope.

        Args:
            scope_refname (str): Reference name of the scope.
            module (ModuleLike): Module reference name or object.
            resolve (bool): Whether to resolve the reference to an actual object.
            relation (str): Relation between references. One of:

                * "direct": References that are connected directly.
                * "nested": References that are connected through other references.
                * "all": Both direct and nested references.
            block (Optional[str]): Block to filter by (e.g., "decorator", "head", "body").
            role (Optional[str]): Syntactic role to filter by (e.g., "expr", "annotation", "default").
            return_meta (bool): If True, returns detailed dependency hit metadata of type `DHitMeta`.

                If False, returns only reference name strings.
            unique_only (bool): If True, return only unique reference names (applies only when `return_meta` is False).

        Returns:
            List[MaybeList[Union[str, DHitMeta]]]: List of reference names or `DHitMeta` instances.
        """
        scope_refname = ensure_refname(scope_refname, module=module, resolve=resolve)
        if module is None:
            module, _ = split_refname(scope_refname, raise_error=True)
        self.index_module(module)

        all_scopes = self.get_dependency_scopes(module, incl_unreachable=True)
        if relation.lower() == "direct":
            scopes = [scope_refname]
        elif relation.lower() == "nested":
            scopes = [s for s in all_scopes if s.startswith(scope_refname + ".")]
        elif relation.lower() == "all":
            scopes = [scope_refname] + [s for s in all_scopes if s.startswith(scope_refname + ".")]
        else:
            raise ValueError(f"Invalid relation: {relation!r}")

        dependencies = []
        for scope in scopes:
            for dependency in self.dependencies[module.__name__]:
                if dependency.scope_refname != scope:
                    continue
                if block is not None and dependency.block != block:
                    continue
                if role is not None and dependency.role != role:
                    continue
                if (
                    dependency.refname == scope_refname
                    or dependency.refname.startswith(scope_refname + ".")
                    or dependency.refname.startswith(scope_refname + "::")
                ):
                    continue
                dependencies.append(dependency)

        if not return_meta:
            refnames = [dependency.refname for dependency in dependencies]
            if unique_only:
                refnames = list(dict.fromkeys(refnames))
            return refnames
        return dependencies

    def is_kind_container(self, kind: tp.Optional[str]) -> bool:
        """Check if the specified kind is a container kind.

        Args:
            kind (Optional[str]): Kind to check.

        Returns:
            bool: True if the kind is a container kind, False otherwise.
        """
        if self.container_kinds is None:
            return True
        if kind is None:
            return False
        return kind in self.container_kinds

    def get_info(
        self,
        obj: tp.Any,
        module: tp.Optional[tp.ModuleLike] = None,
        resolve: bool = True,
        incl_relations: bool = True,
        incl_private: tp.Optional[bool] = None,
        **kwargs,
    ) -> RefInfo:
        """Get information about the specified object.

        Args:
            obj (Any): Object from which to extract the reference name.

                If a string or tuple is provided, it is treated as a reference name.
            module (Optional[ModuleLike]): Module context used in reference resolution.
            resolve (bool): Whether to resolve the reference to an actual object.
            incl_relations (bool): If True, include direct and nested members, bases, and dependencies.
            incl_private (Optional[bool]): Whether to include private members.

                Defaults to `RefIndex.incl_private`.
            **kwargs: Keyword arguments for `RefIndex.get_scope_dependencies`.

        Returns:
            RefInfo: `RefInfo` instance containing information about the object.
        """
        refname, module, qualname = ensure_refname(obj, module=module, resolve=resolve, return_parts=True)
        obj = get_refname_obj(refname, raise_error=False)
        if incl_private is None:
            incl_private = self.incl_private

        ref_info = {}
        ref_info["refname"] = refname
        if qualname is not None:
            ref_info["qualname"] = qualname
        if module is not None:
            ref_info["module"] = module.__name__
        if obj is not None:
            ref_info["type"], ref_info["kind"] = get_type_and_kind(obj)
        refname_parts = refname.split(".")
        if len(refname_parts) > 1:
            ref_info["container"] = ".".join(refname_parts[:-1])
        if incl_relations:
            if obj is not None and self.is_kind_container(ref_info.get("kind", None)):
                attr_meta = get_attrs(obj, incl_private=incl_private, return_meta=True)
                direct_members = []
                for m in attr_meta:
                    if m.refname is not None and m.is_own:
                        direct_members.append(m.refname)
                if direct_members:
                    ref_info["direct_members"] = direct_members
                direct_members_set = set(direct_members)
                nested_members = []
                for m in attr_meta:
                    if m.refname is not None and m.refname not in direct_members_set:
                        nested_members.append(m.refname)
                if nested_members:
                    ref_info["nested_members"] = nested_members
                cls = obj if inspect.isclass(obj) else type(obj)
                direct_bases = []
                for c in cls.__bases__:
                    if c is cls or c is object:
                        continue
                    r = ensure_refname(c, can_be_refname=False, raise_error=False)
                    if r is not None:
                        direct_bases.append(r)
                if direct_bases:
                    ref_info["direct_bases"] = direct_bases
                nested_bases = []
                for c in inspect.getmro(cls):
                    if c is cls or c is object:
                        continue
                    if c in cls.__bases__:
                        continue
                    r = ensure_refname(c, can_be_refname=False, raise_error=False)
                    if r is not None:
                        nested_bases.append(r)
                if nested_bases:
                    ref_info["nested_bases"] = nested_bases
            try:
                direct_dependencies = self.get_scope_dependencies(
                    refname,
                    module=module,
                    resolve=False,
                    relation="direct",
                    return_meta=False,
                    **kwargs,
                )
                if direct_dependencies:
                    ref_info["direct_dependencies"] = direct_dependencies
            except (ModuleNotFoundError, FileNotFoundError, ReferenceResolutionError, SyntaxError):
                pass
            try:
                nested_dependencies = self.get_scope_dependencies(
                    refname,
                    module=module,
                    resolve=False,
                    relation="nested",
                    return_meta=False,
                    **kwargs,
                )
                if nested_dependencies:
                    ref_info["nested_dependencies"] = nested_dependencies
            except (ModuleNotFoundError, FileNotFoundError, ReferenceResolutionError, SyntaxError):
                pass
            ref_info["is_shallow"] = False
        else:
            ref_info["is_shallow"] = True
        return RefInfo(**ref_info)

    def build_graph(
        self,
        obj: tp.Optional[tp.Any] = None,
        module: tp.Optional[tp.ModuleLike] = None,
        resolve: bool = True,
        *,
        own_only: bool = True,
        missing: str = "shallow",
        traversal: str = "BFS",
        max_depth: tp.Optional[int] = None,
        visit_containers: bool = True,
        incl_keys: tp.Optional[tp.Set[str]] = None,
        ref_graph_cls: tp.Optional[tp.Type[RefGraph]] = None,
        ref_graph_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> RefGraph:
        """Traverse the graph of reference names reachable from the object and build a `RefGraph`.

        Starting at the reference name, repeatedly calls `RefIndex.get_info` and then visits every
        reference name found under the selected keys of that information dictionary.
        Stops when there are no new reference names to visit or when `max_depth` is reached.

        Args:
            obj (Optional[Any]): Object from which to extract the reference name.

                If a string or tuple is provided, it is treated as a reference name.
                If None, starts from the module specified by `module`.
            module (Optional[ModuleLike]): Module context used in reference resolution.
            resolve (bool): Whether to resolve the reference to an actual object.
            own_only (bool): If True, only visit reference names defined in the same object
                as the starting reference name.
            missing (str): How to handle references that are seen in relation lists
                but are not traversed due to `own_only` or `max_depth`. One of:

                * "keep": Keep their names in relation lists as-is.
                * "shallow": Create shallow `RefInfo` instances for them.
                * "drop": Remove them from relation lists entirely.
            traversal (str): Traversal strategy.

                * "DFS" for depth-first search.
                * "BFS" for breadth-first search.
            max_depth (Optional[int]): Limit recursion to the specified depth (0 disables traversal, None = unlimited).
            visit_containers (bool): If True, visit the container of each reference name regardless of `own_only`.
            incl_keys (Optional[Set[str]]): Which fields of `RefInfo` to traverse from.
            ref_graph_cls (Optional[Type[RefGraph]]): `RefGraph` subclass to use for building the graph.
            ref_graph_kwargs (KwargsLike): Keyword arguments for `RefGraph`.

                By default, sets `root` to the starting reference name.
            **kwargs: Keyword arguments for `RefIndex.get_info`.

        Returns:
            RefGraph: `RefGraph` instance representing the traversed graph.
        """
        if obj is None:
            if module is None:
                obj = "vectorbtpro"
            else:
                obj = module
            module = None
        refname = ensure_refname(obj, module=module, resolve=resolve)
        if incl_keys is None:
            incl_keys = {
                "container",
                "direct_members",
                "nested_members",
                "direct_bases",
                "nested_bases",
                "direct_dependencies",
                "nested_dependencies",
            }

        def _iter_children(info):
            if "container" in incl_keys and info.container:
                yield info.container, "container"
            if "direct_members" in incl_keys:
                for m in info.direct_members:
                    yield m, "direct_members"
            if "nested_members" in incl_keys:
                for m in info.nested_members:
                    yield m, "nested_members"
            if "direct_bases" in incl_keys:
                for b in info.direct_bases:
                    yield b, "direct_bases"
            if "nested_bases" in incl_keys:
                for b in info.nested_bases:
                    yield b, "nested_bases"
            if "direct_dependencies" in incl_keys:
                for d in info.direct_dependencies:
                    yield d, "direct_dependencies"
            if "nested_dependencies" in incl_keys:
                for d in info.nested_dependencies:
                    yield d, "nested_dependencies"

        def _passes_base_filters(name, is_container=False):
            if self.incl_predicate is not None and self.incl_predicate(name):
                return True
            if not self.incl_builtins and (name == "builtins" or name.startswith("builtins.")):
                return False
            if not self.incl_unreachable and "::" in name:
                return False
            if not is_container and not self.incl_private and not checks.is_public_name(name.split(".")[-1]):
                return False
            if self.incl_modules and not any(
                name == mod.__name__ or name.startswith(mod.__name__ + ".") for mod in self.incl_modules
            ):
                return False
            if self.excl_modules and any(
                name == mod.__name__ or name.startswith(mod.__name__ + ".") for mod in self.excl_modules
            ):
                return False
            if self.excl_predicate is not None and self.excl_predicate(name):
                return False
            return True

        def _passes_own_only(name):
            if not own_only:
                return True
            return name == refname or name.startswith(refname + ".") or name.startswith(refname + "::")

        start = refname
        to_visit = deque()
        to_visit.append((start, 0, True))
        ref_by_name = {}

        while to_visit:
            if traversal.upper() == "DFS":
                current, depth, incl_relations = to_visit.pop()
            elif traversal.upper() == "BFS":
                current, depth, incl_relations = to_visit.popleft()
            else:
                raise ValueError(f"Invalid traversal: {traversal!r}")
            existing = ref_by_name.get(current, None)
            if existing is not None:
                if not existing.is_shallow:
                    continue
                if not incl_relations:
                    continue
            ref_info = self.get_info(
                current,
                resolve=False,
                incl_relations=incl_relations,
                **kwargs,
            )
            ref_by_name[current] = ref_info
            next_depth = depth + 1

            for child, relation_key in _iter_children(ref_info):
                is_container = relation_key == "container"
                if not incl_relations and not (is_container and visit_containers):
                    continue
                if not _passes_base_filters(child, is_container=is_container):
                    continue
                within_depth = (max_depth is None) or (next_depth <= max_depth)
                existing_child = ref_by_name.get(child, None)
                if _passes_own_only(child):
                    if within_depth and (existing_child is None or existing_child.is_shallow):
                        to_visit.append((child, next_depth, True))
                    else:
                        if missing.lower() == "shallow" and existing_child is None:
                            to_visit.append((child, next_depth, False))
                    continue
                if is_container and visit_containers:
                    if within_depth and (existing_child is None or existing_child.is_shallow):
                        to_visit.append((child, next_depth, False))
                    continue
                if missing.lower() == "shallow":
                    if existing_child is None:
                        to_visit.append((child, next_depth, False))
                elif missing.lower() == "keep":
                    pass
                elif missing.lower() == "drop":
                    pass
                else:
                    raise ValueError(f"Invalid missing mode: {missing!r}")

        def _filter_targets(targets):
            out = []
            for t in targets:
                if not _passes_base_filters(t):
                    continue
                if missing.lower() == "drop" and t not in ref_by_name:
                    continue
                out.append(t)
            return out

        new_ref_by_name = {}

        for name, info in ref_by_name.items():
            new_container = info.container
            if new_container is not None:
                if (not _passes_base_filters(new_container, is_container=True)) or (
                    missing.lower() == "drop" and new_container not in ref_by_name
                ):
                    new_container = None

            new_info = info.replace(
                container=new_container,
                direct_members=_filter_targets(info.direct_members),
                nested_members=_filter_targets(info.nested_members),
                direct_bases=_filter_targets(info.direct_bases),
                nested_bases=_filter_targets(info.nested_bases),
                direct_dependencies=_filter_targets(info.direct_dependencies),
                nested_dependencies=_filter_targets(info.nested_dependencies),
            )
            new_ref_by_name[name] = new_info

        ref_infos = list(new_ref_by_name.values())
        if ref_graph_cls is None:
            ref_graph_cls = RefGraph
        if ref_graph_kwargs is None:
            ref_graph_kwargs = {}
        if "root" not in ref_graph_kwargs:
            ref_graph_kwargs["root"] = refname
        return ref_graph_cls.from_ref_infos(ref_infos, **ref_graph_kwargs)
