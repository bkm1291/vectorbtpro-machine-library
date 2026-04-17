# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for modules."""

import importlib
import importlib.util
import inspect
import pkgutil
import sys
from pathlib import Path
from types import ModuleType

from vectorbtpro import _typing as tp
from vectorbtpro._opt_deps import opt_dep_config
from vectorbtpro.utils.config import HybridConfig
from vectorbtpro.utils.warnings_ import warn

__all__ = [
    "import_module_from_path",
]

__pdoc__ = {}

package_shortcut_config = HybridConfig(
    dict(
        vbt="vectorbtpro",
        pd="pandas",
        np="numpy",
        nb="numba",
    )
)
"""_"""

__pdoc__[
    "package_shortcut_config"
] = f"""Config for package shortcuts.

```python
{package_shortcut_config.prettify_doc()}
```
"""


def safe_unwrap(obj: tp.Any, max_depth: int = 64) -> tp.Any:
    """Safely unwrap decorated functions up to a maximum depth.

    Args:
        obj (Any): Object to unwrap.
        max_depth (int): Maximum depth to unwrap.

    Returns:
        Any: Unwrapped object.
    """
    seen = set()
    cur = obj
    for _ in range(max_depth):
        oid = id(cur)
        if oid in seen:
            break
        seen.add(oid)
        nxt = None
        try:
            nxt = inspect.getattr_static(cur, "__wrapped__")
        except AttributeError:
            pass
        if nxt is None:
            break
        cur = nxt
    return cur


def get_module(obj: tp.Any) -> tp.Optional[ModuleType]:
    """Return the module in which the given object is defined.

    Args:
        obj (Any): Object whose module is to be obtained.

    Returns:
        Optional[ModuleType]: Module where the object is defined; None if the module cannot be determined.
    """
    from vectorbtpro.utils.attr_ import get_attr

    if isinstance(obj, ModuleType):
        return obj
    target = safe_unwrap(obj)
    modname = get_attr(target, "__module__", None)
    if modname is None:
        modname = get_attr(type(target), "__module__", None)
    if modname is None:
        return None
    mod = sys.modules.get(modname)
    if mod is not None:
        return mod
    try:
        return import_module(modname)
    except Exception:
        return None


def import_module(module_name: str) -> ModuleType:
    """Import and return the module with the specified name.

    Args:
        module_name (str): Name of the module to import.

    Returns:
        ModuleType: Imported module.
    """
    mod = importlib.import_module(module_name)
    if not isinstance(mod, ModuleType):
        raise ImportError(f"Expected a module for {module_name!r}, got {type(mod)!r}")
    return mod


def resolve_module(module: tp.ModuleLike) -> ModuleType:
    """Return the module object for the given module name or module.

    Args:
        module (ModuleLike): Module reference name or object.

    Returns:
        ModuleType: Resolved module object.
    """
    from vectorbtpro.utils.refs import get_obj

    if isinstance(module, str):
        try:
            return import_module(module)
        except ImportError:
            pass
        obj = get_obj(module, raise_error=False)
        if obj is None:
            raise ModuleNotFoundError(f"Module {module!r} not found")
        if not isinstance(obj, ModuleType):
            raise TypeError(f"Expected module or module name, got {type(obj)}")
        return obj
    if not isinstance(module, ModuleType):
        raise TypeError(f"Expected module or module name, got {type(module)}")
    return module


def is_from_module(obj: tp.Any, module: tp.ModuleLike) -> bool:
    """Return True if the provided object is defined in the specified module; otherwise, return False.

    Args:
        obj (Any): Object to verify.
        module (ModuleLike): Module reference name or object.

    Returns:
        bool: True if the object is from the specified module; otherwise, False.
    """
    mod = get_module(obj)
    module = resolve_module(module)
    return mod is None or mod.__name__ == module.__name__


def list_module_keys(
    module: tp.ModuleLike,
    whitelist: tp.Optional[tp.List[str]] = None,
    blacklist: tp.Optional[tp.List[str]] = None,
) -> tp.List[str]:
    """Return a list of names for all public functions and classes in the specified module.

    Args:
        module (ModuleLike): Module reference name or object.
        whitelist (Optional[List[str]]): Additional names to include.
        blacklist (Optional[List[str]]): Names to exclude from the list.

    Returns:
        List[str]: List of public function and class names.
    """
    if whitelist is None:
        whitelist = []
    if blacklist is None:
        blacklist = []
    if isinstance(module, str):
        module = sys.modules[module]
    return [
        name
        for name, obj in inspect.getmembers(module)
        if (
            not name.startswith("_")
            and is_from_module(obj, module)
            and ((inspect.isroutine(obj) and callable(obj)) or inspect.isclass(obj))
            and name not in blacklist
        )
        or name in whitelist
    ]


def search_package(
    package: tp.ModuleLike,
    match_func: tp.Callable,
    blacklist: tp.Optional[tp.Sequence[str]] = None,
    path_attrs: bool = False,
    return_first: bool = False,
    _visited: tp.Optional[tp.Set[str]] = None,
) -> tp.Union[None, tp.Any, tp.Dict[str, tp.Any]]:
    """Search for objects in a package that satisfy a given condition.

    The matching function should accept the name of an object and the object itself, and return a boolean.

    Args:
        package (ModuleLike): Module reference name or object.
        match_func (Callable): Function that takes an object's name and the object, returning a boolean.
        blacklist (Optional[Sequence[str]]): Names to exclude from the search.
        path_attrs (bool): If True, use reference names for object attributes.
        return_first (bool): If True, return the first matching object.

    Returns:
        Union[None, Any, Dict[str, Any]]: If `return_first` is True, returns the first matching object or None;
            otherwise, returns a dictionary of matching objects.
    """
    if blacklist is None:
        blacklist = []
    if _visited is None:
        _visited = set()
    results = {}

    if isinstance(package, str):
        package = import_module(package)
    if package.__name__ not in _visited:
        _visited.add(package.__name__)
        for attr in dir(package):
            if path_attrs:
                path_attr = package.__name__ + "." + attr
            else:
                path_attr = attr
            if not attr.startswith("_") and match_func(path_attr, getattr(package, attr)):
                if return_first:
                    return getattr(package, attr)
                results[path_attr] = getattr(package, attr)

    for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if ".".join(name.split(".")[:-1]) != package.__name__:
            continue
        try:
            if name in _visited or name in blacklist:
                continue
            _visited.add(name)
            module = import_module(name)
            for attr in dir(module):
                if path_attrs:
                    path_attr = module.__name__ + "." + attr
                else:
                    path_attr = attr
                if not attr.startswith("_") and match_func(path_attr, getattr(module, attr)):
                    if return_first:
                        return getattr(module, attr)
                    results[path_attr] = getattr(module, attr)
            if is_pkg:
                results.update(
                    search_package(
                        name,
                        match_func,
                        blacklist=blacklist,
                        path_attrs=path_attrs,
                        _visited=_visited,
                    )
                )
        except (ModuleNotFoundError, ImportError):
            pass
    if return_first:
        return None
    return results


def check_installed(pkg_name: str) -> bool:
    """Return True if the package with the specified name is installed; otherwise, return False.

    Args:
        pkg_name (str): Name of the package.

    Returns:
        bool: True if the package is installed; otherwise, False.
    """
    return importlib.util.find_spec(pkg_name) is not None


def get_installed_overview() -> tp.Dict[str, bool]:
    """Return a dictionary mapping package names from `opt_dep_config` to their installation status.

    Returns:
        Dict[str, bool]: Mapping where keys are package names and values indicate installation status.
    """
    return {pkg_name: check_installed(pkg_name) for pkg_name in opt_dep_config.keys()}


def get_package_meta(pkg_name: str) -> dict:
    """Return the metadata dictionary for the specified package from `opt_dep_config`.

    Args:
        pkg_name (str): Name of the package.

    Returns:
        dict: Dictionary containing metadata such as 'dist_name', 'version', and 'link'.
    """
    if pkg_name not in opt_dep_config:
        raise KeyError(f"Package {pkg_name!r} not found in opt_dep_config")
    dist_name = opt_dep_config[pkg_name].get("dist_name", pkg_name)
    version = opt_dep_config[pkg_name].get("version", "")
    link = opt_dep_config[pkg_name].get("link", f"https://pypi.org/project/{dist_name}/")
    return dict(dist_name=dist_name, version=version, link=link)


def assert_can_import(pkg_name: str) -> None:
    """Assert that the specified package can be imported.

    The package must be listed in `opt_dep_config`. An `ImportError` is raised if the package
    is not installed or the installed version is incompatible.

    Args:
        pkg_name (str): Name of the package.

    Returns:
        None
    """
    from importlib.metadata import version as get_version

    metadata = get_package_meta(pkg_name)
    dist_name = metadata["dist_name"]
    version = version_str = metadata["version"]
    link = metadata["link"]
    if not check_installed(pkg_name):
        raise ImportError(f"Please install {dist_name}{version_str}. See {link}.")
    if version != "":
        actual_version_parts = get_version(dist_name).split(".")
        actual_version_parts = map(lambda x: x if x.isnumeric() else f"'{x}'", actual_version_parts)
        actual_version = "(" + ",".join(actual_version_parts) + ")"
        if version[0].isdigit():
            operator = "=="
        else:
            operator = version[:2]
            version_parts = version[2:].split(".")
            version_parts = map(lambda x: x if x.isnumeric() else f"'{x}'", version_parts)
            version = "(" + ",".join(version_parts) + ")"
        if not eval(f"{actual_version} {operator} {version}"):
            raise ImportError(f"Please install {dist_name}{version_str}. See {link}.")


def assert_can_import_any(*pkg_names: str) -> None:
    """Assert that at least one of the specified packages can be imported.

    Packages must be listed in `opt_dep_config`. If none of the packages can be imported,
    an ImportError is raised.

    Args:
        *pkg_names (str): Additional package names for checking import.

    Returns:
        None
    """
    if len(pkg_names) == 1:
        return assert_can_import(pkg_names[0])
    for pkg_name in pkg_names:
        try:
            return assert_can_import(pkg_name)
        except ImportError:
            pass
    requirements = []
    for pkg_name in pkg_names:
        metadata = get_package_meta(pkg_name)
        dist_name = metadata["dist_name"]
        version_str = metadata["version"]
        link = metadata["link"]
        requirements.append(f"{dist_name}{version_str} - {link}")
    raise ImportError(f"Please install any of " + ", ".join(requirements))


def warn_cannot_import(pkg_name: str) -> bool:
    """Warn if the specified package cannot be imported.

    The package must be listed in `opt_dep_config`. If the package cannot be imported,
    a warning is issued and True is returned; otherwise, False is returned.

    Args:
        pkg_name (str): Name of the package.

    Returns:
        bool: True if the package cannot be imported; otherwise, False.
    """
    try:
        assert_can_import(pkg_name)
        return False
    except ImportError as e:
        warn(str(e))
        return True


def import_module_from_path(module_path: tp.PathLike, reload: bool = False) -> ModuleType:
    """Import a Python module from a specified file path.

    Args:
        module_path (PathLike): File system path to the module.
        reload (bool): Whether to force reloading if the module is already imported.

    Returns:
        ModuleType: Imported module.
    """
    module_path = Path(module_path)
    spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path.resolve()))
    module = importlib.util.module_from_spec(spec)
    if module.__name__ in sys.modules and not reload:
        return sys.modules[module.__name__]
    spec.loader.exec_module(module)
    sys.modules[module.__name__] = module
    return module
