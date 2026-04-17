# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing functions to compress, decompress, and manage pickling configurations.

!!! info
    For default settings, see `vectorbtpro._settings.pickling`.
"""

import ast
import base64
import datetime
import hashlib
import io
import zipfile
from functools import cached_property as cachedproperty
from graphlib import TopologicalSorter
from pathlib import Path
from types import MethodType

import humanize
import numpy as np

import vectorbtpro as vbt
from vectorbtpro import _typing as tp
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.checks import Comparable, is_hashable, is_deep_equal
from vectorbtpro.utils.formatting import Prettified, prettify_dict
from vectorbtpro.utils.path_ import check_mkdir

PickleableT = tp.TypeVar("PickleableT", bound="Pickleable")

__all__ = [
    "dumps",
    "loads",
    "save_bytes",
    "load_bytes",
    "save",
    "load",
    "RecState",
    "RecInfo",
    "Pickleable",
    "pdict",
    "get_id_from_class",
    "get_class_from_id",
]


def get_serialization_extensions(cls_name: tp.Optional[str] = None) -> tp.Set[str]:
    """Return all supported serialization extensions.

    !!! info
        For default settings, see `vectorbtpro._settings.pickling`.

    Args:
        cls_name (Optional[str]): Class name to retrieve specific serialization extensions.

            If omitted, returns a union of all serialization extensions.

    Returns:
        Set[str]: Set of serialization file extensions.
    """
    from vectorbtpro._settings import settings

    pickling_cfg = settings["pickling"]

    if cls_name is None:
        return set.union(*pickling_cfg["extensions"]["serialization"].values())
    return pickling_cfg["extensions"]["serialization"][cls_name]


def get_compression_extensions(cls_name: tp.Optional[str] = None) -> tp.Set[str]:
    """Return all supported compression extensions.

    !!! info
        For default settings, see `vectorbtpro._settings.pickling`.

    Args:
        cls_name (Optional[str]): Class name to retrieve specific compression extensions.

            If omitted, returns a union of all compression extensions.

    Returns:
        Set[str]: Set of compression file extensions.
    """
    from vectorbtpro._settings import settings

    pickling_cfg = settings["pickling"]

    if cls_name is None:
        return set.union(*pickling_cfg["extensions"]["compression"].values())
    return pickling_cfg["extensions"]["compression"][cls_name]


def compress(
    bytes_: bytes,
    compression: tp.CompressionLike = None,
    file_name: tp.Optional[str] = None,
    **compress_kwargs,
) -> bytes:
    """Compress given bytes using the specified compression format.

    !!! info
        For default settings, see `extensions.compression` in `vectorbtpro._settings.pickling`.

    Args:
        bytes_ (bytes): Byte stream to be compressed.
        compression (CompressionLike): Compression algorithm.

            If `True`, uses the default compression algorithm from settings.
            For options, see `extensions.compression` in `vectorbtpro._settings.pickling`.
        file_name (Optional[str]): Name of the file in the archive when using archive-based compression.
        **compress_kwargs: Keyword arguments for the compression function
            of the compression package.

    Returns:
        bytes: Compressed data.
    """
    from vectorbtpro.utils.module_ import assert_can_import, assert_can_import_any, check_installed

    if isinstance(compression, bool) and compression:
        from vectorbtpro._settings import settings

        pickling_cfg = settings["pickling"]

        compression = pickling_cfg["compression"]
        if compression is None:
            raise ValueError("Set default compression in settings")
    if compression not in (None, False):
        if compression.lower() in get_compression_extensions("zip"):
            zip_buffer = io.BytesIO()
            if "compression" not in compress_kwargs:
                compress_kwargs["compression"] = zipfile.ZIP_DEFLATED
            with zipfile.ZipFile(zip_buffer, "w", **compress_kwargs) as zip_file:
                if file_name is None:
                    file_name = "data.bin"
                zip_file.writestr(file_name, bytes_)
            bytes_ = zip_buffer.getvalue()
        elif compression.lower() in get_compression_extensions("bz2"):
            import bz2

            bytes_ = bz2.compress(bytes_, **compress_kwargs)
        elif compression.lower() in get_compression_extensions("gzip"):
            import gzip

            bytes_ = gzip.compress(bytes_, **compress_kwargs)
        elif compression.lower() in get_compression_extensions("lzma"):
            import lzma

            bytes_ = lzma.compress(bytes_, **compress_kwargs)
        elif compression.lower() in get_compression_extensions("lz4"):
            assert_can_import("lz4")

            import lz4.frame

            bytes_ = lz4.frame.compress(bytes_, return_bytearray=True, **compress_kwargs)
        elif compression.lower() in get_compression_extensions("blosc1"):
            assert_can_import("blosc")

            import blosc

            bytes_ = blosc.compress(bytes_, **compress_kwargs)
        elif compression.lower() in get_compression_extensions("blosc2"):
            assert_can_import("blosc2")

            import blosc2

            if "_ignore_multiple_size" not in compress_kwargs:
                compress_kwargs["_ignore_multiple_size"] = True
            bytes_ = blosc2.compress(bytes_, **compress_kwargs)
        elif compression.lower() in get_compression_extensions("blosc"):
            assert_can_import_any("blosc2", "blosc")

            if check_installed("blosc2"):
                import blosc2

                if "_ignore_multiple_size" not in compress_kwargs:
                    compress_kwargs["_ignore_multiple_size"] = True
                bytes_ = blosc2.compress(bytes_, **compress_kwargs)
            else:
                import blosc

                bytes_ = blosc.compress(bytes_, **compress_kwargs)
        else:
            raise ValueError(f"Invalid compression: {compression!r}")
    return bytes_


def decompress(
    bytes_: bytes,
    compression: tp.CompressionLike = None,
    file_name: tp.Optional[str] = None,
    **decompress_kwargs,
) -> bytes:
    """Decompress given bytes using the specified compression format.

    !!! info
        For default settings, see `extensions.compression` in `vectorbtpro._settings.pickling`.

    Args:
        bytes_ (bytes): Compressed byte stream to be decompressed.
        compression (CompressionLike): Compression algorithm.

            See `compress`.
        file_name (Optional[str]): Name of the file in the archive when using archive-based compression.
        **decompress_kwargs: Keyword arguments for the decompression function
            of the compression package.

    Returns:
        bytes: Decompressed data.
    """
    from vectorbtpro.utils.module_ import assert_can_import, assert_can_import_any, check_installed

    if isinstance(compression, bool) and compression:
        from vectorbtpro._settings import settings

        pickling_cfg = settings["pickling"]

        compression = pickling_cfg["compression"]
        if compression is None:
            raise ValueError("Set default compression in settings")
    if compression not in (None, False):
        if compression.lower() in get_compression_extensions("zip"):
            zip_buffer = io.BytesIO(bytes_)
            with zipfile.ZipFile(zip_buffer, "r", **decompress_kwargs) as zip_file:
                namelist = zip_file.namelist()
                if len(namelist) == 0:
                    raise ValueError("ZIP archive is empty")
                if file_name is not None:
                    if file_name not in namelist:
                        raise FileNotFoundError(f"{file_name!r} not found in the ZIP archive")
                else:
                    if len(namelist) == 1:
                        file_name = namelist[0]
                    else:
                        raise ValueError("Multiple files exist in the ZIP archive. Please specify a filename.")
                with zip_file.open(file_name) as file:
                    bytes_ = file.read()
        elif compression.lower() in get_compression_extensions("bz2"):
            import bz2

            bytes_ = bz2.decompress(bytes_, **decompress_kwargs)
        elif compression.lower() in get_compression_extensions("gzip"):
            import gzip

            bytes_ = gzip.decompress(bytes_, **decompress_kwargs)
        elif compression.lower() in get_compression_extensions("lzma"):
            import lzma

            bytes_ = lzma.decompress(bytes_, **decompress_kwargs)
        elif compression.lower() in get_compression_extensions("lz4"):
            assert_can_import("lz4")

            import lz4.frame

            bytes_ = lz4.frame.decompress(bytes_, return_bytearray=True, **decompress_kwargs)
        elif compression.lower() in get_compression_extensions("blosc1"):
            assert_can_import("blosc")

            import blosc

            bytes_ = blosc.decompress(bytes_, as_bytearray=True, **decompress_kwargs)
        elif compression.lower() in get_compression_extensions("blosc2"):
            assert_can_import("blosc2")

            import blosc2

            bytes_ = blosc2.decompress(bytes_, as_bytearray=True, **decompress_kwargs)
        elif compression.lower() in get_compression_extensions("blosc"):
            assert_can_import_any("blosc2", "blosc")

            if check_installed("blosc2"):
                import blosc2

                bytes_ = blosc2.decompress(bytes_, as_bytearray=True, **decompress_kwargs)
            else:
                import blosc

                bytes_ = blosc.decompress(bytes_, as_bytearray=True, **decompress_kwargs)
        else:
            raise ValueError(f"Invalid compression: {compression!r}")
    return bytes_


def dumps(
    obj: tp.Any,
    compression: tp.CompressionLike = None,
    compress_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> bytes:
    """Serialize an object to a byte stream, optionally compressing the result.

    Uses `dill` for pickling if available and otherwise falls back to the standard library `pickle`.
    Compression is applied using `compress`.

    Args:
        obj (Any): Object to serialize.
        compression (CompressionLike): Compression algorithm.

            See `compress`.
        compress_kwargs (KwargsLike): Keyword arguments for compression.
        **kwargs: Keyword arguments for the pickling library's `dumps` method.

    Returns:
        bytes: Serialized and optionally compressed byte stream.
    """
    from vectorbtpro.utils.module_ import warn_cannot_import

    if warn_cannot_import("dill"):
        import pickle
    else:
        import dill as pickle

    bytes_ = pickle.dumps(obj, **kwargs)
    if compress_kwargs is None:
        compress_kwargs = {}
    return compress(bytes_, compression=compression, **compress_kwargs)


def loads(
    bytes_: bytes,
    compression: tp.CompressionLike = None,
    decompress_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.Any:
    """Deserialize an object from a byte stream, decompressing it if necessary.

    Uses `dill` for unpickling when available, otherwise falls back to the standard library `pickle`.
    Decompression is applied using `decompress`.

    Args:
        bytes_ (bytes): Byte stream containing the serialized object.
        compression (CompressionLike): Compression algorithm.

            See `compress`.
        decompress_kwargs (KwargsLike): Keyword arguments for decompression.
        **kwargs: Keyword arguments for the pickling library's `loads` method.

    Returns:
        Any: Deserialized object.
    """
    from vectorbtpro.utils.module_ import warn_cannot_import

    if warn_cannot_import("dill"):
        import pickle
    else:
        import dill as pickle

    if decompress_kwargs is None:
        decompress_kwargs = {}
    bytes_ = decompress(bytes_, compression=compression, **decompress_kwargs)
    return pickle.loads(bytes_, **kwargs)


def suggest_compression(file_name: str) -> tp.Optional[str]:
    """Suggest a compression algorithm based on the file name extension.

    Args:
        file_name (str): Name of the file.

    Returns:
        Optional[str]: Suggested compression algorithm if recognized; otherwise, None.
    """
    suffixes = [suffix.lower() for suffix in file_name.split(".")[1:]]
    if len(suffixes) > 0 and suffixes[-1] in get_compression_extensions():
        compression = suffixes[-1]
    else:
        compression = None
    return compression


def save_bytes(
    bytes_: bytes,
    path: tp.PathLike,
    mkdir_kwargs: tp.KwargsLike = None,
    compression: tp.CompressionLike = None,
    compress_kwargs: tp.KwargsLike = None,
) -> Path:
    """Write a byte stream to a file with optional compression.

    This function compresses the byte stream using `compress` if a compression algorithm is determined,
    either explicitly or based on the file's extension.

    Args:
        bytes_ (bytes): Byte stream containing the serialized object.
        path (PathLike): Destination file path.
        mkdir_kwargs (KwargsLike): Keyword arguments for directory creation.

            See `vectorbtpro.utils.path_.check_mkdir`.
        compression (CompressionLike): Compression algorithm.

            See `compress`.
        compress_kwargs (KwargsLike): Keyword arguments for compression.

    Returns:
        Path: Path to the written file.
    """
    path = Path(path)
    file_name = None
    if compression is None:
        compression = suggest_compression(path.name)
        if compression is not None:
            file_name = path.with_suffix("").name
    if file_name is not None:
        if compress_kwargs is None:
            compress_kwargs = {}
        if "file_name" not in compress_kwargs:
            compress_kwargs = dict(compress_kwargs)
            compress_kwargs["file_name"] = file_name
    if compress_kwargs is None:
        compress_kwargs = {}
    bytes_ = compress(bytes_, compression=compression, **compress_kwargs)
    if mkdir_kwargs is None:
        mkdir_kwargs = {}
    check_mkdir(path.parent, **mkdir_kwargs)
    with open(path, "wb") as f:
        f.write(bytes_)
    return path


def load_bytes(
    path: tp.PathLike,
    compression: tp.CompressionLike = None,
    decompress_kwargs: tp.KwargsLike = None,
) -> bytes:
    """Read a byte stream from a file with optional decompression.

    This function reads the file and applies decompression using `decompress` if a
    compression algorithm is determined, either explicitly or based on the file's extension.

    Args:
        path (PathLike): File path to read.
        compression (CompressionLike): Compression algorithm.

            See `compress`.
        decompress_kwargs (KwargsLike): Keyword arguments for decompression.

    Returns:
        bytes: Read and optionally decompressed byte stream.
    """
    path = Path(path)
    with open(path, "rb") as f:
        bytes_ = f.read()
    if compression is None:
        compression = suggest_compression(path.name)
    if decompress_kwargs is None:
        decompress_kwargs = {}
    return decompress(bytes_, compression=compression, **decompress_kwargs)


def save(
    obj: tp.Any,
    path: tp.Optional[tp.PathLike] = None,
    mkdir_kwargs: tp.KwargsLike = None,
    compression: tp.CompressionLike = None,
    compress_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> Path:
    """Serialize an object and write it to a file with optional compression.

    This function serializes the object using `dumps` and writes the resulting byte stream
    to a file via `save_bytes`.

    Args:
        obj (Any): Object to serialize.
        path (Optional[PathLike]): File path where the object will be saved.

            If a directory is provided, the file name is derived from the object's class name.
        mkdir_kwargs (KwargsLike): Keyword arguments for directory creation.

            See `vectorbtpro.utils.path_.check_mkdir`.
        compression (CompressionLike): Compression algorithm.

            See `compress`.
        compress_kwargs (KwargsLike): Keyword arguments for compression.
        **kwargs: Keyword arguments for `dumps`.

    Returns:
        Path: Path to the saved file.
    """
    bytes_ = dumps(obj, **kwargs)
    if path is None:
        path = type(obj).__name__
    path = Path(path)
    if path.is_dir():
        path /= type(obj).__name__
    return save_bytes(
        bytes_,
        path,
        mkdir_kwargs=mkdir_kwargs,
        compression=compression,
        compress_kwargs=compress_kwargs,
    )


def load(
    path: tp.PathLike,
    compression: tp.CompressionLike = None,
    decompress_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.Any:
    """Read a byte stream from a file and deserialize the contained object.

    This function uses `load_bytes` to read and decompress the byte stream,
    then deserializes the object using `loads`.

    Args:
        path (PathLike): File path from which to load the object.
        compression (CompressionLike): Compression algorithm.

            See `compress`.
        decompress_kwargs (KwargsLike): Keyword arguments for decompression.
        **kwargs: Keyword arguments for `loads`.

    Returns:
        Any: Deserialized object.
    """
    bytes_ = load_bytes(
        path,
        compression=compression,
        decompress_kwargs=decompress_kwargs,
    )
    return loads(bytes_, **kwargs)


@define
class RecState(DefineMixin):
    """Class representing the reconstruction state for an instance."""

    init_args: tp.Args = define.field(factory=tuple)
    """Positional arguments for instance initialization."""

    init_kwargs: tp.Kwargs = define.field(factory=dict)
    """Keyword arguments for instance initialization."""

    attr_dct: tp.Kwargs = define.field(factory=dict)
    """Mapping of attribute names to their writable values."""


@define
class RecInfo(DefineMixin):
    """Class for encapsulating information required to reconstruct an instance."""

    id_: str = define.field()
    """Unique reconstruction identifier."""

    cls: tp.Type = define.field()
    """Class associated with reconstruction."""

    modify_state: tp.Optional[tp.Callable[[RecState], RecState]] = define.field(default=None)
    """Optional callback that modifies the reconstruction state."""

    def register(self) -> None:
        """Register this instance in `rec_info_registry` using its identifier.

        Returns:
            None
        """
        rec_info_registry[self.id_] = self


rec_info_registry = {}
"""Registry of `RecInfo` instances keyed by their `id_`.

This registry is used during unpickling to reconstruct instances when needed.
"""


def get_id_from_class(obj: tp.Any) -> tp.Optional[str]:
    """Obtain the reconstruction identifier for a class or instance.

    If the object is an instance or subclass of `Pickleable` with a defined `_rec_id`, that value is returned.
    Otherwise, returns the fully qualified reference name.

    Args:
        obj (Any): Class or instance to evaluate.

    Returns:
        Optional[str]: Reconstruction identifier or class reference name, or None if not found.
    """
    from vectorbtpro.utils.refs import refname_exists

    if isinstance(obj, type):
        cls = obj
    else:
        cls = type(obj)
    if issubclass(cls, Pickleable):
        if cls._rec_id is not None:
            if not isinstance(cls._rec_id, str):
                raise TypeError(f"Reconstructing id of class {cls} must be a string")
            return cls._rec_id
    refname = cls.__module__ + "." + cls.__name__
    if refname_exists(refname):
        return refname
    return None


def get_class_from_id(class_id: str) -> tp.Optional[tp.Type]:
    """Retrieve a class object from its reconstruction identifier.

    Args:
        class_id (str): Reconstruction identifier of the class.

    Returns:
        Type: Class associated with the provided identifier.
    """
    from vectorbtpro.utils.refs import get_obj

    if class_id in rec_info_registry:
        return rec_info_registry[class_id].cls
    cls = get_obj(class_id, raise_error=False)
    if cls is not None:
        return cls
    raise ValueError(f"Please register an instance of RecInfo for {class_id!r}")


def reconstruct(cls: tp.Union[tp.Hashable, tp.Type], rec_state: RecState) -> tp.Any:
    """Reconstruct an instance from a given class (or identifier) and reconstruction state.

    The function uses the reconstruction state to initialize a new instance, setting initialization
    arguments and updating attributes. If the provided class is not directly a type, it attempts to
    resolve the class using `rec_info_registry` or `vectorbtpro.utils.refs.get_obj`.

    Args:
        cls (Union[Hashable, Type]): Class or its reconstruction identifier.
        rec_state (RecState): State used for reconstruction, including initialization
            arguments and attribute values.

    Returns:
        Any: Reconstructed instance.
    """
    from vectorbtpro.utils.refs import get_obj

    found_rec = False
    if not isinstance(cls, type):
        class_id = cls
        if class_id in rec_info_registry:
            found_rec = True
            cls = rec_info_registry[class_id].cls
            modify_state = rec_info_registry[class_id].modify_state
            if modify_state is not None:
                rec_state = modify_state(rec_state)
    if not isinstance(cls, type):
        if isinstance(cls, str):
            cls_name = cls
            cls = get_obj(cls_name, raise_error=False)
            if cls is None:
                cls = cls_name
    if not isinstance(cls, type):
        raise ValueError(f"Please register an instance of RecInfo for {cls!r}")
    if not found_rec:
        class_path = type(cls).__module__ + "." + type(cls).__name__
        if class_path in rec_info_registry:
            cls = rec_info_registry[class_path].cls
            modify_state = rec_info_registry[class_path].modify_state
            if modify_state is not None:
                rec_state = modify_state(rec_state)

    if issubclass(cls, Pickleable):
        rec_state = cls.modify_state(rec_state)
    obj = cls(*rec_state.init_args, **rec_state.init_kwargs)
    for k, v in rec_state.attr_dct.items():
        setattr(obj, k, v)
    return obj


class Pickleable(Base):
    """Class for pickle-able objects.

    If a subclass's instance cannot be pickled, override its `rec_state` property to return
    a `RecState` instance for reconstruction. If the class definition itself cannot be pickled
    (e.g., created dynamically), override its `_rec_id` with an arbitrary identifier, dump/save the class,
    and map this identifier to the class in `rec_id_map` for reconstruction.
    """

    _rec_id: tp.ClassVar[tp.Optional[str]] = None
    """Reconstruction identifier."""

    def dumps(self, rec_state_only: bool = False, **kwargs) -> bytes:
        """Serialize the instance to a pickle byte stream.

        Args:
            rec_state_only (bool): If True, serialize only the instance's reconstruction state
                for direct unpickling.
            **kwargs: Keyword arguments for `dumps`.

        Returns:
            bytes: Serialized byte stream.
        """
        if rec_state_only:
            rec_state = self.rec_state
            if rec_state is None:
                raise ValueError("Reconstruction state is None")
            return dumps(rec_state, **kwargs)
        return dumps(self, **kwargs)

    @classmethod
    def loads(cls: tp.Type[PickleableT], bytes_: bytes, check_type: bool = True, **kwargs) -> PickleableT:
        """Reconstruct an instance from a pickle byte stream.

        If the unpickled object is an instance of `RecState`, it is transformed via `reconstruct`.

        Args:
            bytes_ (bytes): Byte stream containing the serialized object.
            check_type (bool): If True, validates that the unpickled object is an instance of the class.
            **kwargs: Keyword arguments for `loads`.

        Returns:
            Pickleable: Unpickled instance.
        """
        obj = loads(bytes_, **kwargs)
        if isinstance(obj, RecState):
            obj = reconstruct(cls, obj)
        if check_type and not isinstance(obj, cls):
            raise TypeError(f"Loaded object must be an instance of {cls}")
        return obj

    def encode_config_node(self, key: str, value: tp.Any, **kwargs) -> tp.Any:
        """Encode a configuration node.

        This method is used to encode the value or prepare it for encoding.

        Args:
            key (str): Key for the configuration node.
            value (Any): Value to encode.
            **kwargs: Keyword arguments for encoding.

        Returns:
            Any: Encoded configuration node.
        """
        return value

    def encode_config(
        self,
        top_name: tp.Optional[str] = None,
        unpack_objects: bool = True,
        compress_unpacked: bool = True,
        use_refs: bool = True,
        use_class_ids: bool = True,
        nested: bool = True,
        to_dict: bool = False,
        parser_kwargs: tp.KwargsLike = None,
        **encode_node_kwargs,
    ) -> str:
        """Encode the instance to a configuration string based on its reconstruction state.

        This method encodes the instance in a format that can be decoded using `Pickleable.decode_config`.
        It uses the instance's `rec_state` property and raises an error if it is None. If an object cannot be
        represented as a string, it is serialized using `dumps`.

        !!! note
            The initial order of keys can be preserved only by using references.

        Args:
            top_name (Optional[str]): Top-level section name.
            unpack_objects (bool): Flag to store a `Pickleable` object's reconstruction state in a separate section.

                Appends `@` and class name to the section name.
            compress_unpacked (bool): Flag to compress empty values in the reconstruction state.

                Keys in the reconstruction state will be appended with `~` to avoid collision with
                user-defined keys having the same name.
            use_refs (bool): Flag to create references for duplicate unhashable objects.

                Out of unhashable objects sharing the same id, only the first one will be defined
                while others will store the reference (`&` + key path) to the first one.
            use_class_ids (bool): Flag to substitute class objects with their identifiers.

                If `get_id_from_class` returns None, will pickle the definition.
            nested (bool): Flag indicating whether to represent sub-dictionaries as individual sections.
            to_dict (bool): Flag to treat objects as dictionaries during encoding.
            parser_kwargs (KwargsLike): Keyword arguments for `configparser.RawConfigParser`.
            **encode_node_kwargs: Keyword arguments for `Pickleable.encode_config_node`.

        Returns:
            str: Encoded configuration string.
        """
        import configparser
        from io import StringIO

        if parser_kwargs is None:
            parser_kwargs = {}
        parser = configparser.RawConfigParser(**parser_kwargs)
        parser.optionxform = str

        def _is_dict(dct, _to_dict=to_dict):
            if _to_dict:
                return isinstance(dct, dict)
            return type(dct) is dict

        def _get_path(k):
            if "@" in k:
                return k.split("@")[0].strip()
            return k

        def _is_referable(k):
            if "@" in k:
                return False
            if k.endswith("~"):
                return False
            return True

        def _preprocess_key(k):
            k = k.replace("#", "__HASH__")
            k = k.replace(":", "__COL__")
            k = k.replace("=", "__EQ__")
            return k

        if top_name is None:
            top_name = "top"
        stack = [(None, top_name, self)]
        dct = dict()
        id_paths = dict()
        id_objs = dict()
        while stack:
            parent_k, k, v = stack.pop(0)
            if not isinstance(k, str):
                raise TypeError("Dictionary keys must be strings")

            if parent_k is not None and use_refs and _is_referable(k):
                if id(v) in id_paths:
                    v = "&" + id_paths[id(v)]
                else:
                    if not is_hashable(v):
                        id_paths[id(v)] = _get_path(parent_k) + "." + _get_path(k)
                        id_objs[id(v)] = v
            if _is_dict(v) and nested:
                if parent_k is not None and use_refs:
                    if parent_k is None:
                        ref_k = _get_path(k)
                    else:
                        ref_k = _get_path(parent_k) + "." + _get_path(k)
                    dct[parent_k][_get_path(k)] = "&" + ref_k
                if parent_k is None:
                    _k = k
                else:
                    _k = _get_path(parent_k) + "." + k
                dct[_k] = dict()
                if len(v) == 0:
                    v = {"_": "_"}
                i = 0
                for k2, v2 in v.items():
                    k2 = _preprocess_key(k2)
                    stack.insert(i, (_k, k2, v2))
                    i += 1
            else:
                if (unpack_objects or k == top_name) and isinstance(v, Pickleable):
                    if use_class_ids:
                        class_id = get_id_from_class(v)
                    else:
                        class_id = None
                    if class_id is None:
                        class_id = "base64," + base64.b64encode(dumps(type(v))).decode("ascii")
                    rec_state = v.rec_state
                    if rec_state is None:
                        if parent_k is None:
                            _k = _get_path(k)
                        else:
                            _k = _get_path(parent_k) + "." + _get_path(k)
                        raise ValueError(f"Must define reconstruction state for {_k!r}")
                    new_v = vars(rec_state)
                    if compress_unpacked and (len(new_v["init_args"]) == 0 and len(new_v["attr_dct"]) == 0):
                        new_v = new_v["init_kwargs"]
                    else:
                        new_v = {k + "~": v for k, v in new_v.items()}
                    k = _preprocess_key(k)
                    stack.insert(0, (parent_k, k + " @" + class_id, new_v))
                else:
                    if parent_k is None:
                        dct[k] = v
                    else:
                        dct[parent_k][k] = v

        for k, v in dct.items():
            parser.add_section(k)
            if len(v) == 0:
                v = {"_": "_"}
            for k2, v2 in v.items():
                v2 = self.encode_config_node(k2, v2, **encode_node_kwargs)
                if isinstance(v2, str):
                    if not (k2 == "_" and v2 == "_") and not v2.startswith("&"):
                        v2 = repr(v2)
                elif isinstance(v2, type):
                    if use_class_ids:
                        class_id = get_id_from_class(v2)
                    else:
                        class_id = None
                    if class_id is None:
                        class_id = "base64," + base64.b64encode(dumps(v2)).decode("ascii")
                    v2 = "@" + class_id
                elif isinstance(v2, float) and np.isnan(v2):
                    v2 = "np.nan"
                elif isinstance(v2, float) and np.isposinf(v2):
                    v2 = "np.inf"
                elif isinstance(v2, float) and np.isneginf(v2):
                    v2 = "-np.inf"
                else:
                    try:
                        ast.literal_eval(repr(v2))
                        v2 = repr(v2)
                    except Exception:
                        try:
                            float(repr(v2))
                            v2 = repr(v2)
                        except Exception:
                            v2 = "!base64," + base64.b64encode(dumps(v2)).decode("ascii")
                parser.set(k, k2, v2)
        with StringIO() as f:
            parser.write(f)
            config_str = f.getvalue()
        return config_str

    @classmethod
    def decode_config_node(cls, key: str, value: tp.Any, **kwargs) -> tp.Any:
        """Decode a configuration node.

        This method is used to decode the value or prepare it for decoding.

        Args:
            key (str): Key for the configuration node.
            value (Any): Value to decode.
            **kwargs: Keyword arguments for decoding.

        Returns:
            Any: Decoded configuration node.
        """
        return value

    @classmethod
    def decode_config(
        cls: tp.Type[PickleableT],
        config_str: str,
        parse_literals: bool = True,
        run_code: bool = True,
        pack_objects: bool = True,
        use_refs: bool = True,
        use_class_ids: bool = True,
        code_context: tp.KwargsLike = None,
        parser_kwargs: tp.KwargsLike = None,
        check_type: bool = True,
        **decode_node_kwargs,
    ) -> PickleableT:
        """Decode an instance from a configuration string.

        This function parses configuration strings and supports dot notation for nesting sections.
        It can parse configs without sections. Sections can also become sub-dictionaries if their names use
        dot notation. For example, the section `a.b` will become a sub-dictionary of the section `a`
        and the section `a.b.c` will become a sub-dictionary of the section `a.b`. You don't have to define
        the section `a` explicitly, it will automatically become the outermost key.
        Sections containing only a single pair (`_ = _`) are treated as empty dictionaries.

        !!! warning
            Unpickling byte streams and running code has important security implications. Don't attempt
            to parse configs coming from untrusted sources as those can contain malicious code!

        Args:
            config_str (str): Configuration string to decode.
            parse_literals (bool): Detect Python literals and container types (e.g., `True`, `[]`),
                including special values like `np.nan`, `np.inf`, and `-np.inf`.
            run_code (bool): Execute Python code prefixed with `!`.

                Uses a context that includes all from `vectorbtpro.imported_star`
                along with any provided in `code_context`.
            pack_objects (bool): Instantiate and reconstruct objects specified by section class paths.

                Section names prefixed with `@` trigger the instantiation of a `RecState` object
                and reconstruction through `reconstruct`.
            use_refs (bool): Substitute reference strings prefixed with `&` with actual objects
                using a DAG constructed with `graphlib`.
            use_class_ids (bool): Replace class identifiers prefixed with `@` with corresponding classes.
            code_context (KwargsLike): Context dictionary used during execution of Python code.
            parser_kwargs (KwargsLike): Keyword arguments for `configparser.RawConfigParser`.
            check_type (bool): If True, validates that the decoded object is an instance of the class.
            **decode_node_kwargs: Keyword arguments for `Pickleable.decode_config_node`.

        Returns:
            Pickleable: Decoded instance.

        Examples:
            File `types.cfg`:

            ```ini
            string = "hello world"
            boolean = False
            int = 123
            float = 123.45
            exp_float = 1e-10
            nan = np.nan
            inf = np.inf
            neg_inf = -np.inf
            numpy = !np.array([1, 2, 3])
            pandas = !pd.Series([1, 2, 3])
            expression = !dict(sub_dict2=dict(some="value"))
            mult_expression = !import math; math.floor(1.5)
            ```

            ```pycon
            >>> from vectorbtpro import *

            >>> vbt.pprint(vbt.pdict.load("types.cfg"))
            pdict(
                string='hello world',
                boolean=False,
                int=123,
                float=123.45,
                exp_float=1e-10,
                nan=np.nan,
                inf=np.inf,
                neg_inf=-np.inf,
                numpy=<numpy.ndarray object at 0x7fe1bf84f690 of shape (3,)>,
                pandas=<pandas.core.series.Series object at 0x7fe1c9a997f0 of shape (3,)>,
                expression=dict(
                    sub_dict2=dict(
                        some='value'
                    )
                ),
                mult_expression=1
            )
            ```

            File `refs.cfg`:

            ```ini
            [top]
            sr = &top.sr

            [top.sr @pandas.Series]
            data = [10756.12, 10876.76, 11764.33]
            index = &top.sr.index
            name = "Open time"

            [top.sr.index @pandas.DatetimeIndex]
            data = ["2023-01-01", "2023-01-02", "2023-01-03"]
            ```

            ```pycon
            >>> vbt.pdict.load("refs.cfg")["sr"]
            2023-01-01    10756.12
            2023-01-02    10876.76
            2023-01-03    11764.33
            Name: Open time, dtype: float64
            ```
        """
        import configparser

        from vectorbtpro.utils.eval_ import evaluate

        if parser_kwargs is None:
            parser_kwargs = {}
        parser = configparser.RawConfigParser(**parser_kwargs)
        parser.optionxform = str

        try:
            parser.read_string(config_str)
        except configparser.MissingSectionHeaderError:
            parser.read_string("[top]\n" + config_str)

        def _preprocess_key(k):
            k = k.replace("__HASH__", "#")
            k = k.replace("__COL__", ":")
            k = k.replace("__EQ__", "=")
            return k

        def _get_path(k):
            if "@" in k:
                return k.split("@")[0].strip()
            return k

        dct = {}
        has_top_section = False
        for k in parser.sections():
            k = _preprocess_key(k)
            v = dict(parser.items(k))
            if _get_path(k) == "top":
                has_top_section = True
            elif not _get_path(k).startswith("top."):
                k = "top." + k
            new_v = {}
            for k2, v2 in v.items():
                k2 = _preprocess_key(k2)
                if use_refs and v2.startswith("&") and not v2[1:].startswith("top."):
                    new_v[k2] = "&top." + v2[1:]
                else:
                    new_v[k2] = v2
            dct[k] = new_v
        if not has_top_section:
            dct = {"top": {"_": "_"}, **dct}

        def _get_class(k):
            if "@" in k:
                return k.split("@")[1].strip()
            return None

        class_map = {_get_path(k): _get_class(k) for k, v in dct.items()}
        dct = {_get_path(k): v for k, v in dct.items()}

        def _get_ref_node(ref):
            if ref in dct:
                ref_edges.add((k, (k, k2)))
                return ref
            ref_section = ".".join(ref.split(".")[:-1])
            ref_key = ref.split(".")[-1]
            if ref_section not in dct:
                raise ValueError(f"Referenced section {ref_section!r} not found")
            if ref_key not in dct[ref_section]:
                raise ValueError(f"Referenced object {ref!r} not found")
            return ref_section, ref_key

        new_dct = dict()
        if code_context is None:
            code_context = {}
        else:
            code_context = dict(code_context)
        try:
            for k, v in vbt.imported_star.items():
                if k not in code_context:
                    code_context[k] = v
        except AttributeError:
            pass
        ref_edges = set()
        for k, v in dct.items():
            new_dct[k] = {}
            if len(v) == 1 and list(v.items())[0] == ("_", "_"):
                continue
            for k2, v2 in v.items():
                v2 = cls.decode_config_node(k2, v2, **decode_node_kwargs)
                if isinstance(v2, str):
                    v2 = v2.strip()
                    if use_refs and v2.startswith("&"):
                        ref_node = _get_ref_node(v2[1:])
                        ref_edges.add((k, (k, k2)))
                        ref_edges.add(((k, k2), ref_node))
                    elif v2.startswith("@"):
                        v2 = v2[1:]
                        if v2.startswith("base64,"):
                            if not run_code:
                                raise ValueError("Running code is disabled")
                            v2 = loads(base64.b64decode(v2[7:]))
                        else:
                            if not use_class_ids:
                                raise ValueError("Class ID resolution is disabled")
                            v2 = get_class_from_id(v2)
                    elif v2.startswith("!"):
                        v2 = v2[1:]
                        if not run_code:
                            raise ValueError("Running code is disabled")
                        if v2.startswith("base64,"):
                            v2 = loads(base64.b64decode(v2[7:]))
                        elif v2.startswith("vbt.loads(") and v2.endswith(")"):
                            v2 = evaluate(v2[len("vbt.") :], context={**code_context, "loads": loads})
                        else:
                            v2 = evaluate(v2, context=code_context)
                    else:
                        if parse_literals:
                            if v2 == "np.nan":
                                v2 = np.nan
                            elif v2 == "np.inf":
                                v2 = np.inf
                            elif v2 == "-np.inf":
                                v2 = -np.inf
                            else:
                                try:
                                    v2 = ast.literal_eval(v2)
                                except Exception:
                                    try:
                                        v2 = float(v2)
                                    except Exception:
                                        pass
                new_dct[k][k2] = v2
        dct = new_dct

        graph = dict()
        keys = sorted(dct.keys())
        hierarchy = [keys[0]]
        for i in range(1, len(keys)):
            while True:
                if keys[i].startswith(hierarchy[-1] + "."):
                    if hierarchy[-1] not in graph:
                        graph[hierarchy[-1]] = set()
                    graph[hierarchy[-1]].add(keys[i])
                    hierarchy.append(keys[i])
                    break
                del hierarchy[-1]
        if use_refs and len(ref_edges) > 0:
            for k1, k2 in ref_edges:
                if k1 not in graph:
                    graph[k1] = set()
                graph[k1].add(k2)
        if len(graph) > 0:
            sorter = TopologicalSorter(graph)
            topo_order = list(sorter.static_order())

            resolved_nodes = dict()
            for k in topo_order:
                if isinstance(k, tuple):
                    v = dct[k[0]][k[1]]
                    if use_refs and isinstance(v, str) and v.startswith("&"):
                        ref_node = _get_ref_node(v[1:])
                        v = resolved_nodes[ref_node]
                else:
                    section_dct = dict(dct[k])
                    if k in graph:
                        for k2 in graph[k]:
                            if isinstance(k2, tuple):
                                section_dct[k2[1]] = resolved_nodes[k2]
                            else:
                                _k2 = k2[len(k) + 1 :]
                                last_k = _k2.split(".")[-1]
                                d = section_dct
                                for s in _k2.split(".")[:-1]:
                                    if s not in d:
                                        d[s] = dict()
                                    d = d[s]
                                d[last_k] = resolved_nodes[k2]
                    if class_map.get(k, None) is not None and (pack_objects or k == "top"):
                        section_cls = class_map[k]
                        if section_cls.startswith("base64,"):
                            if not run_code:
                                raise ValueError("Running code is disabled")
                            section_cls = loads(base64.b64decode(section_cls[7:]))
                        elif not use_class_ids:
                            raise ValueError("Class ID resolution is disabled")
                        init_args = section_dct.pop("init_args~", ())
                        init_kwargs = section_dct.pop("init_kwargs~", {})
                        attr_dct = section_dct.pop("attr_dct~", {})
                        init_kwargs.update(section_dct)
                        rec_state = RecState(
                            init_args=init_args,
                            init_kwargs=init_kwargs,
                            attr_dct=attr_dct,
                        )
                        v = reconstruct(section_cls, rec_state)
                    else:
                        v = section_dct
                resolved_nodes[k] = v

            obj = resolved_nodes[topo_order[-1]]
        else:
            obj = dct["top"]
        if isinstance(obj, dict) and not isinstance(obj, Pickleable):
            obj = reconstruct(cls, RecState(init_kwargs=obj))
        if check_type and not isinstance(obj, cls):
            raise TypeError(f"Decoded object must be an instance of {cls}, got {type(obj)}")
        return obj

    def encode_yaml_node(self, dumper: tp.Any, obj: tp.Any, **kwargs) -> tp.Any:
        """Encode a YAML node.

        This method is used to encode the object into a YAML node or prepare it for encoding.
        It isn't called for basic types (excluding tuples), which are handled by the YAML dumper directly.

        Args:
            dumper (str): YAML dumper.
            obj (Any): Object to encode.
            **kwargs: Keyword arguments for encoding.

        Returns:
            Any: Encoded YAML node.
        """
        return obj

    def encode_yaml(
        self,
        root_key: tp.Optional[str] = None,
        unpack_objects: bool = True,
        compress_unpacked: bool = True,
        use_refs: bool = True,
        use_class_ids: bool = True,
        use_ruamel: tp.Optional[bool] = None,
        collapse_keys: tp.Union[bool, str] = True,
        collapse_sep: str = ".",
        collapse_esc: tp.Optional[str] = "\\",
        yaml_kwargs: tp.KwargsLike = None,
        **encode_node_kwargs,
    ) -> str:
        """Serialize the instance to a YAML string with rich, customizable representation.

        This method supports two YAML backends (`ruamel.yaml` and PyYAML), optional
        use of class IDs, controlled unpacking/compression of custom `Pickleable`
        objects, and alias/reference management. Depending on the object types and
        the flags provided, different YAML tags are emitted to capture enough
        information for faithful deserialization.

        Tag mapping:

        * `!tuple`: Python `tuple` objects are emitted as YAML sequences tagged with `!tuple`
            to preserve immutability semantics.
        * `!class`: Python classes with a registered class ID (and when `use_class_ids`
            is True) are represented as a scalar tagged `!class` containing the class ID string.
            If no class ID exists or `use_class_ids` is False, the class is fall-back pickled.
        * `!pickle`: Opaque pickled representation used for any object that does not fall into
            a special category, classes without a usable class ID or when class IDs are disabled,
            and `Pickleable` instances when `unpack_objects` is False.
        * `!rec:<class_id>`: `Pickleable` instances with a registered class ID are unpacked
            (when `unpack_objects` is True) into their reconstruction state. The tag encodes the
            class ID, and the mapping contains state components (e.g., `init_args~`, `init_kwargs~`,
            `attr_dct~`). If `compress_unpacked` is enabled and the state is trivial, it may emit
            a simplified mapping (e.g., dropping default entries) with tilde-suffixed keys only when needed.
        * Alias/reference control: If `use_refs` is False, YAML aliasing is
            suppressed even for repeated structures.

        Args:
            root_key (Optional[str]): Root key for the YAML mapping.

                Can contain the `collapse_sep` character to represent a nested structure (won't be escaped).
            unpack_objects (bool): If True, `Pickleable` objects are represented via their
                reconstruction state (`!rec:...`) rather than being pickled as opaque blobs.
            compress_unpacked (bool): If True, the unpacked reconstruction state is simplified
                when it contains only default or empty components.
            use_refs (bool): Whether to allow YAML aliases/references; if False, aliasing is
                disabled during serialization.
            use_class_ids (bool): Enable use of registered class IDs for `!class` and `!rec:...`
                tags instead of raw pickles.
            use_ruamel (Optional[bool]): Override auto-detection of the YAML engine.

                If None, presence of the `ruamel` package is checked and used if available.
            collapse_keys (Union[bool, str]): Whether to collapse nested dictionary keys.

                Supports the following values:

                * "none" or False: no collapsing.
                * "single" or True: flatten 1-key chains.
                * "all": flatten every path.
            collapse_sep (str): Separator inserted between collapsed segments.
            collapse_esc (Optional[str]): Prefix that protects literal `collapse_sep` inside keys.

                Set to None to disable escaping.
            yaml_kwargs (KwargsLike): Keyword arguments for the underlying `dump` method of
                the chosen YAML engine.
            **encode_node_kwargs: Keyword arguments for `Pickleable.encode_yaml_node`.

        Returns:
            str: YAML-formatted string representing the instance.

        Raises:
            ValueError: If a `Pickleable` object is unpacked but lacks a valid
                reconstruction ID or necessary reconstruction state.
            ImportError: If the requested YAML backend is unavailable or fails import.
        """
        if use_ruamel is None:
            from vectorbtpro.utils.module_ import check_installed

            use_ruamel = check_installed("ruamel")
        elif use_ruamel is True:
            from vectorbtpro.utils.module_ import warn_cannot_import

            if warn_cannot_import("ruamel"):
                use_ruamel = False
        if use_ruamel:
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("ruamel")

            from ruamel.yaml import YAML
            from ruamel.yaml.nodes import Node

            yaml_engine = YAML()
            yaml_engine.default_flow_style = False
            orig_ignore_aliases = yaml_engine.representer.ignore_aliases

            def _ignore_aliases(self, data):
                if use_refs:
                    return orig_ignore_aliases(data)
                return True

            yaml_engine.representer.ignore_aliases = MethodType(_ignore_aliases, yaml_engine.representer)
            yaml_representers = yaml_engine.representer.yaml_representers
            orig_yaml_representers = dict(yaml_representers)
            yaml_mult_representers = yaml_engine.representer.yaml_multi_representers
            orig_yaml_mult_representers = dict(yaml_mult_representers)
        else:
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("yaml")

            import yaml as _pyyaml
            from yaml.nodes import Node

            class _Dumper(_pyyaml.SafeDumper):
                default_flow_style = False

                def ignore_aliases(self, data):
                    if use_refs:
                        return super().ignore_aliases(data)
                    return True

            dumper_cls = _Dumper
            yaml_engine = _pyyaml

        def _represent_tuple(dumper, obj):
            obj = self.encode_yaml_node(dumper, obj, **encode_node_kwargs)
            if isinstance(obj, Node):
                return obj

            return dumper.represent_sequence("!tuple", list(obj))

        if collapse_keys is True:
            collapse_keys = "single"
        elif collapse_keys is False:
            collapse_keys = "none"
        elif isinstance(collapse_keys, str):
            collapse_keys = collapse_keys.lower()
        if not isinstance(collapse_keys, str) or collapse_keys not in ("none", "single", "all"):
            raise ValueError(f"Invalid collapse_keys: {collapse_keys!r}")

        def _escape_key(key):
            if collapse_esc is None:
                return key
            return key.replace(collapse_esc, collapse_esc + collapse_esc).replace(
                collapse_sep, collapse_esc + collapse_sep
            )

        class _CollapsedDict(dict):
            __slots__ = ()

        def _collapse_dict(d):
            if type(d) is not dict:
                return d
            if not d:
                return _CollapsedDict()
            if collapse_keys == "none":
                return _CollapsedDict({_escape_key(k) if isinstance(k, str) else k: v for k, v in d.items()})

            result = {}

            def _visit(curr, path):
                if type(curr) is dict and not curr:
                    key = collapse_sep.join(path) if path else ""
                    result[key] = _CollapsedDict()
                    return

                if type(curr) is dict:
                    if collapse_keys == "single" and len(curr) > 1 and path:
                        combined = collapse_sep.join(path)
                        inner = _CollapsedDict()
                        for k, v in curr.items():
                            if isinstance(k, str):
                                k_esc = _escape_key(str(k))
                                collapsed = _collapse_dict(v) if type(v) is dict else v
                                if type(collapsed) is dict and len(collapsed) == 1:
                                    sk, sv = next(iter(collapsed.items()))
                                    inner[f"{k_esc}{collapse_sep}{sk}" if sk else k_esc] = sv
                                else:
                                    inner[k_esc] = collapsed
                            else:
                                inner[k] = _collapse_dict(v) if type(v) is dict else v
                        result[combined] = inner
                    else:
                        for k, v in curr.items():
                            if isinstance(k, str):
                                _visit(v, path + [_escape_key(str(k))])
                            else:
                                result[k] = _collapse_dict(v) if type(v) is dict else v
                else:
                    key = collapse_sep.join(path) if path else ""
                    result[key] = curr

            _visit(d, [])
            return _CollapsedDict(result)

        def _represent_mapping(dumper, obj):
            obj = self.encode_yaml_node(dumper, obj, **encode_node_kwargs)
            if isinstance(obj, Node):
                return obj

            if isinstance(obj, _CollapsedDict):
                return dumper.represent_mapping("tag:yaml.org,2002:map", obj)
            if collapse_keys != "none" and any(isinstance(v, dict) for v in obj.values()):
                obj = _collapse_dict(obj)
            return dumper.represent_mapping("tag:yaml.org,2002:map", obj)

        def _represent_object(dumper, obj):
            obj = self.encode_yaml_node(dumper, obj, **encode_node_kwargs)
            if isinstance(obj, Node):
                return obj

            if isinstance(obj, type):
                if use_class_ids:
                    class_id = get_id_from_class(obj)
                else:
                    class_id = None
                if class_id is None:
                    class_id = "base64," + base64.b64encode(dumps(obj)).decode("ascii")
                return dumper.represent_scalar("!class", class_id)

            if isinstance(obj, Pickleable) and unpack_objects:
                if use_class_ids:
                    class_id = get_id_from_class(obj)
                else:
                    class_id = None
                if class_id is None:
                    class_id = "base64," + base64.b64encode(dumps(type(obj))).decode("ascii")
                rec_state = obj.rec_state
                if rec_state is None:
                    raise ValueError(f"Must define reconstruction state for {type(obj)}")
                mapping = vars(rec_state)
                if compress_unpacked and (not mapping.get("init_args", ()) and not mapping.get("attr_dct", {})):
                    mapping = mapping.get("init_kwargs", {})
                else:
                    mapping = {k + "~": v for k, v in mapping.items()}
                mapping = _collapse_dict(mapping)
                return dumper.represent_mapping(f"!rec:{class_id}", mapping)

            encoded = "base64," + base64.b64encode(dumps(obj)).decode("ascii")
            return dumper.represent_scalar("!pickle", encoded)

        if use_ruamel:
            representer = yaml_engine.representer
        else:
            representer = dumper_cls
        representer.add_representer(tuple, _represent_tuple)
        representer.add_representer(_CollapsedDict, _represent_mapping)
        representer.add_representer(dict, _represent_mapping)
        representer.add_multi_representer(object, _represent_object)

        if root_key is not None:
            obj = {root_key: self}
        else:
            obj = self

        if yaml_kwargs is None:
            yaml_kwargs = {}
        else:
            yaml_kwargs = dict(yaml_kwargs)
        if use_ruamel:
            from io import StringIO

            buf = StringIO()
            yaml_engine.dump(obj, buf, **yaml_kwargs)
            yaml_engine.representer.yaml_representers.clear()
            yaml_engine.representer.yaml_representers.update(orig_yaml_representers)
            yaml_engine.representer.yaml_multi_representers.clear()
            yaml_engine.representer.yaml_multi_representers.update(orig_yaml_mult_representers)
            return buf.getvalue()
        dumper_cls = yaml_kwargs.pop("Dumper", dumper_cls)
        yaml_kwargs.setdefault("sort_keys", False)
        yaml_kwargs.setdefault("allow_unicode", True)
        return yaml_engine.dump(obj, Dumper=dumper_cls, **yaml_kwargs)

    @classmethod
    def decode_yaml_node(cls, loader: tp.Any, tag: str, node: tp.Any, **kwargs) -> tp.Any:
        """Decode a YAML node.

        This method is used to decode the YAML node into an object or prepare it for decoding.
        It isn't called for basic types (excluding tuples), which are handled by the YAML loader directly.

        Args:
            loader (str): YAML loader.
            tag (str): YAML tag.
            node (Any): YAML node to decode.
            **kwargs: Keyword arguments for decoding.

        Returns:
            Any: Decoded object.
        """
        return node

    @classmethod
    def decode_yaml(
        cls: tp.Type[PickleableT],
        yaml_str: str,
        run_code: bool = True,
        pack_objects: bool = True,
        use_class_ids: bool = True,
        code_context: tp.KwargsLike = None,
        use_ruamel: tp.Optional[bool] = None,
        collapse_sep: str = ".",
        collapse_esc: tp.Optional[str] = "\\",
        yaml_kwargs: tp.KwargsLike = None,
        check_type: bool = True,
        **decode_node_kwargs,
    ) -> PickleableT:
        """Deserialize a YAML string back into an instance of this class.

        Supports safe reconstruction of complex objects including unpacked `Pickleable` instances,
        class resolution via registered IDs, and optional execution of embedded expressions
        or pickled payloads.

        Tag mapping:

        * `!tuple`: YAML sequences tagged with `!tuple` are converted to Python `tuple` objects.
        * `!class`: Scalars tagged with `!class` are resolved to Python classes if `use_class_ids`
            is True; otherwise an error is raised.
        * `!pickle`: Scalars tagged with `!pickle` contain base64-encoded pickled data.
            If `run_code` is True, this data is unpickled and the original object is returned.
            If `run_code` is False, the raw scalar (e.g., base64 string) remains.
        * `!expr`: Tagged expressions are evaluated via the internal evaluator when `run_code` is True,
            using `code_context` to supply names. If `run_code` is False, the expression is not executed.
        * `!rec:<class_id>`: Reconstruction state for a `Pickleable` object with the given class ID.
            When `pack_objects` is True, the mapping is used to build a `RecState` and the object is
            reconstructed. If `pack_objects` is False, the raw mapping is returned unassembled.

        Fallback behavior:
            Standard YAML scalars, sequences, and mappings are handled by the
            underlying loader. After loading, if the top-level result is a plain
            dictionary and not already an instance of `Pickleable`, it is treated
            as keyword arguments for constructing an instance of this class.

        !!! warning
            Unpickling byte streams and running code has important security implications. Don't attempt
            to parse configs coming from untrusted sources as those can contain malicious code!

        Args:
            yaml_str (str): YAML content to decode.
            run_code (bool): Whether to execute code in `!expr` tags and to unpickle `!pickle` payloads.

                If False, those tags yield raw content.
            pack_objects (bool): If True, reconstruction mappings (`!rec:...`) are converted back into objects;
                otherwise raw mappings are returned.
            use_class_ids (bool): Enable resolution of `!class` and `!rec:...` references to
                actual classes via registered IDs.
            code_context (KwargsLike): Optional namespace for code evaluation; prepopulated with existing
                imports when not provided.
            use_ruamel (Optional[bool]): Override auto-detection of the YAML engine.

                If None, presence of the `ruamel` package is checked and used if available.
            collapse_sep (str): Separator inserted between collapsed segments.
            collapse_esc (Optional[str]): Prefix that protects literal `collapse_sep` inside keys.

                Set to None to disable escaping.
            yaml_kwargs (KwargsLike): Keyword arguments for the underlying `load` method of
                the chosen YAML engine.
            check_type (bool): If True, ensures the final decoded object is an instance of this class
                and raises if it is not.
            **encode_node_kwargs: Keyword arguments for `Pickleable.decode_yaml_node`.

        Returns:
            Pickleable: The reconstructed object, respecting type checking if requested.

        Raises:
            TypeError: If `check_type` is True and the decoded object is not an instance of this class.
            ConstructorError: On failures during object construction, such as missing
                class IDs when required, disabled class references, or invalid serialized content.
            ImportError: If the requested YAML backend is unavailable or cannot be imported.

        Examples:
            File `types.yaml`:

            ```yaml
            string: 'hello world'
            boolean: false
            int: 123
            float: 123.45
            exp_float: 1e-10
            nan: .nan
            inf: .inf
            neg_inf: -.inf
            none:
            list:
              - &one 1
              - &two 2
              - &three 3
            dict:
              a: *one
              b: *two
              c: *three
            collapsed_dict.a: 1
            collapsed_dict.b: 2
            collapsed_dict.c: 3
            escaped_dict\\.a: 1
            escaped_dict\\.b: 2
            escaped_dict\\.c: 3
            tuple: !tuple [1, 2, 3]
            class: !class pd.Series
            instance: !rec:pd.Series
              data: [1, 2, 3]
            expl_instance: !rec:pd.Timedelta
              init_args~: []
              init_kwargs~:
                days: 1
              attr_dct~: {}
            pickle: !pickle
              base64,gASVSAAAAAAAAACMHnBhbmRhcy5fbGlicy50c2xpYnMudGltZWRlbHRhc5SME190aW1lZGVsdGFfdW5waWNrbGWUk5SKBgAAT5GUTksKhpRSlC4=
            expression: !expr dict(sub_dict2=dict(some="value"))
            mult_expression: !expr |
              import math

              math.floor(1.5)
            ```

            ```pycon
            >>> from vectorbtpro import *

            >>> vbt.pprint(vbt.pdict.load("types.yml"))
            pdict({
                'string': 'hello world',
                'boolean': False,
                'int': 123,
                'float': 123.45,
                'exp_float': 1e-10,
                'nan': np.nan,
                'inf': np.inf,
                'neg_inf': -np.inf,
                'none': None,
                'list': [
                    1,
                    2,
                    3
                ],
                'dict': dict(
                    a=1,
                    b=2,
                    c=3
                ),
                'collapsed_dict': dict(
                    a=1,
                    b=2,
                    c=3
                ),
                'escaped_dict.a': 1,
                'escaped_dict.b': 2,
                'escaped_dict.c': 3,
                'tuple': (
                    1,
                    2,
                    3
                ),
                'class': <class 'pandas.core.series.Series'>,
                'instance': <pandas.core.series.Series object at 0x14d14a7d0 with shape (3,)>,
                'expl_instance': Timedelta('1 days 00:00:00'),
                'pickle': Timedelta('1 days 00:00:00'),
                'expression': dict(
                    sub_dict2=dict(
                        some='value'
                    )
                ),
                'mult_expression': 1
            })
            ```
        """
        from vectorbtpro.utils.eval_ import evaluate

        if use_ruamel is None:
            from vectorbtpro.utils.module_ import check_installed

            use_ruamel = check_installed("ruamel")
        elif use_ruamel is True:
            from vectorbtpro.utils.module_ import warn_cannot_import

            if warn_cannot_import("ruamel"):
                use_ruamel = False
        if use_ruamel:
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("ruamel")

            from ruamel.yaml import YAML
            from ruamel.yaml.nodes import Node, ScalarNode, SequenceNode, MappingNode
            from ruamel.yaml.constructor import ConstructorError
            from ruamel.yaml.comments import CommentedMap

            yaml_engine = YAML(typ="safe")
            yaml_constructors = yaml_engine.constructor.yaml_constructors
            orig_yaml_constructors = dict(yaml_constructors)
            yaml_mult_constructors = yaml_engine.constructor.yaml_multi_constructors
            orig_yaml_mult_constructors = dict(yaml_mult_constructors)
        else:
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("yaml")

            import yaml as _pyyaml
            from yaml.nodes import Node, ScalarNode, SequenceNode, MappingNode
            from yaml.constructor import ConstructorError

            CommentedMap = type(None)

            class _Loader(_pyyaml.SafeLoader):
                pass

            loader_cls = _Loader
            yaml_engine = _pyyaml

        if code_context is None:
            code_context = {}
        else:
            code_context = dict(code_context)
        try:
            for k, v in vbt.imported_star.items():
                if k not in code_context:
                    code_context[k] = v
        except AttributeError:
            pass

        def _construct_tuple(loader, node):
            node = cls.decode_yaml_node(loader, "!tuple", node, **decode_node_kwargs)
            if not isinstance(node, Node):
                return node

            return tuple(loader.construct_sequence(node))

        def _construct_mapping(loader, node):
            node = cls.decode_yaml_node(loader, "tag:yaml.org,2002:map", node, **decode_node_kwargs)
            if not isinstance(node, Node):
                return node

            return _expand_dict(loader.construct_mapping(node, deep=True))

        def _split_key(key):
            parts, buf, escape = [], [], False
            for ch in key:
                if escape:
                    buf.append(ch)
                    escape = False
                elif ch == collapse_esc:
                    escape = True
                elif ch == collapse_sep:
                    parts.append("".join(buf))
                    buf.clear()
                else:
                    buf.append(ch)
            if escape:
                buf.append(collapse_esc)
            parts.append("".join(buf))
            return parts

        def _expand_dict(obj):
            if type(obj) not in (dict, CommentedMap):
                return obj

            out = {}
            for k, v in obj.items():
                if isinstance(k, str):
                    segs = _split_key(k)
                else:
                    segs = [k]
                cursor = out
                for seg in segs[:-1]:
                    nxt = cursor.setdefault(seg, {})
                    if type(nxt) not in (dict, CommentedMap):
                        cursor[seg] = nxt = {"": nxt}
                    cursor = nxt
                cursor[segs[-1]] = v
            return out

        def _construct_object(loader, tag, node):
            node = cls.decode_yaml_node(loader, tag, node, **decode_node_kwargs)
            if not isinstance(node, Node):
                return node

            if tag == "!class":
                class_id = loader.construct_scalar(node)
                if class_id.startswith("base64,"):
                    if not run_code:
                        raise ConstructorError(None, None, "Running code is disabled", node.start_mark)
                    return loads(base64.b64decode(class_id[7:]))
                if not use_class_ids:
                    raise ConstructorError(None, None, "Class references are disabled", node.start_mark)
                try:
                    return get_class_from_id(class_id)
                except ValueError as e:
                    raise ConstructorError(None, None, str(e), node.start_mark) from e

            if tag == "!pickle":
                if not run_code:
                    raise ConstructorError(None, None, "Running code is disabled", node.start_mark)
                value = loader.construct_scalar(node)
                if not value.startswith("base64,"):
                    raise ConstructorError(None, None, "Pickle payload must be base64-encoded", node.start_mark)
                return loads(base64.b64decode(value[7:]))

            if tag == "!expr":
                if not run_code:
                    raise ConstructorError(None, None, "Running code is disabled", node.start_mark)
                code = loader.construct_scalar(node)
                return evaluate(code, context=code_context)

            if tag.startswith("!rec:"):
                class_id = tag[len("!rec:") :]
                if class_id.startswith("base64,"):
                    if not run_code:
                        raise ConstructorError(None, None, "Running code is disabled", node.start_mark)
                    class_id = loads(base64.b64decode(class_id[7:]))
                elif not use_class_ids:
                    raise ConstructorError(None, None, "Class references are disabled", node.start_mark)
                mapping = _expand_dict(loader.construct_mapping(node, deep=True))
                if not pack_objects:
                    return mapping
                init_args = mapping.pop("init_args~", ())
                init_kwargs = mapping.pop("init_kwargs~", {})
                attr_dct = mapping.pop("attr_dct~", {})
                init_kwargs.update(mapping)
                rec_state = RecState(
                    init_args=init_args,
                    init_kwargs=init_kwargs,
                    attr_dct=attr_dct,
                )
                return reconstruct(class_id, rec_state)

            if isinstance(node, ScalarNode):
                return loader.construct_scalar(node)
            if isinstance(node, SequenceNode):
                return loader.construct_sequence(node)
            if isinstance(node, MappingNode):
                return _expand_dict(loader.construct_mapping(node, deep=True))
            if hasattr(loader, "construct_object"):
                return loader.construct_object(node)
            return node

        if use_ruamel:
            yaml_engine.load("null")
            constructor = yaml_engine.constructor
        else:
            constructor = loader_cls
        constructor.add_constructor("!tuple", _construct_tuple)
        constructor.add_constructor("tag:yaml.org,2002:map", _construct_mapping)
        constructor.add_multi_constructor("", _construct_object)

        if yaml_kwargs is None:
            yaml_kwargs = {}
        else:
            yaml_kwargs = dict(yaml_kwargs)
        if use_ruamel:
            obj = yaml_engine.load(yaml_str, **yaml_kwargs)
            yaml_engine.constructor.yaml_constructors.clear()
            yaml_engine.constructor.yaml_constructors.update(orig_yaml_constructors)
            yaml_engine.constructor.yaml_multi_constructors.clear()
            yaml_engine.constructor.yaml_multi_constructors.update(orig_yaml_mult_constructors)
        else:
            loader_cls = yaml_kwargs.pop("Loader", loader_cls)
            obj = yaml_engine.load(yaml_str, Loader=loader_cls, **yaml_kwargs)

        if isinstance(obj, dict) and not isinstance(obj, Pickleable):
            obj = reconstruct(cls, RecState(init_kwargs=obj))
        if check_type and not isinstance(obj, cls):
            raise TypeError(f"Decoded object must be an instance of {cls}, got {type(obj)}")
        return obj

    def encode_toml_node(self, key: tp.Any, value: tp.Any, **kwargs) -> tp.Any:
        """Encode a TOML node.

        This method is a hook to encode a value or prepare it for encoding before it's
        processed by the main TOML encoder. It can be overridden in subclasses to
        provide custom serialization logic for specific types.

        Args:
            key (Any): Key associated with the value (e.g., dictionary key or list index).
            value (Any): Value to encode.
            **kwargs: Keyword arguments for encoding.

        Returns:
            Any: Encoded or transformed value.
        """
        return value

    def encode_toml(
        self,
        root_key: tp.Optional[str] = None,
        unpack_objects: bool = True,
        compress_unpacked: bool = True,
        use_refs: bool = True,
        use_class_ids: bool = True,
        to_dict: bool = False,
        toml_kwargs: tp.KwargsLike = None,
        **encode_node_kwargs,
    ) -> str:
        """Serialize the instance to a TOML string with extended type support.

        This method converts the object into a TOML representation, using a custom
        schema with `__vbt_...` keys to handle complex types not natively
        supported by TOML. It can serialize custom objects, handle object
        references, and represent types using class identifiers.

        Special keys are used to represent different types:

        * `__vbt_ref__`: Reference to another object to avoid duplication.
        * `__vbt_tuple__`: Python tuple.
        * `__vbt_dict__`: Dictionary with non-string or reserved keys.
        * `__vbt_rec__`: `Pickleable` object's reconstruction state.
        * `__vbt_class__`: Python class type.
        * `__vbt_bytes__`: Bytes object, base64-encoded.
        * `__vbt_expr__`: Python literal expression.
        * `__vbt_pickle__`: Base64-encoded pickled object for unsupported types.

        !!! warning
            TOML doesn't guarantee the order of keys in tables.

        Args:
            root_key (Optional[str]): If provided, wraps the entire output in a table
                with this key.

                Dot-separated keys create nested tables.
            unpack_objects (bool): If True, `Pickleable` objects are serialized to their
                reconstruction state (`__vbt_rec__` key).

                Otherwise, they are pickled.
            compress_unpacked (bool): If True, simplifies the reconstruction state of
                unpacked objects by omitting default or empty components.
            use_refs (bool): If True, detects and replaces duplicate unhashable objects
                with references (`__vbt_ref__` key) to reduce redundancy.
            use_class_ids (bool): If True, serializes class objects using their
                registered identifiers (`__vbt_class__` and `__vbt_rec__` keys) instead of pickling them.
            to_dict (bool): If True, treats all dict-like objects as standard
                dictionaries during serialization.
            toml_kwargs (KwargsLike): Keyword arguments passed to `tomlkit.dumps`.
            **encode_node_kwargs: Keyword arguments for `Pickleable.encode_toml_node`.

        Returns:
            str: TOML-formatted string representing the instance.

        Raises:
            ValueError: If a `Pickleable` object is unpacked but lacks a valid
                reconstruction ID or state.
            ImportError: If `tomlkit` is not installed.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("tomlkit")

        import tomlkit
        from tomlkit.items import Item
        from tomlkit.exceptions import ConvertError

        K_REF = "__vbt_ref__"
        K_TUPLE = "__vbt_tuple__"
        K_DICT = "__vbt_dict__"
        K_REC = "__vbt_rec__"
        K_CLASS = "__vbt_class__"
        K_BYTES = "__vbt_bytes__"
        K_EXPR = "__vbt_expr__"
        K_PICKLE = "__vbt_pickle__"

        K_KEY = "__vbt_key__"
        K_VAL = "__vbt_val__"

        RESERVED = {K_REF, K_TUPLE, K_DICT, K_REC, K_CLASS, K_BYTES, K_EXPR, K_PICKLE}
        PRIMITIVE_TYPES = {bool, int, float, str, datetime.datetime, datetime.date, datetime.time}
        CONTAINER_TYPES = {dict, list, tuple}

        id_paths = {}
        id_keepalive = {}

        def _create_wrapper(path, key, value, extra=None):
            obj = {key: value}
            if extra:
                obj.update(extra)
            if not path:
                return obj

            if key in {K_REF, K_BYTES, K_EXPR, K_PICKLE} and not extra:
                tbl = tomlkit.inline_table()
                for k, v in obj.items():
                    tbl.add(k, v)
                return tbl
            return obj

        def _record_or_ref(path, value):
            if not use_refs or is_hashable(value):
                return None
            if id(value) in id_paths:
                return _create_wrapper(path, K_REF, list(id_paths[id(value)]))

            id_paths[id(value)] = list(path)
            id_keepalive[id(value)] = value
            return None

        def _encode(path, value):
            value = self.encode_toml_node(path, value, **encode_node_kwargs)
            if isinstance(value, Item):
                return value

            ref = _record_or_ref(path, value)
            if ref is not None:
                return ref

            if type(value) is dict or (to_dict and isinstance(value, dict)):
                if not all(isinstance(k, str) and k not in RESERVED for k in value):
                    kv_items = [
                        {
                            K_KEY: _encode(path + ["$key"], k),
                            K_VAL: _encode(path + [k], v),
                        }
                        for k, v in value.items()
                    ]
                    return {K_DICT: kv_items}
                return {k: _encode(path + [k], v) for k, v in value.items()}

            if type(value) is list:
                return [_encode(path + [i], item) for i, item in enumerate(value)]

            if type(value) is tuple:
                return {K_TUPLE: [_encode(path + [i], item) for i, item in enumerate(value)]}

            if isinstance(value, Pickleable) and unpack_objects:
                if use_class_ids:
                    class_id = get_id_from_class(value)
                else:
                    class_id = None
                if class_id is None:
                    class_id = "base64," + base64.b64encode(dumps(type(value))).decode("ascii")
                rec_state = value.rec_state
                if rec_state is None:
                    raise ValueError(f"Must define reconstruction state for {type(value)}")
                mapping = vars(rec_state)
                if compress_unpacked and (not mapping.get("init_args", ()) and not mapping.get("attr_dct", {})):
                    mapping = mapping.get("init_kwargs", {})
                else:
                    mapping = {k + "~": v for k, v in mapping.items()}
                encoded_body = {k: _encode(path + [k], v) for k, v in mapping.items()}
                return {K_REC: class_id, **encoded_body}

            if isinstance(value, type):
                if use_class_ids:
                    class_id = get_id_from_class(value)
                else:
                    class_id = None
                if class_id is None:
                    class_id = "base64," + base64.b64encode(dumps(value)).decode("ascii")
                return _create_wrapper(path, K_CLASS, class_id)

            if type(value) is bytes:
                return _create_wrapper(path, K_BYTES, "base64," + base64.b64encode(value).decode("ascii"))

            if (type(value) in PRIMITIVE_TYPES) or (not isinstance(value, tuple(PRIMITIVE_TYPES | CONTAINER_TYPES))):
                try:
                    return tomlkit.item(value)
                except ConvertError:
                    pass

            try:
                lit_repr = repr(value)
                if ast.literal_eval(lit_repr) == value:
                    return _create_wrapper(path, K_EXPR, lit_repr)
            except Exception:
                pass

            payload = "base64," + base64.b64encode(dumps(value)).decode("ascii")
            return _create_wrapper(path, K_PICKLE, payload)

        def _split_key(key):
            parts, buf, escape = [], [], False
            for ch in key:
                if escape:
                    buf.append(ch)
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == ".":
                    parts.append("".join(buf))
                    buf.clear()
                else:
                    buf.append(ch)

            if escape:
                buf.append("\\")
            parts.append("".join(buf))
            return parts

        if toml_kwargs is None:
            toml_kwargs = {}

        obj = self
        if root_key is not None:
            for seg in reversed(_split_key(root_key)):
                obj = {seg: obj}

        encoded = _encode([], obj)
        return tomlkit.dumps(encoded, **toml_kwargs)

    @classmethod
    def decode_toml_node(cls, key: tp.Any, value: tp.Any, **kwargs) -> tp.Any:
        """Decode a TOML node.

        This method is a hook to decode a value or prepare it for decoding after it's
        been parsed from the TOML string. It can be overridden in subclasses to
        provide custom deserialization logic for specific keys or value types.

        Args:
            key (Any): Key associated with the value (e.g., table key or list index).
            value (Any): Raw value from the parsed TOML data.
            **kwargs: Keyword arguments for decoding.

        Returns:
            Any: Decoded or transformed value.
        """
        return value

    @classmethod
    def decode_toml(
        cls: tp.Type[PickleableT],
        toml_str: str,
        run_code: bool = True,
        pack_objects: bool = True,
        use_class_ids: bool = True,
        use_refs: bool = True,
        code_context: tp.KwargsLike = None,
        toml_kwargs: tp.KwargsLike = None,
        check_type: bool = True,
        **decode_node_kwargs,
    ) -> PickleableT:
        """Deserialize a TOML string back into an instance of this class.

        This method parses a TOML string that was encoded with `encode_toml`,
        interpreting the custom `__vbt_...` schema to reconstruct complex Python
        objects. It builds and resolves a dependency graph to handle object
        references correctly.

        !!! warning
            Unpickling byte streams and running code has important security implications.
            Only parse TOML from trusted sources.

        Args:
            toml_str (str): TOML content to decode.
            run_code (bool): If True, executes code in `__vbt_expr__` wrappers and unpickles
                data from `__vbt_pickle__` wrappers.
            pack_objects (bool): If True, reconstructs `Pickleable` instances from
                `__vbt_rec__` wrappers.

                Otherwise, returns the raw state dictionary.
            use_class_ids (bool): If True, resolves class identifiers in `__vbt_class__`
                and `__vbt_rec__` wrappers to actual class objects.
            use_refs (bool): If True, resolves `__vbt_ref__` wrappers to the referenced objects.
            code_context (KwargsLike): Dictionary providing a namespace for evaluating
                code in `__vbt_expr__` wrappers.
            toml_kwargs (KwargsLike): Keyword arguments passed to `tomlkit.loads`.
            check_type (bool): If True, verifies that the final decoded object is an
                instance of this class and raises a `TypeError` if not.
            **decode_node_kwargs: Keyword arguments for `Pickleable.decode_toml_node`.

        Returns:
            Pickleable: Reconstructed object.

        Raises:
            TypeError: If `check_type` is True and the decoded object's type is incorrect.
            ValueError: If a reference cannot be resolved, a class ID is not found,
                or the dependency graph contains circular references.
            ImportError: If `tomlkit` is not installed.

        Examples:
            File `types.toml`:

            ```toml
            string = "hello world"
            boolean = false
            int = 123
            float = 123.45
            exp_float = 1.0e-10
            nan = nan
            inf = inf
            neg_inf = -inf
            none = { __vbt_expr__ = "None" }
            list = [1, 2, 3]

            collapsed_dict.a = 1
            collapsed_dict.b = 2
            collapsed_dict.c = 3

            "escaped_dict.a" = 1
            "escaped_dict.b" = 2
            "escaped_dict.c" = 3

            [dict]
            a = 1
            b = 2
            c = 3

            [tuple]
            __vbt_tuple__ = [1, 2, 3]

            [class]
            __vbt_class__ = "pd.Series"

            [instance]
            __vbt_rec__ = "pd.Series"
            data = { __vbt_ref__ = "list" }

            [expl_instance]
            __vbt_rec__ = "pd.Timedelta"
            "init_args~" = []
            "init_kwargs~" = { days = 1 }
            "attr_dct~" = {}

            [pickle]
            __vbt_pickle__ = "base64,gASVSAAAAAAAAACMHnBhbmRhcy5fbGlicy50c2xpYnMudGltZWRlbHRhc5SME190aW1lZGVsdGFfdW5waWNrbGWUk5SKBgAAT5GUTksKhpRSlC4="

            [expression]
            __vbt_expr__ = "dict(sub_dict2=dict(some='value'))"

            [mult_expression]
            __vbt_expr__ = "import math\\n\\nmath.floor(1.5)"
            ```

            ```pycon
            >>> from vectorbtpro import *

            >>> vbt.pprint(vbt.pdict.load("types.toml"))
            pdict({
                'string': 'hello world',
                'boolean': False,
                'int': 123,
                'float': 123.45,
                'exp_float': 1e-10,
                'nan': np.nan,
                'inf': np.inf,
                'neg_inf': -np.inf,
                'none': None,
                'list': [
                    1,
                    2,
                    3
                ],
                'collapsed_dict': dict(
                    a=1,
                    b=2,
                    c=3
                ),
                'escaped_dict.a': 1,
                'escaped_dict.b': 2,
                'escaped_dict.c': 3,
                'dict': dict(
                    a=1,
                    b=2,
                    c=3
                ),
                'tuple': (
                    1,
                    2,
                    3
                ),
                'class': <class 'pandas.core.series.Series'>,
                'instance': <pandas.core.series.Series object at 0x14f782b10 with shape (3,)>,
                'expl_instance': Timedelta('1 days 00:00:00'),
                'pickle': Timedelta('1 days 00:00:00'),
                'expression': dict(
                    sub_dict2=dict(
                        some='value'
                    )
                ),
                'mult_expression': 1
            })
            ```
        """
        from vectorbtpro.utils.module_ import assert_can_import
        from vectorbtpro.utils.eval_ import evaluate

        assert_can_import("tomlkit")

        import tomlkit

        K_REF = "__vbt_ref__"
        K_TUPLE = "__vbt_tuple__"
        K_DICT = "__vbt_dict__"
        K_REC = "__vbt_rec__"
        K_CLASS = "__vbt_class__"
        K_BYTES = "__vbt_bytes__"
        K_EXPR = "__vbt_expr__"
        K_PICKLE = "__vbt_pickle__"

        K_KEY = "__vbt_key__"
        K_VAL = "__vbt_val__"

        if toml_kwargs is None:
            toml_kwargs = {}
        doc = tomlkit.loads(toml_str, **toml_kwargs)
        raw_data = doc.unwrap()

        if code_context is None:
            code_context = {}
        else:
            code_context = dict(code_context)
        try:
            for k, v in vbt.imported_star.items():
                if k not in code_context:
                    code_context[k] = v
        except AttributeError:
            pass

        def _split_key(key):
            parts, buf, escape = [], [], False
            for ch in key:
                if escape:
                    buf.append(ch)
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == ".":
                    parts.append("".join(buf))
                    buf.clear()
                else:
                    buf.append(ch)

            if escape:
                buf.append("\\")
            parts.append("".join(buf))
            return parts

        graph = {}
        nodes = {}

        def build_graph(path, value):
            nodes[path] = value
            if path not in graph:
                graph[path] = set()

            if isinstance(value, dict):
                if K_REF in value:
                    if use_refs:
                        ref_data = value[K_REF]
                        if isinstance(ref_data, str):
                            path_segments = _split_key(ref_data)
                        else:
                            path_segments = ref_data
                        target_path = tuple(path_segments)
                        graph[path].add(target_path)
                    return

                if K_REC in value:
                    body = {k: v for k, v in value.items() if k != K_REC}
                    for k, v_child in body.items():
                        child_path = path + (k,)
                        graph[path].add(child_path)
                        build_graph(child_path, v_child)
                    return

                if K_TUPLE in value:
                    for i, item in enumerate(value[K_TUPLE]):
                        child_path = path + (i,)
                        graph[path].add(child_path)
                        build_graph(child_path, item)
                    return

                if K_DICT in value:
                    for i, item in enumerate(value[K_DICT]):
                        key_path = path + (i, K_KEY)
                        val_path = path + (i, K_VAL)
                        graph[path].add(key_path)
                        graph[path].add(val_path)
                        build_graph(key_path, item[K_KEY])
                        build_graph(val_path, item[K_VAL])
                    return

                if any(k in value for k in [K_CLASS, K_BYTES, K_EXPR, K_PICKLE]):
                    return

                for k, v_child in value.items():
                    child_path = path + (k,)
                    graph[path].add(child_path)
                    build_graph(child_path, v_child)
            elif isinstance(value, list):
                for i, v_child in enumerate(value):
                    child_path = path + (i,)
                    graph[path].add(child_path)
                    build_graph(child_path, v_child)

        build_graph((), raw_data)

        sorter = TopologicalSorter(graph)
        try:
            topo_order = list(sorter.static_order())
        except Exception as e:
            raise ValueError("Failed to resolve dependency graph. Check for circular references.") from e

        resolved_nodes = {}

        for path in topo_order:
            raw_node = nodes[path]
            key = path[-1] if path else None

            value = cls.decode_toml_node(key, raw_node, **decode_node_kwargs)

            resolved_value = None
            if isinstance(value, dict):
                if K_REF in value:
                    if not use_refs:
                        raise ValueError(f"Node {path}: References are disabled")
                    ref_data = value[K_REF]
                    if isinstance(ref_data, str):
                        path_segments = _split_key(ref_data)
                    else:
                        path_segments = ref_data
                    target_path = tuple(path_segments)
                    try:
                        resolved_value = resolved_nodes[target_path]
                    except KeyError:
                        raise ValueError(f"Node {path}: Failed to resolve reference to path {target_path}") from None

                elif K_REC in value:
                    body = {k: v for k, v in value.items() if k != K_REC}
                    if not pack_objects:
                        resolved_value = {k: resolved_nodes[path + (k,)] for k in body}
                    else:
                        class_id = value[K_REC]
                        if class_id.startswith("base64,"):
                            if not run_code:
                                raise ValueError(f"Node {path}: Running code is disabled")
                            class_id = loads(base64.b64decode(class_id[7:]))
                        elif not use_class_ids:
                            raise ValueError(f"Node {path}: Class ID resolution is disabled")

                        resolved_body = {k: resolved_nodes[path + (k,)] for k in body}

                        if any(k.endswith("~") for k in resolved_body):
                            init_args = resolved_body.pop("init_args~", ())
                            init_kwargs = resolved_body.pop("init_kwargs~", {})
                            attr_dct = resolved_body.pop("attr_dct~", {})
                        else:
                            init_args, attr_dct = (), {}
                            init_kwargs = resolved_body
                        rec_state = RecState(init_args=init_args, init_kwargs=init_kwargs, attr_dct=attr_dct)
                        resolved_value = reconstruct(class_id, rec_state)

                elif K_CLASS in value:
                    class_id = value[K_CLASS]
                    if class_id.startswith("base64,"):
                        if not run_code:
                            raise ValueError(f"Node {path}: Running code is disabled")
                        resolved_value = loads(base64.b64decode(class_id[7:]))
                    else:
                        if not use_class_ids:
                            raise ValueError(f"Node {path}: Class ID resolution is disabled")
                        resolved_value = get_class_from_id(class_id)

                elif K_TUPLE in value:
                    resolved_value = tuple(resolved_nodes[path + (i,)] for i in range(len(value[K_TUPLE])))

                elif K_DICT in value:
                    res = {}
                    for i, item in enumerate(value[K_DICT]):
                        d_key = resolved_nodes[path + (i, K_KEY)]
                        d_val = resolved_nodes[path + (i, K_VAL)]
                        res[d_key] = d_val
                    resolved_value = res

                elif K_BYTES in value:
                    if not value[K_BYTES].startswith("base64,"):
                        raise ValueError(f"Node {path}: Invalid bytes format")
                    resolved_value = base64.b64decode(value[K_BYTES][7:])

                elif K_EXPR in value:
                    if not run_code:
                        raise ValueError(f"Node {path}: Running code is disabled")
                    resolved_value = evaluate(value[K_EXPR], context=code_context)

                elif K_PICKLE in value:
                    if not run_code:
                        raise ValueError(f"Node {path}: Running code is disabled")
                    if not value[K_PICKLE].startswith("base64,"):
                        raise ValueError(f"Node {path}: Invalid pickle format")
                    resolved_value = loads(base64.b64decode(value[K_PICKLE][7:]))

                else:
                    resolved_value = {k: resolved_nodes[path + (k,)] for k in value}

            elif isinstance(value, list):
                resolved_value = [resolved_nodes[path + (i,)] for i in range(len(value))]
            else:
                resolved_value = value

            resolved_nodes[path] = resolved_value

        obj = resolved_nodes.get((), None)
        if isinstance(obj, dict) and not isinstance(obj, Pickleable):
            obj = reconstruct(cls, RecState(init_kwargs=obj))

        if check_type and not isinstance(obj, cls):
            raise TypeError(f"Decoded object must be an instance of {cls}, got {type(obj)}")
        return obj

    @classmethod
    def resolve_file_path(
        cls,
        path: tp.Optional[tp.PathLike] = None,
        file_format: tp.Optional[str] = None,
        compression: tp.CompressionLike = None,
        for_save: bool = False,
    ) -> Path:
        """Resolve a file path ensuring valid file format and optional compression.

        File format and compression can be provided either via a suffix in `path`,
        or via the argument `file_format` and `compression` respectively.

        !!! note
            When saving, default `file_format` and `compression` values are taken from
            `vectorbtpro._settings.pickling`. When loading, the function searches for matching
            files in the current directory.

        !!! info
            For default settings, see `vectorbtpro._settings.pickling`.

        Args:
            path (Optional[PathLike]): File path, directory, or None.

                If None or a directory, the file name defaults to the class name.
            file_format (Optional[str]): Format specifier for determining the file extension.

                For options, see `extensions.serialization` in `vectorbtpro._settings.pickling`.
            compression (CompressionLike): Compression algorithm.

                See `compress`.
            for_save (bool): Resolve the file path for saving if True; otherwise, for loading.

        Returns:
            Path: Resolved file path with the appropriate extensions.
        """
        from vectorbtpro._settings import settings

        pickling_cfg = settings["pickling"]

        default_file_format = pickling_cfg["file_format"]
        default_compression = pickling_cfg["compression"]
        if isinstance(compression, bool) and compression:
            compression = default_compression
            if compression is None:
                raise ValueError("Set default compression in settings")

        if path is None:
            path = cls.__name__
        path = Path(path)
        if path.is_dir():
            path /= cls.__name__

        serialization_extensions = get_serialization_extensions()
        compression_extensions = get_compression_extensions()
        suffixes = [suffix[1:].lower() for suffix in path.suffixes]
        if len(suffixes) > 2:
            raise ValueError("Only two file extensions are supported: file format and compression")
        if len(suffixes) >= 1:
            if file_format is not None:
                raise ValueError("File format is already provided via file extension")
            file_format = suffixes[0]
        if len(suffixes) == 2:
            if compression is not None:
                raise ValueError("Compression is already provided via file extension")
            compression = suffixes[1]
        if file_format is not None:
            file_format = file_format.lower()
            if file_format not in serialization_extensions:
                raise ValueError(f"Invalid file_format: {file_format!r}")
        if compression not in (None, False):
            compression = compression.lower()
            if compression not in compression_extensions:
                raise ValueError(f"Invalid compression: {compression!r}")
        for _ in range(len(suffixes)):
            path = path.with_suffix("")

        if for_save:
            new_suffixes = []
            if file_format is None:
                file_format = default_file_format
            new_suffixes.append(file_format)
            if compression is None and file_format in get_serialization_extensions("pickle"):
                compression = default_compression
            if compression not in (None, False):
                if file_format not in get_serialization_extensions("pickle"):
                    raise ValueError("Compression can be used only with pickling")
                new_suffixes.append(compression)
            new_path = path.with_suffix("." + ".".join(new_suffixes))
            return new_path

        def _extensions_match(a, b, ext_type):
            from vectorbtpro._settings import settings

            pickling_cfg = settings["pickling"]

            for extensions in pickling_cfg["extensions"][ext_type].values():
                if a in extensions and b in extensions:
                    return True
            return False

        if file_format is not None:
            if compression not in (None, False):
                new_path = path.with_suffix(f".{file_format}.{compression}")
                if new_path.exists():
                    return new_path
            elif default_compression not in (None, False):
                new_path = path.with_suffix(f".{file_format}.{default_compression}")
                if new_path.exists():
                    return new_path
            else:
                new_path = path.with_suffix(f".{file_format}")
                if new_path.exists():
                    return new_path
        else:
            if compression not in (None, False):
                new_path = path.with_suffix(f".{default_file_format}.{compression}")
                if new_path.exists():
                    return new_path
            elif default_compression not in (None, False):
                new_path = path.with_suffix(f".{default_file_format}.{default_compression}")
                if new_path.exists():
                    return new_path
            else:
                new_path = path.with_suffix(f".{default_file_format}")
                if new_path.exists():
                    return new_path

        paths = []
        for p in path.parent.iterdir():
            if p.is_file():
                if p.stem.split(".")[0] == path.stem.split(".")[0]:
                    suffixes = [suffix[1:].lower() for suffix in p.suffixes]
                    if len(suffixes) == 0:
                        continue
                    if file_format is None:
                        if suffixes[0] not in serialization_extensions:
                            continue
                    else:
                        if not _extensions_match(suffixes[0], file_format, "serialization"):
                            continue
                    if compression is False:
                        if len(suffixes) >= 2:
                            continue
                    elif compression is None:
                        if len(suffixes) >= 2 and suffixes[1] not in compression_extensions:
                            continue
                    else:
                        if len(suffixes) == 1 or not _extensions_match(suffixes[1], compression, "compression"):
                            continue
                    paths.append(p)
        if len(paths) == 1:
            return paths[0]
        if len(paths) > 1:
            raise ValueError(
                f"Multiple files found under path {str(path.resolve())!r}: {paths}. Please provide an extension."
            )
        error_message = f"No file found with path {str(path.resolve())!r}"
        if file_format is not None:
            error_message += f", file format {file_format!r}"
        if compression not in (None, False):
            error_message += f", compression {compression!r}"
        raise FileNotFoundError(error_message)

    @classmethod
    def file_exists(cls, *args, **kwargs) -> bool:
        """Return whether a file exists.

        Args:
            *args: Positional arguments for `Pickleable.resolve_file_path`.
            **kwargs: Keyword arguments for `Pickleable.resolve_file_path`.
        """
        try:
            cls.resolve_file_path(*args, **kwargs)
            return True
        except FileNotFoundError:
            return False

    def save(
        self,
        path: tp.Optional[tp.PathLike] = None,
        file_format: tp.Optional[str] = None,
        compression: tp.CompressionLike = None,
        mkdir_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> Path:
        """Serialize and save the instance to a file.

        File path resolution is performed using `Pickleable.resolve_file_path`.

        Args:
            path (Optional[PathLike]): File path to save the instance.
            file_format (Optional[str]): Format specifier for determining the file extension.
            compression (CompressionLike): Compression algorithm.

                See `compress`.
            mkdir_kwargs (KwargsLike): Keyword arguments for directory creation.

                See `vectorbtpro.utils.path_.check_mkdir`.
            **kwargs: Keyword arguments for `Pickleable.dumps` for pickle extensions,
                `Pickleable.encode_config` for config extensions, `Pickleable.encode_yaml`
                for YAML extensions, and `Pickleable.encode_toml` for TOML extensions.

        Returns:
            Path: File path where the instance was saved.
        """
        if mkdir_kwargs is None:
            mkdir_kwargs = {}

        path = self.resolve_file_path(path=path, file_format=file_format, compression=compression, for_save=True)
        suffixes = [suffix[1:].lower() for suffix in path.suffixes]
        if suffixes[0] in get_serialization_extensions("pickle"):
            if compression is None:
                suffixes = [suffix[1:].lower() for suffix in path.suffixes]
                if len(suffixes) > 0 and suffixes[-1] in get_compression_extensions():
                    compression = suffixes[-1]
            bytes_ = self.dumps(compression=compression, **kwargs)
            check_mkdir(path.parent, **mkdir_kwargs)
            with open(path, "wb") as f:
                f.write(bytes_)
        elif suffixes[0] in get_serialization_extensions("config"):
            config_str = self.encode_config(**kwargs)
            check_mkdir(path.parent, **mkdir_kwargs)
            with open(path, "w") as f:
                f.write(config_str)
        elif suffixes[0] in get_serialization_extensions("yaml"):
            config_str = self.encode_yaml(**kwargs)
            check_mkdir(path.parent, **mkdir_kwargs)
            with open(path, "w") as f:
                f.write(config_str)
        elif suffixes[0] in get_serialization_extensions("toml"):
            config_str = self.encode_toml(**kwargs)
            check_mkdir(path.parent, **mkdir_kwargs)
            with open(path, "w") as f:
                f.write(config_str)
        else:
            raise ValueError(f"Invalid file extension: {path.suffix!r}")
        return path

    @classmethod
    def load(
        cls: tp.Type[PickleableT],
        path: tp.Optional[tp.PathLike] = None,
        file_format: tp.Optional[str] = None,
        compression: tp.CompressionLike = None,
        **kwargs,
    ) -> PickleableT:
        """Deserialize and return an instance from a file.

        File path resolution is performed using `Pickleable.resolve_file_path`.

        Args:
            path (Optional[PathLike]): Path of the file to load.
            file_format (Optional[str]): Format specifier for determining the file extension.
            compression (CompressionLike): Compression algorithm.

                See `compress`.
            **kwargs: Keyword arguments for `Pickleable.loads` for pickle extensions,
                `Pickleable.decode_config` for config extensions, `Pickleable.decode_yaml`
                for YAML extensions, and `Pickleable.decode_toml` for TOML extensions.

        Returns:
            Pickleable: Deserialized instance.
        """
        path = cls.resolve_file_path(path=path, file_format=file_format, compression=compression)
        suffixes = [suffix[1:].lower() for suffix in path.suffixes]
        if suffixes[0] in get_serialization_extensions("pickle"):
            if compression is None:
                suffixes = [suffix[1:].lower() for suffix in path.suffixes]
                if len(suffixes) > 0 and suffixes[-1] in get_compression_extensions():
                    compression = suffixes[-1]
            with open(path, "rb") as f:
                bytes_ = f.read()
            return cls.loads(bytes_, compression=compression, **kwargs)
        elif suffixes[0] in get_serialization_extensions("config"):
            with open(path, "r") as f:
                config_str = f.read()
            return cls.decode_config(config_str, **kwargs)
        elif suffixes[0] in get_serialization_extensions("yaml"):
            with open(path, "r") as f:
                config_str = f.read()
            return cls.decode_yaml(config_str, **kwargs)
        elif suffixes[0] in get_serialization_extensions("toml"):
            with open(path, "r") as f:
                config_str = f.read()
            return cls.decode_toml(config_str, **kwargs)
        else:
            raise ValueError(f"Invalid file extension: {path.suffix!r}")

    @property
    def rec_state(self) -> tp.Optional[RecState]:
        """Reconstruction state for recreating the object.

        Returns:
            Optional[RecState]: Reconstruction state used for object reconstruction.
        """
        return None

    @classmethod
    def modify_state(cls, rec_state: RecState) -> RecState:
        """Modify the reconstruction state prior to object reconstruction.

        Args:
            rec_state (RecState): Original reconstruction state.

        Returns:
            RecState: Modified reconstruction state.
        """
        return rec_state

    def __reduce__(self) -> tp.Union[str, tp.Tuple]:
        rec_state = self.rec_state
        if rec_state is None:
            return object.__reduce__(self)
        class_id = get_id_from_class(self)
        if class_id is None:
            cls = type(self)
        else:
            cls = class_id
        return reconstruct, (cls, rec_state)
    
    def __sizeof__(self) -> int:
        return len(self.dumps())

    def getsize(self, readable: bool = True, **kwargs) -> tp.Union[str, int]:
        """Return the size of this object.

        Args:
            readable (bool): Whether to use a human-readable format.
            **kwargs: Keyword arguments for `humanize.naturalsize`.

        Returns:
            Union[str, int]: Object's size as a human-readable string if `readable` is True,
                otherwise as an integer in bytes.
        """
        if readable:
            return humanize.naturalsize(self.__sizeof__(), **kwargs)
        return self.__sizeof__()

    def digest(self) -> str:
        """Return the hash digest of the object's serialized form.
        
        Returns:
            str: Hash digest of the object.
        """
        return hashlib.blake2b(self.dumps(), digest_size=16).hexdigest()


pdictT = tp.TypeVar("pdictT", bound="pdict")


class pdict(Comparable, Pickleable, Prettified, dict):
    """Class for a pickleable dictionary that supports comparison, serialization, and prettification."""

    def load_update(self, path: tp.Optional[tp.PathLike] = None, clear: bool = False, **kwargs) -> None:
        """Load serialized data from a file and update this dictionary instance in place.

        Args:
            path (Optional[PathLike]): File path to load data from.
            clear (bool): If True, clear the existing dictionary before updating.
            **kwargs: Keyword arguments for `pdict.load`.

        Returns:
            None
        """
        if clear:
            self.clear()
        self.update(self.load(path=path, **kwargs))

    @property
    def rec_state(self) -> tp.Optional[RecState]:
        init_args = ()
        init_kwargs = dict(self)
        for k in list(init_kwargs):
            if not isinstance(k, str):
                if len(init_args) == 0:
                    init_args = (dict(),)
                init_args[0][k] = init_kwargs.pop(k)
        return RecState(init_args=init_args, init_kwargs=init_kwargs)

    def equals(
        self,
        other: tp.Any,
        check_types: bool = True,
        _key: tp.Optional[str] = None,
        **kwargs,
    ) -> bool:
        """Perform a deep equality check between this dictionary and another object.

        Args:
            other (Any): Object to compare against.
            check_types (bool): Whether to verify types during comparison.
            **kwargs: Keyword arguments for `vectorbtpro.utils.checks.is_deep_equal`.

        Returns:
            bool: True if the objects are deeply equal, otherwise False.
        """
        if _key is None:
            _key = type(self).__name__
        if "only_types" in kwargs:
            del kwargs["only_types"]
        if check_types and not is_deep_equal(
            self,
            other,
            _key=_key,
            only_types=True,
            **kwargs,
        ):
            return False
        return is_deep_equal(
            dict(self),
            dict(other),
            _key=_key,
            **kwargs,
        )

    def prettify(self, **kwargs) -> str:
        return prettify_dict(self, **kwargs)

    def __repr__(self):
        return type(self).__name__ + "(" + repr(dict(self)) + ")"
