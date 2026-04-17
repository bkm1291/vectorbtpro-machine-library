# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing classes and utilities for storing documents."""

import hashlib
import inspect
import sys
from collections.abc import MutableMapping
from pathlib import Path

from vectorbtpro import _typing as tp
from vectorbtpro.knowledge.text_splitting import split_text
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.config import merge_dicts, flat_merge_dicts, Configured
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.template import CustomTemplate, SafeSub, RepFunc

if tp.TYPE_CHECKING:
    from lmdbm import Lmdb as LmdbT
else:
    LmdbT = "lmdbm.Lmdb"

__all__ = [
    "StoreObject",
    "StoreData",
    "StoreDocument",
    "TextDocument",
    "StoreEmbedding",
    "ObjectStore",
    "DictStore",
    "MemoryStore",
    "FileStore",
    "LMDBStore",
]


StoreObjectT = tp.TypeVar("StoreObjectT", bound="StoreObject")


@define
class StoreObject(DefineMixin):
    """Class representing an object managed by a store."""

    id_: str = define.field()
    """Object identifier."""

    @property
    def hash_key(self) -> tuple:
        return (self.id_,)


StoreDataT = tp.TypeVar("StoreDataT", bound="StoreData")


@define
class StoreData(StoreObject, DefineMixin):
    """Class for any data to be stored.

    Accepts the same arguments as in `StoreObject` + the ones listed below.
    """

    data: tp.Any = define.field()
    """Stored data."""

    @classmethod
    def id_from_data(cls, data: tp.Any) -> str:
        """Return a unique identifier computed from the given data.

        Args:
            data (Any): Data from which to generate the identifier.

        Returns:
            str: Unique cache key as a hexadecimal string.
        """
        from vectorbtpro.utils.pickling import dumps

        return hashlib.blake2b(dumps(data), digest_size=16).hexdigest()

    @classmethod
    def from_data(
        cls: tp.Type[StoreDataT],
        data: tp.Any,
        id_: tp.Optional[str] = None,
        **kwargs,
    ) -> StoreDataT:
        """Return a new instance of `StoreData` derived from the provided data.

        Args:
            data (Any): Data to store.
            id_ (Optional[str]): Optional identifier; if not provided, one is generated.
            **kwargs: Keyword arguments for `StoreData`.

        Returns:
            StoreData: New instance of `StoreData`.
        """
        if id_ is None:
            id_ = cls.id_from_data(data)
        return cls(id_, data, **kwargs)

    def __attrs_post_init__(self):
        if self.id_ is None:
            new_id = self.id_from_data(self.data)
            object.__setattr__(self, "id_", new_id)


StoreDocumentT = tp.TypeVar("StoreDocumentT", bound="StoreDocument")


@define
class StoreDocument(StoreData, DefineMixin):
    """Abstract class for documents to be stored."""

    template_context: tp.KwargsLike = define.field(factory=dict)
    """Context for substituting template variables."""

    def get_content(self, for_embed: bool = False) -> tp.Optional[str]:
        """Return the document content.

        !!! abstract
            This method should be overridden in a subclass.

        Returns:
            Optional[str]: Content if available, otherwise None.
        """
        raise NotImplementedError

    def split(self: StoreDocumentT) -> tp.List[StoreDocumentT]:
        """Return a list of document instances resulting from splitting the current document.

        !!! abstract
            This method should be overridden in a subclass.

        Returns:
            List[StoreDocument]: List of document chunks.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return self.get_content()


TextDocumentT = tp.TypeVar("TextDocumentT", bound="TextDocument")


def def_metadata_template(metadata_content: str) -> str:
    """Return a formatted metadata template string.

    Args:
        metadata_content (str): Metadata content to include.

    Returns:
        str: Formatted metadata template string with front matter delimiters.
    """
    if metadata_content.endswith("\n"):
        return "---\n{metadata_content}---\n\n".format(metadata_content=metadata_content)
    return "---\n{metadata_content}\n---\n\n".format(metadata_content=metadata_content)


@define
class TextDocument(StoreDocument, DefineMixin):
    """Class for text documents."""

    text_path: tp.Optional[tp.PathLikeKey] = define.field(default=None)
    """Path to the text field within the data."""

    split_text_kwargs: tp.KwargsLike = define.field(factory=dict)
    """Keyword arguments for `vectorbtpro.knowledge.text_splitting.split_text`."""

    excl_metadata: tp.Union[bool, tp.MaybeList[tp.PathLikeKey]] = define.field(default=False)
    """Indicates whether to exclude metadata or specify fields to exclude. 

    If False, metadata includes all fields except text.
    """

    excl_embed_metadata: tp.Union[None, bool, tp.MaybeList[tp.PathLikeKey]] = define.field(default=None)
    """Indicates whether to exclude metadata for embeddings; if None, defaults to `excl_metadata`."""

    skip_missing: bool = define.field(default=True)
    """Determines whether missing text or metadata returns None instead of raising an error."""

    dump_kwargs: tp.KwargsLike = define.field(factory=dict)
    """Keyword arguments for the dump formatting function."""

    metadata_template: tp.CustomTemplateLike = define.field(
        default=RepFunc(def_metadata_template, eval_id="metadata_template")
    )
    """Template for formatting metadata via the `format()` method."""

    content_template: tp.CustomTemplateLike = define.field(
        default=SafeSub("${metadata_content}${text}", eval_id="content_template")
    )
    """Template for formatting content via the `format()` method."""

    def get_text(self) -> tp.Optional[str]:
        """Return the text content of the document.

        Returns:
            Optional[str]: Document's text, or None if not available.
        """
        from vectorbtpro.utils.search_ import get_pathlike_key

        if self.data is None:
            return None
        if isinstance(self.data, str):
            return self.data
        if self.text_path is not None:
            try:
                text = get_pathlike_key(self.data, self.text_path, keep_path=False)
            except (KeyError, IndexError, AttributeError):
                if not self.skip_missing:
                    raise
                return None
            if text is None:
                return None
            if not isinstance(text, str):
                raise TypeError(f"Text field must be a string, not {type(text)}")
            return text
        raise TypeError(f"If text path is not provided, data item must be a string, not {type(self.data)}")

    def get_metadata(self, for_embed: bool = False) -> tp.Optional[tp.Any]:
        """Return the metadata extracted from the document's data.

        Args:
            for_embed (bool): Flag indicating if metadata for embeddings should be retrieved.

        Returns:
            Optional[Any]: Metadata if available, otherwise None.
        """
        from vectorbtpro.utils.search_ import remove_pathlike_key

        if self.data is None or isinstance(self.data, str) or self.text_path is None:
            return None
        prev_keys = []
        data = self.data
        try:
            data = remove_pathlike_key(data, self.text_path, make_copy=True, prev_keys=prev_keys)
        except (KeyError, IndexError, AttributeError):
            if not self.skip_missing:
                raise
        excl_metadata = self.excl_metadata
        if for_embed:
            excl_embed_metadata = self.excl_embed_metadata
            if excl_embed_metadata is None:
                excl_embed_metadata = excl_metadata
            excl_metadata = excl_embed_metadata
        if isinstance(excl_metadata, bool):
            if excl_metadata:
                return None
            return data
        if not excl_metadata:
            return data
        if not isinstance(excl_metadata, list):
            excl_metadata = [excl_metadata]
        for p in excl_metadata:
            try:
                data = remove_pathlike_key(data, p, make_copy=True, prev_keys=prev_keys)
            except (KeyError, IndexError, AttributeError):
                continue
        return data

    def get_metadata_content(self, for_embed: bool = False) -> tp.Optional[str]:
        """Return the metadata content as a formatted string.

        Args:
            for_embed (bool): Flag indicating if metadata for embeddings should be retrieved.

        Returns:
            Optional[str]: Formatted metadata content, or None if metadata is missing.
        """
        from vectorbtpro.utils.formatting import dump

        metadata = self.get_metadata(for_embed=for_embed)
        if metadata is None:
            return None
        return dump(metadata, **self.dump_kwargs)

    def get_content(self, for_embed: bool = False) -> tp.Optional[str]:
        text = self.get_text()
        metadata_content = self.get_metadata_content(for_embed=for_embed)
        if text is None and metadata_content is None:
            return None
        if text is None:
            text = ""
        if metadata_content is None:
            metadata_content = ""
        if metadata_content:
            metadata_template = self.metadata_template
            if isinstance(metadata_template, str):
                metadata_template = SafeSub(metadata_template)
            elif checks.is_function(metadata_template):
                metadata_template = RepFunc(metadata_template)
            elif not isinstance(metadata_template, CustomTemplate):
                raise TypeError("Metadata template must be a string, function, or template")
            template_context = flat_merge_dicts(
                dict(metadata_content=metadata_content),
                self.template_context,
            )
            metadata_content = metadata_template.substitute(template_context, eval_id="metadata_template")
        content_template = self.content_template
        if isinstance(content_template, str):
            content_template = SafeSub(content_template)
        elif checks.is_function(content_template):
            content_template = RepFunc(content_template)
        elif not isinstance(content_template, CustomTemplate):
            raise TypeError("Content template must be a string, function, or template")
        template_context = flat_merge_dicts(
            dict(metadata_content=metadata_content, text=text),
            self.template_context,
        )
        return content_template.substitute(template_context, eval_id="content_template")

    def split(self: TextDocumentT) -> tp.List[TextDocumentT]:
        from vectorbtpro.utils.search_ import set_pathlike_key

        text = self.get_text()
        if text is None:
            return [self]
        text_chunks = split_text(text, **self.split_text_kwargs)
        document_chunks = []
        for text_chunk in text_chunks:
            if not isinstance(self.data, str) and self.text_path is not None:
                data_chunk = set_pathlike_key(
                    self.data,
                    self.text_path,
                    text_chunk,
                    make_copy=True,
                )
            else:
                data_chunk = text_chunk
            document_chunks.append(self.replace(data=data_chunk, id_=None))
        return document_chunks


@define
class StoreEmbedding(StoreObject, DefineMixin):
    """Class for embeddings to be stored."""

    parent_id: tp.Optional[str] = define.field(default=None)
    """Identifier of the parent object."""

    child_ids: tp.List[str] = define.field(factory=list)
    """List of identifiers for the child objects."""

    embedding: tp.Optional[tp.List[int]] = define.field(default=None, repr=lambda x: f"List[{len(x)}]" if x else None)
    """Embedding vector."""


class MetaObjectStore(type(Configured), type(MutableMapping)):
    """Metaclass for `ObjectStore`.

    Serves as the metaclass combining configuration from `vectorbtpro.utils.config.Configured`
    and mutable mapping behavior.
    """

    pass


class ObjectStore(Configured, MutableMapping, metaclass=MetaObjectStore):
    """Class for managing an object store.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and
        its sub-configurations `chat` and `chat.obj_store_config`.

    Args:
        store_id (Optional[str]): Identifier for the store.
        purge_on_open (Optional[bool]): Indicates if the store should be purged upon opening.
        template_context (KwargsLike): Additional context for template substitution.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name identifier for the store class."""

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat", "knowledge.chat.obj_store_config"]

    def __init__(
        self,
        store_id: tp.Optional[str] = None,
        purge_on_open: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            store_id=store_id,
            purge_on_open=purge_on_open,
            template_context=template_context,
            **kwargs,
        )

        store_id = self.resolve_setting(store_id, "store_id")
        purge_on_open = self.resolve_setting(purge_on_open, "purge_on_open")
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        self._store_id = store_id
        self._purge_on_open = purge_on_open
        self._template_context = template_context

        self._opened = False
        self._enter_calls = 0

    @property
    def store_id(self) -> str:
        """Store identifier.

        Returns:
            str: Unique identifier of the store.
        """
        return self._store_id

    @property
    def purge_on_open(self) -> bool:
        """Flag indicating whether the store should be purged upon opening.

        Returns:
            bool: True if the store will be purged on open; otherwise, False.
        """
        return self._purge_on_open

    @property
    def template_context(self) -> tp.Kwargs:
        """Additional context for template substitution.

        Returns:
            Kwargs: Dictionary of context variables for template substitution.
        """
        return self._template_context

    @property
    def opened(self) -> bool:
        """Indicates whether the store is currently open.

        Returns:
            bool: True if the store is open; otherwise, False.
        """
        return self._opened

    @property
    def enter_calls(self) -> int:
        """Number of times the store has been entered.

        Returns:
            int: Count of how many times the store's context has been entered.
        """
        return self._enter_calls

    @property
    def mirror_store_id(self) -> tp.Optional[str]:
        """Mirror store identifier.

        Returns:
            Optional[str]: Mirror store ID if applicable; otherwise, None.
        """
        return None

    def open(self) -> None:
        """Open the store.

        If already open, close it first; purge if `purge_on_open` is True.

        Returns:
            None
        """
        if self.opened:
            self.close()
        if self.purge_on_open:
            self.purge()
        self._opened = True

    def check_opened(self) -> None:
        """Ensure the store is open; raise an exception if it is not.

        Returns:
            None
        """
        if not self.opened:
            raise Exception(f"{type(self)} must be opened first")

    def commit(self) -> None:
        """Commit any pending changes to the store.

        Returns:
            None
        """
        pass

    def close(self) -> None:
        """Close the store by committing changes and marking it as closed.

        Returns:
            None
        """
        self.commit()
        self._opened = False

    def purge(self) -> None:
        """Purge the store by closing it.

        Returns:
            None
        """
        self.close()

    def __getitem__(self, id_: str) -> StoreObjectT:
        """Retrieve an object from the store using its identifier.

        !!! abstract
            This method should be overridden in a subclass.

        Args:
            id_ (str): Identifier of the object to retrieve.

        Returns:
            StoreObject: Object associated with the given identifier.
        """
        raise NotImplementedError

    def __setitem__(self, id_: str, obj: StoreObjectT) -> None:
        """Store an object in the store using its identifier.

        !!! abstract
            This method should be overridden in a subclass.

        Args:
            id_ (str): Identifier for the object to store.
            obj (StoreObject): Object to store.

        Returns:
            None
        """
        raise NotImplementedError

    def __delitem__(self, id_: str) -> None:
        """Delete an object from the store using its identifier.

        !!! abstract
            This method should be overridden in a subclass.

        Args:
            id_ (str): Identifier of the object to delete.

        Returns:
            None
        """
        raise NotImplementedError

    def __iter__(self) -> tp.Iterator[str]:
        """Return an iterator over the identifiers of the objects in the store.

        !!! abstract
            This method should be overridden in a subclass.

        Returns:
            Iterator[str]: Iterator over the identifiers of the objects in the store.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of objects in the store.

        !!! abstract
            This method should be overridden in a subclass.

        Returns:
            int: Number of objects in the store.
        """
        raise NotImplementedError

    def __enter__(self) -> tp.Self:
        if not self.opened:
            self.open()
        self._enter_calls += 1
        return self

    def __exit__(self, *args) -> None:
        if self.enter_calls == 1:
            self.close()
            self._close_on_exit = False
        self._enter_calls -= 1
        if self.enter_calls < 0:
            self._enter_calls = 0


class DictStore(ObjectStore):
    """Store class based on a dictionary that holds objects in memory.

    !!! info
        For default settings, see `chat.obj_store_configs.dict` in `vectorbtpro._settings.knowledge`.

    Args:
        **kwargs: Keyword arguments for `ObjectStore`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "dict"

    _settings_path: tp.SettingsPath = "knowledge.chat.obj_store_configs.dict"

    def __init__(self, **kwargs) -> None:
        ObjectStore.__init__(self, **kwargs)

        self._store = {}

    @property
    def store(self) -> tp.Dict[str, StoreObjectT]:
        """Underlying dictionary storing the objects.

        Returns:
            Dict[str, StoreObject]: Dictionary holding the objects.
        """
        return self._store

    def purge(self) -> None:
        ObjectStore.purge(self)
        self.store.clear()

    def __getitem__(self, id_: str) -> StoreObjectT:
        self.check_opened()
        return self.store[id_]

    def __setitem__(self, id_: str, obj: StoreObjectT) -> None:
        self.check_opened()
        self.store[id_] = obj

    def __delitem__(self, id_: str) -> None:
        self.check_opened()
        del self.store[id_]

    def __iter__(self) -> tp.Iterator[str]:
        self.check_opened()
        return iter(self.store)

    def __len__(self) -> int:
        self.check_opened()
        return len(self.store)


memory_store: tp.Dict[str, tp.Dict[str, StoreObjectT]] = {}
"""Dictionary mapping store identifiers to their corresponding object dictionaries used by `MemoryStore`."""


class MemoryStore(DictStore):
    """Store class for in-memory object storage that commits changes to `memory_store`.

    !!! info
        For default settings, see `chat.obj_store_configs.memory` in `vectorbtpro._settings.knowledge`.

    Args:
        **kwargs: Keyword arguments for `DictStore`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "memory"

    _settings_path: tp.SettingsPath = "knowledge.chat.obj_store_configs.memory"

    def __init__(self, **kwargs) -> None:
        DictStore.__init__(self, **kwargs)

    @property
    def store(self) -> tp.Dict[str, StoreObjectT]:
        return self._store

    def store_exists(self) -> bool:
        """Return whether a store exists for the current store identifier in `memory_store`.

        Returns:
            bool: True if the store exists, otherwise False.
        """
        return self.store_id in memory_store

    def open(self) -> None:
        DictStore.open(self)
        if self.store_exists():
            self._store = dict(memory_store[self.store_id])

    def commit(self) -> None:
        DictStore.commit(self)
        memory_store[self.store_id] = dict(self.store)

    def purge(self) -> None:
        DictStore.purge(self)
        if self.store_exists():
            del memory_store[self.store_id]


class FileStore(DictStore):
    """Store class based on files.

    This class manages file-based storage. It either commits all changes to a single file
    (with the file name corresponding to the index id) or applies an initial commit to a base
    file and subsequent modifications as patch files (with the directory name serving as the index id).

    !!! info
        For default settings, see `chat.obj_store_configs.file` in `vectorbtpro._settings.knowledge`.

    Args:
        dir_path (Optional[PathLike]): Directory path used for file storage.
        compression (CompressionLike): Compression algorithm.

            See `vectorbtpro.utils.pickling.compress`.
        save_kwargs (KwargsLike): Keyword arguments for saving objects.

            See `vectorbtpro.utils.pickling.save`.
        load_kwargs (KwargsLike): Keyword arguments for loading objects.

            See `vectorbtpro.utils.pickling.load`.
        use_patching (Optional[bool]): Whether patch files are used instead of a single file.
        consolidate (Optional[bool]): Whether patch files should be consolidated.
        **kwargs: Keyword arguments for `DictStore`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "file"

    _settings_path: tp.SettingsPath = "knowledge.chat.obj_store_configs.file"

    def __init__(
        self,
        dir_path: tp.Optional[tp.PathLike] = None,
        compression: tp.Union[None, bool, str] = None,
        save_kwargs: tp.KwargsLike = None,
        load_kwargs: tp.KwargsLike = None,
        use_patching: tp.Optional[bool] = None,
        consolidate: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        DictStore.__init__(
            self,
            dir_path=dir_path,
            compression=compression,
            save_kwargs=save_kwargs,
            load_kwargs=load_kwargs,
            use_patching=use_patching,
            consolidate=consolidate,
            **kwargs,
        )

        dir_path = self.resolve_setting(dir_path, "dir_path")
        template_context = self.template_context
        if isinstance(dir_path, CustomTemplate):
            cache_dir = self.get_setting("cache_dir", default=None)
            if cache_dir is not None:
                if isinstance(cache_dir, CustomTemplate):
                    try:
                        if "cache_dir" in cache_dir.get_context_vars():
                            from vectorbtpro._settings import settings

                            _cache_dir = settings["knowledge"]["cache_dir"]
                            if isinstance(_cache_dir, CustomTemplate):
                                _cache_dir = _cache_dir.substitute(template_context, eval_id="cache_dir")
                            template_context = flat_merge_dicts(
                                dict(cache_dir=_cache_dir),
                                template_context,
                            )
                    except NotImplementedError:
                        pass
                    cache_dir = cache_dir.substitute(template_context, eval_id="cache_dir")
                template_context = flat_merge_dicts(dict(cache_dir=cache_dir), template_context)
            release_dir = self.get_setting("release_dir", default=None)
            if release_dir is not None:
                if isinstance(release_dir, CustomTemplate):
                    release_dir = release_dir.substitute(template_context, eval_id="release_dir")
                template_context = flat_merge_dicts(dict(release_dir=release_dir), template_context)
            dir_path = dir_path.substitute(template_context, eval_id="dir_path")
        compression = self.resolve_setting(compression, "compression")
        save_kwargs = self.resolve_setting(save_kwargs, "save_kwargs", merge=True)
        load_kwargs = self.resolve_setting(load_kwargs, "load_kwargs", merge=True)
        use_patching = self.resolve_setting(use_patching, "use_patching")
        consolidate = self.resolve_setting(consolidate, "consolidate")

        self._dir_path = dir_path
        self._compression = compression
        self._save_kwargs = save_kwargs
        self._load_kwargs = load_kwargs
        self._use_patching = use_patching
        self._consolidate = consolidate

        self._store_changes = {}
        self._new_keys = set()

    @property
    def dir_path(self) -> tp.Optional[tp.Path]:
        """Directory path used for file storage.

        Returns:
            Optional[Path]: Directory path, or None if not set.
        """
        return self._dir_path

    @property
    def compression(self) -> tp.CompressionLike:
        """Compression setting used for file operations.

        Returns:
            CompressionLike: Compression configuration used (e.g., None, True, or a specific compression type).
        """
        return self._compression

    @property
    def save_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for saving objects.

        See `vectorbtpro.utils.pickling.save`.

        Returns:
            Kwargs: Dictionary of parameters used when saving objects.
        """
        return self._save_kwargs

    @property
    def load_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for loading objects.

        See `vectorbtpro.utils.pickling.load`.

        Returns:
            Kwargs: Dictionary of parameters used when loading objects.
        """
        return self._load_kwargs

    @property
    def use_patching(self) -> bool:
        """Whether patch files are used instead of a single file.

        Returns:
            bool: True if patch files are used, otherwise False.
        """
        return self._use_patching

    @property
    def consolidate(self) -> bool:
        """Whether patch files should be consolidated.

        Returns:
            bool: True if patch consolidation is enabled, otherwise False.
        """
        return self._consolidate

    @property
    def store_changes(self) -> tp.Dict[str, StoreObjectT]:
        """Dictionary of newly added or modified objects.

        Returns:
            Dict[str, StoreObject]: Mapping of object keys to their associated updated objects.
        """
        return self._store_changes

    @property
    def new_keys(self) -> tp.Set[str]:
        """Keys representing objects not yet added to the main store.

        Returns:
            Set[str]: Set of new object keys.
        """
        return self._new_keys

    def reset_state(self) -> None:
        """Reset the internal state tracking modifications and new keys.

        This method clears any tracked changes and resets consolidation status.

        Returns:
            None
        """
        self._consolidate = False
        self._store_changes = {}
        self._new_keys = set()

    @property
    def store_path(self) -> tp.Path:
        """Filesystem path to the store.

        If patching is used, the path points to the directory containing patch files;
        otherwise, it points to a single file.

        Returns:
            Path: Complete filesystem path for the store.
        """
        dir_path = self.dir_path
        if dir_path is None:
            dir_path = "."
        dir_path = Path(dir_path)
        return dir_path / self.store_id

    @property
    def mirror_store_id(self) -> str:
        return str(self.store_path.resolve())

    def get_next_patch_path(self) -> tp.Path:
        """Return the path for the next patch file to be saved, using an incremented index.

        Returns:
            Path: Path for the next patch file.
        """
        indices = []
        for file in self.store_path.glob("patch_*"):
            indices.append(int(file.stem.split("_")[1]))
        next_index = max(indices) + 1 if indices else 0
        return self.store_path / f"patch_{next_index}"

    def open(self) -> None:
        DictStore.open(self)
        if self.store_path.exists():
            from vectorbtpro.utils.pickling import load

            if self.store_path.is_dir():
                store = {}
                store.update(
                    load(
                        path=self.store_path / "base",
                        compression=self.compression,
                        **self.load_kwargs,
                    )
                )
                patch_paths = sorted(self.store_path.glob("patch_*"), key=lambda f: int(f.stem.split("_")[1]))
                for patch_path in patch_paths:
                    store.update(
                        load(
                            path=patch_path,
                            compression=self.compression,
                            **self.load_kwargs,
                        )
                    )
            else:
                store = load(
                    path=self.store_path,
                    compression=self.compression,
                    **self.load_kwargs,
                )
            self._store = store
        self.reset_state()

    def commit(self) -> tp.Optional[tp.Path]:
        DictStore.commit(self)
        from vectorbtpro.utils.pickling import save

        file_path = None
        if self.use_patching:
            base_path = self.store_path / "base"
            if self.consolidate:
                self.purge()
                file_path = save(
                    self.store,
                    path=base_path,
                    compression=self.compression,
                    **self.save_kwargs,
                )
            elif self.store_changes:
                if self.store_path.exists() and self.store_path.is_file():
                    self.purge()
                if not base_path.exists():
                    file_path = save(
                        self.store_changes,
                        path=base_path,
                        compression=self.compression,
                        **self.save_kwargs,
                    )
                else:
                    file_path = save(
                        self.store_changes,
                        path=self.get_next_patch_path(),
                        compression=self.compression,
                        **self.save_kwargs,
                    )
        else:
            if self.consolidate or self.store_changes:
                if self.store_path.exists() and self.store_path.is_dir():
                    self.purge()
                file_path = save(
                    self.store,
                    path=self.store_path,
                    compression=self.compression,
                    **self.save_kwargs,
                )

        self.reset_state()
        return file_path

    def close(self) -> None:
        DictStore.close(self)
        self.reset_state()

    def purge(self) -> None:
        DictStore.purge(self)
        from vectorbtpro.utils.path_ import remove_file, remove_dir

        if self.store_path.exists():
            if self.store_path.is_dir():
                remove_dir(self.store_path, with_contents=True)
            else:
                remove_file(self.store_path)
        self.reset_state()

    def __setitem__(self, id_: str, obj: StoreObjectT) -> None:
        if obj.id_ not in self:
            self.new_keys.add(obj.id_)
        self.store_changes[obj.id_] = obj
        DictStore.__setitem__(self, id_, obj)

    def __delitem__(self, id_: str) -> None:
        if id_ in self.new_keys:
            del self.store_changes[id_]
            self.new_keys.remove(id_)
        else:
            if id_ in self.store_changes:
                del self.store_changes[id_]
        DictStore.__delitem__(self, id_)


class LMDBStore(ObjectStore):
    """Store class based on LMDB (Lightning Memory-Mapped Database) using the `lmdbm` package.

    !!! info
        For default settings, see `chat.obj_store_configs.lmdb` in `vectorbtpro._settings.knowledge`.

    Args:
        dir_path (Optional[PathLike]): Directory path used for the LMDB store.
        mkdir_kwargs (KwargsLike): Keyword arguments for directory creation.

            See `vectorbtpro.utils.path_.check_mkdir`.
        open_kwargs (KwargsLike): Keyword arguments used when opening the LMDB database via `Lmdb.open`.
        dumps_kwargs (KwargsLike): Keyword arguments used for serializing objects.

            See `vectorbtpro.utils.pickling.dumps`.
        loads_kwargs (KwargsLike): Keyword arguments used for deserializing objects.

            See `vectorbtpro.utils.pickling.loads`.
        **kwargs: Keyword arguments for `ObjectStore`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "lmdb"

    _expected_keys_mode: tp.ExpectedKeysMode = "disable"

    _settings_path: tp.SettingsPath = "knowledge.chat.obj_store_configs.lmdb"

    def __init__(
        self,
        dir_path: tp.Optional[tp.PathLike] = None,
        mkdir_kwargs: tp.KwargsLike = None,
        open_kwargs: tp.KwargsLike = None,
        dumps_kwargs: tp.KwargsLike = None,
        loads_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        ObjectStore.__init__(
            self,
            dir_path=dir_path,
            mkdir_kwargs=mkdir_kwargs,
            open_kwargs=open_kwargs,
            dumps_kwargs=dumps_kwargs,
            loads_kwargs=loads_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("lmdbm")

        lmdb_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_dir_path = lmdb_config.pop("dir_path", None)
        def_mkdir_kwargs = lmdb_config.pop("mkdir_kwargs", None)
        def_open_kwargs = lmdb_config.pop("open_kwargs", None)
        def_dumps_kwargs = lmdb_config.pop("dumps_kwargs", None)
        def_loads_kwargs = lmdb_config.pop("loads_kwargs", None)

        if dir_path is None:
            dir_path = def_dir_path
        template_context = self.template_context
        if isinstance(dir_path, CustomTemplate):
            cache_dir = self.get_setting("cache_dir", default=None)
            if cache_dir is not None:
                if isinstance(cache_dir, CustomTemplate):
                    try:
                        if "cache_dir" in cache_dir.get_context_vars():
                            from vectorbtpro._settings import settings

                            _cache_dir = settings["knowledge"]["cache_dir"]
                            if isinstance(_cache_dir, CustomTemplate):
                                _cache_dir = _cache_dir.substitute(template_context, eval_id="cache_dir")
                            template_context = flat_merge_dicts(
                                dict(cache_dir=_cache_dir),
                                template_context,
                            )
                    except NotImplementedError:
                        pass
                    cache_dir = cache_dir.substitute(template_context, eval_id="cache_dir")
                template_context = flat_merge_dicts(dict(cache_dir=cache_dir), template_context)
            release_dir = self.get_setting("release_dir", default=None)
            if release_dir is not None:
                if isinstance(release_dir, CustomTemplate):
                    release_dir = release_dir.substitute(template_context, eval_id="release_dir")
                template_context = flat_merge_dicts(dict(release_dir=release_dir), template_context)
            dir_path = dir_path.substitute(template_context, eval_id="dir_path")

        mkdir_kwargs = merge_dicts(def_mkdir_kwargs, mkdir_kwargs)
        dumps_kwargs = merge_dicts(def_dumps_kwargs, dumps_kwargs)
        loads_kwargs = merge_dicts(def_loads_kwargs, loads_kwargs)

        init_arg_names = set(get_func_arg_names(ObjectStore.__init__)) | set(get_func_arg_names(type(self).__init__))
        for arg_name in init_arg_names:
            if arg_name in lmdb_config:
                del lmdb_config[arg_name]
        if "mirror" in lmdb_config:
            del lmdb_config["mirror"]
        open_kwargs = merge_dicts(lmdb_config, def_open_kwargs, open_kwargs)

        self._dir_path = dir_path
        self._mkdir_kwargs = mkdir_kwargs
        self._open_kwargs = open_kwargs
        self._dumps_kwargs = dumps_kwargs
        self._loads_kwargs = loads_kwargs

        self._db = None

    @property
    def dir_path(self) -> tp.Optional[tp.Path]:
        """Directory path used for the LMDB store.

        Returns:
            Optional[Path]: Directory path for the LMDB store, or None if not set.
        """
        return self._dir_path

    @property
    def mkdir_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used for directory creation.

        See `vectorbtpro.utils.path_.check_mkdir`.

        Returns:
            Kwargs: Dictionary of parameters for directory creation.
        """
        return self._mkdir_kwargs

    @property
    def open_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used when opening the LMDB database via `Lmdb.open`.

        Returns:
            Kwargs: Dictionary of parameters for opening the LMDB database.
        """
        return self._open_kwargs

    @property
    def dumps_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used for serializing objects.

        See `vectorbtpro.utils.pickling.dumps`.

        Returns:
            Kwargs: Dictionary of parameters for object serialization.
        """
        return self._dumps_kwargs

    @property
    def loads_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used for deserializing objects.

        See `vectorbtpro.utils.pickling.loads`.

        Returns:
            Kwargs: Dictionary of parameters for object deserialization.
        """
        return self._loads_kwargs

    @property
    def db_path(self) -> tp.Path:
        """File system path to the LMDB database.

        Constructs the path by combining the directory (defaulting to "." if not set) with the store identifier.

        Returns:
            Path: Complete file system path pointing to the LMDB database.
        """
        dir_path = self.dir_path
        if dir_path is None:
            dir_path = "."
        dir_path = Path(dir_path)
        return dir_path / self.store_id

    @property
    def mirror_store_id(self) -> str:
        return str(self.db_path.resolve())

    @property
    def db(self) -> tp.Optional[LmdbT]:
        """LMDB database instance.

        Returns:
            Optional[Lmdb]: LMDB database instance if the store is open; otherwise, None.
        """
        return self._db

    def open(self) -> None:
        ObjectStore.open(self)
        from lmdbm import Lmdb
        from vectorbtpro.utils.path_ import check_mkdir

        check_mkdir(self.db_path.parent, **self.mkdir_kwargs)
        self._db = Lmdb.open(str(self.db_path.resolve()), **self.open_kwargs)

    def close(self) -> None:
        ObjectStore.close(self)
        if self.db:
            self.db.close()
        self._db = None

    def purge(self) -> None:
        ObjectStore.purge(self)
        from vectorbtpro.utils.path_ import remove_dir

        remove_dir(self.db_path, missing_ok=True, with_contents=True)

    def encode(self, obj: StoreObjectT) -> bytes:
        """Encode the given object to a bytes representation using the configured serialization settings.

        Args:
            obj (StoreObject): Object to encode.

        Returns:
            bytes: Serialized bytes of the object.
        """
        from vectorbtpro.utils.pickling import dumps

        return dumps(obj, **self.dumps_kwargs)

    def decode(self, bytes_: bytes) -> StoreObjectT:
        """Decode the given bytes into an object using the configured deserialization settings.

        Args:
            bytes_ (bytes): Byte stream containing the serialized object.

        Returns:
            StoreObject: Deserialized object.
        """
        from vectorbtpro.utils.pickling import loads

        return loads(bytes_, **self.loads_kwargs)

    def __getitem__(self, id_: str) -> StoreObjectT:
        self.check_opened()
        return self.decode(self.db[id_])

    def __setitem__(self, id_: str, obj: StoreObjectT) -> None:
        self.check_opened()
        self.db[id_] = self.encode(obj)

    def __delitem__(self, id_: str) -> None:
        self.check_opened()
        del self.db[id_]

    def __iter__(self) -> tp.Iterator[str]:
        self.check_opened()
        return iter(self.db)

    def __len__(self) -> int:
        self.check_opened()
        return len(self.db)


class CachedStore(DictStore):
    """Store class acting as a temporary cache for another store.

    !!! info
        For default settings, see `chat.obj_store_configs.cached` in `vectorbtpro._settings.knowledge`.

    Args:
        obj_store (ObjectStore): Underlying object store to cache.
        lazy_open (Optional[bool]): Flag indicating whether to open the store lazily.
        mirror (Optional[bool]): Flag indicating whether to mirror the store in `memory_store`.
        **kwargs: Keyword arguments for `DictStore`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "cached"

    _settings_path: tp.SettingsPath = "knowledge.chat.obj_store_configs.cached"

    def __init__(
        self,
        obj_store: ObjectStore,
        lazy_open: tp.Optional[bool] = None,
        mirror: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        DictStore.__init__(
            self,
            obj_store=obj_store,
            lazy_open=lazy_open,
            mirror=mirror,
            **kwargs,
        )

        lazy_open = self.resolve_setting(lazy_open, "lazy_open")
        mirror = obj_store.resolve_setting(mirror, "mirror", default=None)
        mirror = self.resolve_setting(mirror, "mirror")
        if mirror and obj_store.mirror_store_id is None:
            mirror = False

        self._obj_store = obj_store
        self._lazy_open = lazy_open
        self._mirror = mirror

        self._force_open = False

    @property
    def obj_store(self) -> ObjectStore:
        """Underlying object store.

        Returns:
            ObjectStore: Object store instance being cached.
        """
        return self._obj_store

    @property
    def lazy_open(self) -> bool:
        """Whether the store opens lazily.

        Returns:
            bool: True if the store opens lazily; otherwise, False.
        """
        return self._lazy_open

    @property
    def mirror(self) -> bool:
        """Whether the store is mirrored in `memory_store`.

        Returns:
            bool: True if the store is mirrored; otherwise, False.
        """
        return self._mirror

    @property
    def force_open(self) -> bool:
        """Whether the store is forced open.

        Returns:
            bool: True if the store is forced open; otherwise, False.
        """
        return self._force_open

    def open(self) -> None:
        DictStore.open(self)
        if self.mirror and self.obj_store.mirror_store_id in memory_store:
            self.store.update(memory_store[self.obj_store.mirror_store_id])
        elif not self.lazy_open or self.force_open:
            self.obj_store.open()

    def check_opened(self) -> None:
        if self.lazy_open and not self.obj_store.opened:
            self._force_open = True
            self.obj_store.open()
        DictStore.check_opened(self)

    def commit(self) -> None:
        DictStore.commit(self)
        self.check_opened()
        self.obj_store.commit()
        if self.mirror:
            memory_store[self.obj_store.mirror_store_id] = dict(self.store)

    def close(self) -> None:
        DictStore.close(self)
        self.obj_store.close()
        self._force_open = False

    def purge(self) -> None:
        DictStore.purge(self)
        self.obj_store.purge()
        if self.mirror and self.obj_store.mirror_store_id in memory_store:
            del memory_store[self.obj_store.mirror_store_id]

    def __getitem__(self, id_: str) -> StoreObjectT:
        if id_ in self.store:
            return self.store[id_]
        self.check_opened()
        obj = self.obj_store[id_]
        self.store[id_] = obj
        return obj

    def __setitem__(self, id_: str, obj: StoreObjectT) -> None:
        self.check_opened()
        self.store[id_] = obj
        self.obj_store[id_] = obj

    def __delitem__(self, id_: str) -> None:
        self.check_opened()
        if id_ in self.store:
            del self.store[id_]
        del self.obj_store[id_]

    def __iter__(self) -> tp.Iterator[str]:
        self.check_opened()
        return iter(self.obj_store)

    def __len__(self) -> int:
        self.check_opened()
        return len(self.obj_store)


def resolve_obj_store(obj_store: tp.ObjectStoreLike = None) -> tp.MaybeType[ObjectStore]:
    """Resolve a subclass or an instance of `ObjectStore`.

    !!! info
        For default settings, see `chat` in `vectorbtpro._settings.knowledge`.

    Args:
        obj_store (ObjectStoreLike): Identifier, subclass, or instance of `ObjectStore`.

            Supported identifiers:

            * "dict" for `DictStore`
            * "memory" for `MemoryStore`
            * "file" for `FileStore`
            * "lmdb" for `LMDBStore`
            * "cached" for `CachedStore`

    Returns:
        ObjectStore: Resolved object store.
    """
    if obj_store is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["chat"]
        obj_store = chat_cfg["obj_store"]
    if isinstance(obj_store, str):
        curr_module = sys.modules[__name__]
        found_obj_store = None
        for name, cls in inspect.getmembers(curr_module, inspect.isclass):
            if name.endswith("Store"):
                _short_name = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == obj_store.lower():
                    found_obj_store = cls
                    break
        if found_obj_store is None:
            raise ValueError(f"Invalid obj_store: {obj_store!r}")
        obj_store = found_obj_store
    if isinstance(obj_store, type):
        checks.assert_subclass_of(obj_store, ObjectStore, arg_name="obj_store")
    else:
        checks.assert_instance_of(obj_store, ObjectStore, arg_name="obj_store")
    return obj_store
