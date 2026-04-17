# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `ArcticDBData` class for fetching data from ArcticDB."""

import pandas as pd

from datetime import datetime
from pathlib import Path
from threading import Lock

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.remote import RemoteData
from vectorbtpro.utils import checks, datetime_ as dt
from vectorbtpro.utils.config import merge_dicts

if tp.TYPE_CHECKING:
    from arcticdb import Arctic as ArcticT
else:
    ArcticT = "arcticdb.Arctic"

__all__ = [
    "ArcticDBData",
]

ARCTIC_CACHE = {}
"""Cache for ArcticDB connections keyed by their URI and parameters."""

ARCTIC_LOCK = Lock()
"""Lock to ensure thread-safe access to the ArcticDB connection cache."""

ArcticDBDataT = tp.TypeVar("ArcticDBDataT", bound="ArcticDBData")


class ArcticDBData(RemoteData):
    """Data class for fetching data from ArcticDB.

    !!! info
        For default settings, see `custom.arctic` in `vectorbtpro._settings.data`.

    See:
        * https://github.com/man-group/ArcticDB for the `arcticdb` library.
        * `ArcticDBData.fetch_key` for argument details.
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.arctic")

    @classmethod
    def connect(
        cls,
        uri: str,
        cache: bool = True,
        refresh_cache: bool = False,
        clear_cache: bool = False,
        **kwargs,
    ) -> ArcticT:
        """Connect to ArcticDB using the provided URI and parameters.

        Uses a cache (`ARCTIC_CACHE`) to reuse existing connections.
        This requires the URI and parameters to be hashable.

        Args:
            uri (str): URI for connecting to ArcticDB (e.g., "lmdb://path/to/db").
            cache (bool): Whether to use the connection cache.

                If False, a new connection is created without caching.
            refresh_cache (bool): Whether to refresh the connection in the cache.
            clear_cache (bool): Whether to clear the entire connection cache before connecting.
            **kwargs: Keyword arguments for `arcticdb.Arctic` constructor.

        Returns:
            Arctic: Connected ArcticDB instance.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("arcticdb")
        from arcticdb import Arctic

        if clear_cache:
            with ARCTIC_LOCK:
                ARCTIC_CACHE.clear()
        if not cache and not refresh_cache:
            return Arctic(uri, **kwargs)
        key = (uri, tuple(sorted(kwargs.items())))
        if not checks.is_hashable(key):
            return Arctic(uri, **kwargs)

        with ARCTIC_LOCK:
            ac = ARCTIC_CACHE.get(key, None)
            if ac is None or refresh_cache:
                ac = Arctic(uri, **kwargs)
                ARCTIC_CACHE[key] = ac
            return ac

    @classmethod
    def resolve_connection(
        cls,
        connection: tp.Union[None, tp.PathLike, ArcticT] = None,
        **connection_config,
    ) -> ArcticT:
        """Resolve and return an ArcticDB connection based on provided parameters.

        Args:
            connection (Union[None, PathLike, Arctic]): ArcticDB connection string, path, or instance.

                If a string, it will be used as an URI, such as S3, Azure, or LMDB URI.
                If a path, it will be used to connect to a local LMDB database.
                If None, in-memory Arctic connection will be created (`mem://`).
            **connection_config: Keyword arguments for `ArcticDBData.connect`.

        Returns:
            Arctic: ArcticDB connection.
        """
        connection = cls.resolve_custom_setting(connection, "connection")
        has_connection_config = len(connection_config) > 0
        connection_config = cls.resolve_custom_setting(connection_config, "connection_config", merge=True)

        if connection is None:
            connection = cls.connect("mem://", **connection_config)
        elif isinstance(connection, str):
            connection = cls.connect(connection, **connection_config)
        elif isinstance(connection, Path):
            connection = cls.connect("lmdb://" + str(connection.resolve()), **connection_config)
        elif has_connection_config:
            raise ValueError("Cannot apply connection_config to already initialized connection")
        return connection

    @classmethod
    def list_libraries(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        connection: tp.Union[None, tp.PathLike, ArcticT] = None,
        connection_config: tp.KwargsLike = None,
    ) -> tp.List[str]:
        """List available libraries from an ArcticDB connection.

        Args:
            pattern (Optional[str]): Pattern to filter library names.
            use_regex (bool): Flag indicating whether the pattern is a regular expression.
            sort (bool): Flag indicating whether to sort the resulting library names.
            connection (Union[None, PathLike, Arctic]): ArcticDB connection string, path, or instance.
            connection_config (KwargsLike): Configuration parameters for creating an ArcticDB connection.

        Returns:
            List[str]: List of library names.
        """
        if connection_config is None:
            connection_config = {}
        connection = cls.resolve_connection(connection, **connection_config)

        libraries = []
        for library in connection.list_libraries():
            if pattern is not None:
                if not cls.key_match(library, pattern, use_regex=use_regex):
                    continue
            libraries.append(library)
        if sort:
            return sorted(dict.fromkeys(libraries))
        return list(dict.fromkeys(libraries))

    @classmethod
    def list_symbols(
        cls,
        symbol_pattern: tp.Optional[str] = None,
        library_pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        library: tp.Optional[str] = None,
        prefix_library: tp.Optional[bool] = None,
        connection: tp.Union[None, tp.PathLike, ArcticT] = None,
        connection_config: tp.KwargsLike = None,
    ) -> tp.List[str]:
        """List symbol names from an ArcticDB connection.

        Uses `vectorbtpro.data.custom.custom.CustomData.key_match` to perform pattern matching on symbol names.

        Args:
            symbol_pattern (Optional[str]): Pattern to filter symbol names.
            library_pattern (Optional[str]): Pattern to filter library names.
            use_regex (bool): Flag indicating whether the pattern is a regular expression.
            sort (bool): Indicates whether to sort the resulting list.
            library (Optional[str]): Filter results to the specified library.
            prefix_library (Optional[bool]): If True, prefixes symbol names with their library names.
            connection (Union[None, PathLike, ArcticT]): ArcticDB connection string, path, or instance.
            connection_config (KwargsLike): Configuration parameters for creating an ArcticDB connection.

        Returns:
            List[str]: List of symbol names, optionally prefixed with library names.
        """
        if connection_config is None:
            connection_config = {}
        connection = cls.resolve_connection(connection, **connection_config)

        if library is None:
            libraries = cls.list_libraries(
                pattern=library_pattern,
                use_regex=use_regex,
                sort=sort,
                connection=connection,
                connection_config=connection_config,
            )
            if prefix_library is None:
                if library_pattern is None and len(libraries) == 1:
                    prefix_library = False
                else:
                    prefix_library = True
        else:
            libraries = [library]
            if prefix_library is None:
                prefix_library = False

        symbols = []
        for library in libraries:
            lib = connection.get_library(library)
            for symbol in lib.list_symbols():
                if symbol_pattern is not None:
                    if not cls.key_match(symbol, symbol_pattern, use_regex=use_regex):
                        continue
                if not prefix_library:
                    symbols.append(symbol)
                else:
                    symbols.append(f"{library}:{symbol}")

        if sort:
            return sorted(dict.fromkeys(symbols))
        return list(dict.fromkeys(symbols))

    @classmethod
    def resolve_keys_meta(
        cls,
        keys: tp.Union[None, dict, tp.MaybeKeys] = None,
        keys_are_features: tp.Optional[bool] = None,
        features: tp.Union[None, dict, tp.MaybeFeatures] = None,
        symbols: tp.Union[None, dict, tp.MaybeSymbols] = None,
        library: tp.Optional[str] = None,
        list_symbols_kwargs: tp.KwargsLike = None,
        connection: tp.Union[None, tp.PathLike, ArcticT] = None,
        connection_config: tp.KwargsLike = None,
    ) -> tp.Kwargs:
        keys_meta = RemoteData.resolve_keys_meta(
            keys=keys,
            keys_are_features=keys_are_features,
            features=features,
            symbols=symbols,
        )
        if keys_meta["keys"] is None:
            if cls.has_key_dict(library):
                raise ValueError("Cannot populate keys if library is defined per key")
            if cls.has_key_dict(list_symbols_kwargs):
                raise ValueError("Cannot populate keys if list_symbols_kwargs is defined per key")
            if cls.has_key_dict(connection):
                raise ValueError("Cannot populate keys if connection is defined per key")
            if cls.has_key_dict(connection_config):
                raise ValueError("Cannot populate keys if connection_config is defined per key")
            if list_symbols_kwargs is None:
                list_symbols_kwargs = {}
            keys_meta["keys"] = cls.list_symbols(
                library=library,
                connection=connection,
                connection_config=connection_config,
                **list_symbols_kwargs,
            )
        return keys_meta

    @classmethod
    def pull(
        cls: tp.Type[ArcticDBDataT],
        keys: tp.Union[tp.MaybeKeys] = None,
        *,
        keys_are_features: tp.Optional[bool] = None,
        features: tp.Union[tp.MaybeFeatures] = None,
        symbols: tp.Union[tp.MaybeSymbols] = None,
        library: tp.Optional[str] = None,
        list_symbols_kwargs: tp.KwargsLike = None,
        connection: tp.Union[None, tp.PathLike, ArcticT] = None,
        connection_config: tp.KwargsLike = None,
        share_connection: tp.Optional[bool] = None,
        **kwargs,
    ) -> ArcticDBDataT:
        """Override `vectorbtpro.data.custom.remote.RemoteData.pull` to resolve and share the ArcticDB
        connection among provided keys.

        Args:
            keys (MaybeKeys): Feature or symbol identifier(s).

                If not provided, available symbol names are used.
            keys_are_features (Optional[bool]): Flag indicating whether the keys represent features.
            features (MaybeFeatures): Feature identifier(s).
            symbols (MaybeSymbols): Symbol identifier(s).
            library (Optional[str]): Library name for symbol lookup.
            connection (Union[None, PathLike, Arctic]): ArcticDB connection string, path, or instance.
            connection_config (KwargsLike): Configuration parameters for creating an ArcticDB connection.
            share_connection (Optional[bool]): If True, uses a shared connection among keys.
            **kwargs: Keyword arguments for `vectorbtpro.data.custom.remote.RemoteData.pull`.

        Returns:
            ArcticDBData: Instance containing the pulled data with resolved keys and connection.
        """
        if share_connection is None:
            if not cls.has_key_dict(connection) and not cls.has_key_dict(connection_config):
                share_connection = True
            else:
                share_connection = False
        if share_connection:
            if connection_config is None:
                connection_config = {}
            connection = cls.resolve_connection(connection, **connection_config)
        keys_meta = cls.resolve_keys_meta(
            keys=keys,
            keys_are_features=keys_are_features,
            features=features,
            symbols=symbols,
            library=library,
            list_symbols_kwargs=list_symbols_kwargs,
            connection=connection,
            connection_config=connection_config,
        )
        keys = keys_meta["keys"]
        keys_are_features = keys_meta["keys_are_features"]
        return super(RemoteData, cls).pull(
            keys,
            keys_are_features=keys_are_features,
            library=library,
            connection=connection,
            connection_config=connection_config,
            **kwargs,
        )

    @classmethod
    def fetch_key(
        cls,
        key: tp.Key,
        library: tp.Optional[str] = None,
        version: tp.Union[None, int, str, datetime] = None,
        connection: tp.Union[None, tp.PathLike, ArcticT] = None,
        connection_config: tp.KwargsLike = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        tz: tp.TimezoneLike = None,
        start_row: tp.Optional[int] = None,
        end_row: tp.Optional[int] = None,
        columns: tp.Optional[tp.MaybeColumns] = None,
        squeeze: tp.Optional[bool] = None,
        **read_kwargs,
    ) -> tp.KeyData:
        """Fetch the ArcticDB data for a feature or symbol.

        Args:
            key (Key): Feature or symbol identifier.

                If the key contains a colon (`:`), it must follow the `LIBRARY:SYMBOL` format,
                and the `library` argument is ignored.
            library (Optional[str]): Library name for symbol lookup.
            version (Optional[Union[None, int, str, datetime]]): Version identifier.
            connection (Union[None, PathLike, Arctic]): ArcticDB connection string, path, or instance.

                See `ArcticDBData.resolve_connection`.
            connection_config (KwargsLike): Configuration parameters for creating an ArcticDB connection.

                See `ArcticDBData.resolve_connection`.
            start (Optional[DatetimeLike]): Start datetime (e.g., "2024-01-01", "1 year ago").

                Extracts the object's index and compares it to this date using the object's timezone.
                See `vectorbtpro.utils.datetime_.to_timestamp`.
            end (Optional[DatetimeLike]): End datetime (e.g., "2025-01-01", "now").

                Extracts the object's index and compares it to this date using the object's timezone.
                See `vectorbtpro.utils.datetime_.to_timestamp`.
            tz (TimezoneLike): Timezone specification (e.g., "UTC", "America/New_York").

                See `vectorbtpro.utils.datetime_.to_timezone`.
            start_row (Optional[int]): Index of the starting row (inclusive).
            end_row (Optional[int]): Index of the ending row (exclusive).
            columns (Optional[MaybeColumns]): Column(s) to return.
            squeeze (Optional[bool]): Flag indicating whether to squeeze the resulting DataFrame
                to a Series if `columns` is a single column.
            **read_kwargs: Keyword arguments for `arcticdb.Library.read`.

        Returns:
            KeyData: Fetched data and a metadata dictionary.
        """
        if connection_config is None:
            connection_config = {}
        connection = cls.resolve_connection(connection, **connection_config)

        if library is None:
            if ":" in key:
                key_parts = key.split(":")
                library, symbol = key_parts
            else:
                libraries = cls.list_libraries(
                    connection=connection,
                    connection_config=connection_config,
                )
                if len(libraries) == 1:
                    library = libraries[0]
                    symbol = key
                elif len(libraries) == 0:
                    raise ValueError("No libraries available in the connection")
                else:
                    symbols = cls.list_symbols(
                        symbol_pattern=key,
                        prefix_library=True,
                        connection=connection,
                        connection_config=connection_config,
                    )
                    if len(symbols) == 1:
                        symbol_parts = symbols[0].split(":")
                        library, symbol = symbol_parts
                    elif len(symbols) == 0:
                        raise ValueError(f"No symbol matching {key!r} available in the connection")
                    else:
                        raise ValueError(f"Multiple symbols matching {key!r} available in the connection: {symbols!r}")
        else:
            symbol = key

        library = cls.resolve_custom_setting(library, "library")
        version = cls.resolve_custom_setting(version, "version")
        start = cls.resolve_custom_setting(start, "start")
        end = cls.resolve_custom_setting(end, "end")
        tz = cls.resolve_custom_setting(tz, "tz")
        start_row = cls.resolve_custom_setting(start_row, "start_row")
        end_row = cls.resolve_custom_setting(end_row, "end_row")
        columns = cls.resolve_custom_setting(columns, "columns")
        squeeze = cls.resolve_custom_setting(squeeze, "squeeze")
        read_kwargs = cls.resolve_custom_setting(read_kwargs, "read_kwargs", merge=True)

        if start is not None:
            start = dt.to_datetime(start, tz=tz)
        if end is not None:
            end = dt.to_datetime(end, tz=tz)
        if start or end:
            date_range = (start, end)
        else:
            date_range = None
        if start_row is not None or end_row is not None:
            row_range = (start_row, end_row)
        else:
            row_range = None
        single_column = False
        if columns is not None:
            if isinstance(columns, str):
                columns = [columns]
                single_column = True
            else:
                columns = list(columns)

        lib = connection.get_library(library)
        item = lib.read(
            symbol,
            as_of=version,
            date_range=date_range,
            row_range=row_range,
            columns=columns,
            **read_kwargs,
        )
        obj = item.data
        version = item.version
        metadata = item.metadata
        if isinstance(obj.index, pd.DatetimeIndex) and tz is None:
            tz = obj.index.tz
        if isinstance(obj, pd.DataFrame) and single_column and squeeze:
            obj = obj.squeeze("columns")
        return obj, dict(tz=tz, version=version, metadata=metadata)

    @classmethod
    def fetch_feature(cls, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        """Fetch the ArcticDB data for a feature.

        Args:
            feature (Feature): Feature identifier.
            **kwargs: Keyword arguments for `ArcticDB.fetch_key`.

        Returns:
            FeatureData: Fetched data and a metadata dictionary.
        """
        return cls.fetch_key(feature, **kwargs)

    @classmethod
    def fetch_symbol(cls, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        """Fetch the ArcticDB data for a symbol.

        Args:
            symbol (Symbol): Symbol identifier.
            **kwargs: Keyword arguments for `ArcticDB.fetch_key`.

        Returns:
            SymbolData: Fetched data and a metadata dictionary.
        """
        return cls.fetch_key(symbol, **kwargs)

    def update_key(self, key: tp.Key, key_is_feature: bool = False, **kwargs) -> tp.KeyData:
        """Update the ArcticDB data for a feature or symbol.

        Args:
            key (Key): Feature or symbol identifier.
            key_is_feature (bool): Flag indicating whether the key represents a feature.
            **kwargs: Keyword arguments for `ArcticDB.fetch_feature` or `ArcticDB.fetch_symbol`.

        Returns:
            KeyData: Updated data and a metadata dictionary.
        """
        fetch_kwargs = self.select_fetch_kwargs(key)
        fetch_kwargs["start"] = self.select_last_index(key)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        if key_is_feature:
            return self.fetch_feature(key, **kwargs)
        return self.fetch_symbol(key, **kwargs)

    def update_feature(self, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        return self.update_key(feature, key_is_feature=True, **kwargs)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        return self.update_key(symbol, key_is_feature=False, **kwargs)
