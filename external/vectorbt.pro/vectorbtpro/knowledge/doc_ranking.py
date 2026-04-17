# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing classes and utilities for ranking documents."""

import re

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.knowledge.completions import resolve_completions
from vectorbtpro.knowledge.doc_storing import (
    StoreDocument,
    MemoryStore,
    StoreData,
    StoreEmbedding,
    ObjectStore,
    CachedStore,
    resolve_obj_store,
)
from vectorbtpro.knowledge.embeddings import Embeddings, resolve_embeddings
from vectorbtpro.knowledge.tokenization import resolve_tokenizer
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.config import merge_dicts, Configured, HasSettings, ExtSettingsPath
from vectorbtpro.utils.decorators import hybrid_method
from vectorbtpro.utils.parsing import get_func_arg_names, get_forward_args
from vectorbtpro.utils.warnings_ import warn

if tp.TYPE_CHECKING:
    from bm25s.tokenization import Tokenizer as BM25TokenizerT
    from bm25s import BM25 as BM25T
else:
    BM25TokenizerT = "bm25s.tokenization.Tokenizer"
    BM25T = "bm25s.BM25"

__all__ = [
    "EmbeddedDocument",
    "ScoredDocument",
    "DocumentRanker",
    "embed_documents",
    "rank_documents",
    "Rankable",
    "Contextable",
    "RankContextable",
]


@define
class EmbeddedDocument(DefineMixin):
    """Define an abstract class for embedded documents."""

    document: StoreDocument = define.field()
    """Primary document content."""

    embedding: tp.Optional[tp.List[float]] = define.field(default=None)
    """List of floats representing the document's embedding."""

    child_documents: tp.List["EmbeddedDocument"] = define.field(factory=list)
    """List of embedded child documents."""


@define
class ScoredDocument(DefineMixin):
    """Define an abstract class for scored documents with an associated numerical score."""

    document: StoreDocument = define.field()
    """Primary document content."""

    score: float = define.field(default=float("nan"))
    """Numeric score assigned to the document."""

    child_documents: tp.List["ScoredDocument"] = define.field(factory=list)
    """List of scored child documents."""


class FallbackError(Exception):
    """Exception raised when a fallback is triggered."""

    pass


class DocumentRanker(Configured):
    """Class for embedding, scoring, and ranking documents.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and
        its sub-configurations `chat` and `chat.doc_ranker_config`.

    Args:
        dataset_id (Optional[str]): Identifier for the dataset.
        embeddings (EmbeddingsLike): Identifier, subclass, or instance of `vectorbtpro.knowledge.embeddings.Embeddings`.

            Resolved using `vectorbtpro.knowledge.embeddings.resolve_embeddings`.
        embeddings_kwargs (KwargsLike): Keyword arguments to initialize or update `embeddings`.
        doc_store (ObjectStoreLike): Identifier, subclass, or instance of
            `vectorbtpro.knowledge.doc_storing.ObjectStore` for documents.

            Resolved using `vectorbtpro.knowledge.doc_storing.resolve_obj_store`.
        doc_store_kwargs (KwargsLike): Keyword arguments to initialize or update `doc_store`.
        cache_doc_store (Optional[bool]): Flag to indicate if `doc_store` should be cached.
        emb_store (ObjectStoreLike): Identifier, subclass, or instance of
            `vectorbtpro.knowledge.doc_storing.ObjectStore` for embeddings.

            Resolved using `vectorbtpro.knowledge.doc_storing.resolve_obj_store`.
        emb_store_kwargs (KwargsLike): Keyword arguments to initialize or update `emb_store`.
        cache_emb_store (Optional[bool]): Flag to indicate if `emb_store` should be cached.
        search_method (Optional[str]): Strategy for document search.

            Supported strategies:

            * "bm25": Use BM25 for document search.
            * "embeddings": Use embeddings for document search.
                Embeds documents that don't have embeddings, which can be time-consuming.
            * "hybrid": Use a combination of embeddings and BM25 for document search.
                Embeds documents that don't have embeddings, which can be time-consuming.
            * "embeddings_fallback": Use "embeddings" if all documents have embeddings, otherwise use "bm25".
            * "hybrid_fallback": Use "hybrid" if all documents have embeddings, otherwise use "bm25".
        bm25_tokenizer (Optional[BM25Tokenizer]): BM25 tokenizer instance or type for processing text.

            Resolved using `DocumentRanker.resolve_bm25_tokenizer`.
        bm25_tokenizer_kwargs (KwargsLike): Keyword arguments to initialize `bm25_tokenizer`.
        bm25_retriever (Optional[MaybeType[BM25]]): BM25 retriever instance or type for document retrieval.

            Resolved using `DocumentRanker.resolve_bm25_retriever`.
        bm25_retriever_kwargs (KwargsLike): Keyword arguments to initialize `bm25_retriever`.
        bm25_mirror_store_id (Optional[str]): Identifier for the BM25 mirror store.
        rrf_k (Optional[int]): K parameter for RRF (Reciprocal Rank Fusion).
        rrf_bm25_weight (Optional[float]): BM25 weight for RRF (Reciprocal Rank Fusion).

            The embedding weight is computed as 1 minus this value.
        score_func (Union[None, str, Callable]): Function or identifier for scoring documents.

            See `DocumentRanker.compute_score`.
        score_agg_func (Union[None, str, Callable]): Function or identifier for aggregating scores.
        normalize_scores (Optional[bool]): Whether scores should be normalized before filtering.
        show_progress (Optional[bool]): Flag indicating whether to display the progress bar.
        pbar_kwargs (KwargsLike): Keyword arguments for configuring the progress bar.

            See `vectorbtpro.utils.pbar.ProgressBar`.
        template_context (KwargsLike): Additional context for template substitution.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.
    """

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat", "knowledge.chat.doc_ranker_config"]

    def __init__(
        self,
        dataset_id: tp.Optional[str] = None,
        embeddings: tp.EmbeddingsLike = None,
        embeddings_kwargs: tp.KwargsLike = None,
        doc_store: tp.ObjectStoreLike = None,
        doc_store_kwargs: tp.KwargsLike = None,
        cache_doc_store: tp.Optional[bool] = None,
        emb_store: tp.ObjectStoreLike = None,
        emb_store_kwargs: tp.KwargsLike = None,
        cache_emb_store: tp.Optional[bool] = None,
        search_method: tp.Optional[str] = None,
        bm25_tokenizer: tp.Optional[tp.MaybeType[BM25TokenizerT]] = None,
        bm25_tokenizer_kwargs: tp.KwargsLike = None,
        bm25_retriever: tp.Optional[tp.MaybeType[BM25T]] = None,
        bm25_retriever_kwargs: tp.KwargsLike = None,
        bm25_mirror_store_id: tp.Optional[str] = None,
        rrf_k: tp.Optional[int] = None,
        rrf_bm25_weight: tp.Optional[float] = None,
        score_func: tp.Union[None, str, tp.Callable] = None,
        score_agg_func: tp.Union[None, str, tp.Callable] = None,
        normalize_scores: tp.Optional[bool] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            dataset_id=dataset_id,
            embeddings=embeddings,
            embeddings_kwargs=embeddings_kwargs,
            doc_store=doc_store,
            doc_store_kwargs=doc_store_kwargs,
            cache_doc_store=cache_doc_store,
            emb_store=emb_store,
            emb_store_kwargs=emb_store_kwargs,
            cache_emb_store=cache_emb_store,
            search_method=search_method,
            bm25_tokenizer=bm25_tokenizer,
            bm25_tokenizer_kwargs=bm25_tokenizer_kwargs,
            bm25_retriever=bm25_retriever,
            bm25_retriever_kwargs=bm25_retriever_kwargs,
            bm25_mirror_store_id=bm25_mirror_store_id,
            rrf_k=rrf_k,
            rrf_bm25_weight=rrf_bm25_weight,
            score_func=score_func,
            score_agg_func=score_agg_func,
            normalize_scores=normalize_scores,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            template_context=template_context,
            **kwargs,
        )

        dataset_id = self.resolve_setting(dataset_id, "dataset_id")
        embeddings = self.resolve_setting(embeddings, "embeddings", default=None)
        embeddings_kwargs = self.resolve_setting(embeddings_kwargs, "embeddings_kwargs", default=None, merge=True)
        doc_store = self.resolve_setting(doc_store, "doc_store", default=None)
        doc_store_kwargs = self.resolve_setting(doc_store_kwargs, "doc_store_kwargs", default=None, merge=True)
        cache_doc_store = self.resolve_setting(cache_doc_store, "cache_doc_store")
        emb_store = self.resolve_setting(emb_store, "emb_store", default=None)
        emb_store_kwargs = self.resolve_setting(emb_store_kwargs, "emb_store_kwargs", default=None, merge=True)
        cache_emb_store = self.resolve_setting(cache_emb_store, "cache_emb_store")
        search_method = self.resolve_setting(search_method, "search_method")
        bm25_mirror_store_id = self.resolve_setting(bm25_mirror_store_id, "bm25_mirror_store_id")
        rrf_k = self.resolve_setting(rrf_k, "rrf_k")
        rrf_bm25_weight = self.resolve_setting(rrf_bm25_weight, "rrf_bm25_weight")
        score_func = self.resolve_setting(score_func, "score_func")
        score_agg_func = self.resolve_setting(score_agg_func, "score_agg_func")
        normalize_scores = self.resolve_setting(normalize_scores, "normalize_scores")
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        obj_store = self.get_setting("obj_store", default=None)
        obj_store_kwargs = self.get_setting("obj_store_kwargs", default=None, merge=True)
        if doc_store is None:
            doc_store = obj_store
        doc_store_kwargs = merge_dicts(obj_store_kwargs, doc_store_kwargs)
        if emb_store is None:
            emb_store = obj_store
        emb_store_kwargs = merge_dicts(obj_store_kwargs, emb_store_kwargs)

        search_method = search_method.lower()
        checks.assert_in(
            search_method,
            ("bm25", "embeddings", "hybrid", "embeddings_fallback", "hybrid_fallback"),
            arg_name="search_method",
        )
        if search_method in ("embeddings", "hybrid", "embeddings_fallback", "hybrid_fallback"):
            try:
                embeddings = resolve_embeddings(embeddings)
                if isinstance(embeddings, type):
                    embeddings_kwargs = dict(embeddings_kwargs)
                    embeddings_kwargs["template_context"] = merge_dicts(
                        template_context, embeddings_kwargs.get("template_context", None)
                    )
                    embeddings = embeddings(**embeddings_kwargs)
                elif embeddings_kwargs:
                    embeddings = embeddings.replace(**embeddings_kwargs)
            except Exception as e:
                if search_method in ("embeddings_fallback", "hybrid_fallback"):
                    warn(f'Failed to resolve embeddings: "{e}"')
                    embeddings = None
                else:
                    raise e
        else:
            embeddings = None

        if isinstance(self._settings_path, list):
            if not isinstance(self._settings_path[-1], str):
                raise TypeError("_settings_path[-1] for DocumentRanker and its subclasses must be a string")
            target_settings_path = self._settings_path[-1]
        elif isinstance(self._settings_path, str):
            target_settings_path = self._settings_path
        else:
            raise TypeError("_settings_path for DocumentRanker and its subclasses must be a list or string")

        doc_store = resolve_obj_store(doc_store)
        if not isinstance(doc_store._settings_path, str):
            raise TypeError("_settings_path for ObjectStore and its subclasses must be a string")
        doc_store_cls = doc_store if isinstance(doc_store, type) else type(doc_store)
        doc_store_settings_path = doc_store._settings_path
        doc_store_settings_path = doc_store_settings_path.replace("knowledge.chat", target_settings_path)
        doc_store_settings_path = doc_store_settings_path.replace("obj_store", "doc_store")
        with ExtSettingsPath([(doc_store_cls, doc_store_settings_path)]):
            if isinstance(doc_store, type):
                doc_store_kwargs = dict(doc_store_kwargs)
                if dataset_id is not None and "store_id" not in doc_store_kwargs:
                    doc_store_kwargs["store_id"] = dataset_id
                doc_store_kwargs["template_context"] = merge_dicts(
                    template_context, doc_store_kwargs.get("template_context", None)
                )
                doc_store = doc_store(**doc_store_kwargs)
            elif doc_store_kwargs:
                doc_store = doc_store.replace(**doc_store_kwargs)
        if cache_doc_store and not isinstance(doc_store, CachedStore):
            doc_store = CachedStore(doc_store)

        emb_store = resolve_obj_store(emb_store)
        if not isinstance(emb_store._settings_path, str):
            raise TypeError("_settings_path for ObjectStore and its subclasses must be a string")
        emb_store_cls = emb_store if isinstance(emb_store, type) else type(emb_store)
        emb_store_settings_path = emb_store._settings_path
        emb_store_settings_path = emb_store_settings_path.replace("knowledge.chat", target_settings_path)
        emb_store_settings_path = emb_store_settings_path.replace("obj_store", "emb_store")
        with ExtSettingsPath([(emb_store_cls, emb_store_settings_path)]):
            if isinstance(emb_store, type):
                emb_store_kwargs = dict(emb_store_kwargs)
                if dataset_id is not None and "store_id" not in emb_store_kwargs:
                    emb_store_kwargs["store_id"] = dataset_id
                elif embeddings is not None and "store_id" not in emb_store_kwargs:
                    emb_store_kwargs["store_id"] = embeddings.digest(incl_settings=True, inherit=False)
                emb_store_kwargs["template_context"] = merge_dicts(
                    template_context, emb_store_kwargs.get("template_context", None)
                )
                emb_store = emb_store(**emb_store_kwargs)
            elif emb_store_kwargs:
                emb_store = emb_store.replace(**emb_store_kwargs)
        if cache_emb_store and not isinstance(emb_store, CachedStore):
            emb_store = CachedStore(emb_store)

        if search_method in ("bm25", "hybrid", "embeddings_fallback", "hybrid_fallback"):
            if bm25_tokenizer_kwargs is None:
                bm25_tokenizer_kwargs = {}
            if bm25_retriever_kwargs is None:
                bm25_retriever_kwargs = {}
            if bm25_mirror_store_id is not None:
                with MemoryStore(store_id=bm25_mirror_store_id) as bm25_memory_store:
                    if bm25_memory_store.store_exists():
                        bm25_tokenizer = bm25_memory_store["bm25_tokenizer"].data
                        bm25_retriever = bm25_memory_store["bm25_retriever"].data
            bm25_tokenizer, bm25_tokenize_kwargs = self.resolve_bm25_tokenizer(
                bm25_tokenizer=bm25_tokenizer, **bm25_tokenizer_kwargs
            )
            bm25_retriever, bm25_retrieve_kwargs = self.resolve_bm25_retriever(
                bm25_retriever=bm25_retriever, **bm25_retriever_kwargs
            )
            if bm25_mirror_store_id is not None:
                with MemoryStore(store_id=bm25_mirror_store_id) as bm25_memory_store:
                    bm25_memory_store["bm25_tokenizer"] = StoreData("bm25_tokenizer", bm25_tokenizer)
                    bm25_memory_store["bm25_retriever"] = StoreData("bm25_retriever", bm25_retriever)
        else:
            bm25_tokenizer = None
            bm25_tokenize_kwargs = {}
            bm25_retriever = None
            bm25_retrieve_kwargs = {}

        if isinstance(score_agg_func, str):
            score_agg_func = getattr(np, score_agg_func)

        self._embeddings = embeddings
        self._doc_store = doc_store
        self._emb_store = emb_store
        self._search_method = search_method
        self._bm25_tokenizer = bm25_tokenizer
        self._bm25_tokenize_kwargs = bm25_tokenize_kwargs
        self._bm25_retriever = bm25_retriever
        self._bm25_retrieve_kwargs = bm25_retrieve_kwargs
        self._rrf_k = rrf_k
        self._rrf_bm25_weight = rrf_bm25_weight
        self._score_func = score_func
        self._score_agg_func = score_agg_func
        self._normalize_scores = normalize_scores
        self._show_progress = show_progress
        self._pbar_kwargs = pbar_kwargs
        self._template_context = template_context

    @property
    def embeddings(self) -> tp.Optional[Embeddings]:
        """Instance of `vectorbtpro.knowledge.embeddings.Embeddings`.

        Returns:
            Embeddings: Embeddings engine or class used for processing document embeddings; None if not set.
        """
        return self._embeddings

    @property
    def doc_store(self) -> ObjectStore:
        """Instance of `vectorbtpro.knowledge.doc_storing.ObjectStore` used for documents.

        Returns:
            ObjectStore: Document store instance used for managing documents.
        """
        return self._doc_store

    @property
    def emb_store(self) -> ObjectStore:
        """Instance of `vectorbtpro.knowledge.doc_storing.ObjectStore` used for embeddings.

        Returns:
            ObjectStore: Embedding store instance used for managing embeddings.
        """
        return self._emb_store

    @property
    def search_method(self) -> str:
        """Strategy for document search.

        Supported strategies:

        * "bm25": Use BM25 for document search.
        * "embeddings": Use embeddings for document search.
            Embeds documents that don't have embeddings, which can be time-consuming.
        * "hybrid": Use a combination of embeddings and BM25 for document search.
            Embeds documents that don't have embeddings, which can be time-consuming.
        * "embeddings_fallback": Use "embeddings" if all documents have embeddings, otherwise use "bm25".
        * "hybrid_fallback": Use "hybrid" if all documents have embeddings, otherwise use "bm25".

        Returns:
            str: Search method used for document retrieval.
        """
        return self._search_method

    @property
    def bm25_tokenizer(self) -> tp.Optional[BM25TokenizerT]:
        """BM25 tokenizer instance from `bm25s.tokenization.Tokenizer`.

        Returns:
            Optional[BM25Tokenizer]: BM25 tokenizer instance used for processing text; None if not set.
        """
        return self._bm25_tokenizer

    @property
    def bm25_tokenize_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for the `tokenize` method of `bm25s.tokenization.Tokenizer`.

        Returns:
            Kwargs: Dictionary of parameters for the tokenization process.
        """
        return self._bm25_tokenize_kwargs

    @property
    def bm25_retriever(self) -> tp.Optional[BM25T]:
        """BM25 retriever instance from `bm25s.BM25`.

        Returns:
            Optional[BM25]: BM25 retriever instance used for document retrieval; None if not set.
        """
        return self._bm25_retriever

    @property
    def bm25_retrieve_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for the `retrieve` method of `bm25s.BM25`.

        Returns:
            Kwargs: Dictionary of parameters for the retrieval process.
        """
        return self._bm25_retrieve_kwargs

    @property
    def rrf_k(self) -> int:
        """K parameter for RRF (Reciprocal Rank Fusion).

        Returns:
            int: K parameter used in RRF.
        """
        return self._rrf_k

    @property
    def rrf_bm25_weight(self) -> float:
        """BM25 weight for RRF (Reciprocal Rank Fusion).

        The embedding weight is computed as 1 minus this value.

        Returns:
            float: BM25 weight used in RRF.
        """
        return self._rrf_bm25_weight

    @property
    def score_func(self) -> tp.Union[str, tp.Callable]:
        """Score function or its name used for computing document scores.

        See `DocumentRanker.compute_score`.

        Returns:
            Union[str, Callable]: Score function used for computing document scores.
        """
        return self._score_func

    @property
    def score_agg_func(self) -> tp.Callable:
        """Function used to aggregate scores.

        Returns:
            Callable: Function used for aggregating scores.
        """
        return self._score_agg_func

    @property
    def normalize_scores(self) -> bool:
        """Whether scores should be normalized before filtering.

        Returns:
            bool: True if scores should be normalized; otherwise, False.
        """
        return self._normalize_scores

    @property
    def show_progress(self) -> tp.Optional[bool]:
        """Whether to display a progress bar.

        Returns:
            Optional[bool]: True if a progress bar should be shown; otherwise, False.
        """
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for configuring the progress bar.

        See `vectorbtpro.utils.pbar.ProgressBar`.

        Returns:
            Kwargs: Dictionary of parameters for the progress bar.
        """
        return self._pbar_kwargs

    @property
    def template_context(self) -> tp.Kwargs:
        """Additional context for template substitution.

        Returns:
            Kwargs: Dictionary of context variables for template substitution.
        """
        return self._template_context

    def resolve_bm25_tokenizer(
        cls,
        bm25_tokenizer: tp.Optional[tp.MaybeType[BM25TokenizerT]] = None,
        **kwargs,
    ) -> tp.Tuple[BM25TokenizerT, tp.Kwargs]:
        """Return a tuple containing a resolved instance of `bm25s.tokenization.Tokenizer` and
        tokenization keyword arguments.

        Args:
            bm25_tokenizer (Optional[BM25Tokenizer]): BM25 tokenizer instance or type.
            **kwargs: Keyword arguments for initializing `bm25_tokenizer` and tokenization.

        Returns:
            Tuple[BM25TokenizerT, Kwargs]: Resolved BM25 tokenizer and the tokenization keyword arguments.
        """
        from vectorbtpro.utils.module_ import assert_can_import, check_installed

        assert_can_import("bm25s")

        from bm25s.tokenization import Tokenizer

        bm25_tokenizer = cls.resolve_setting(bm25_tokenizer, "bm25_tokenizer")
        kwargs = cls.resolve_setting(kwargs, "bm25_tokenizer_kwargs", merge=True)

        if bm25_tokenizer is None:
            bm25_tokenizer = Tokenizer
        if isinstance(bm25_tokenizer, type):
            checks.assert_subclass_of(bm25_tokenizer, Tokenizer, arg_name="bm25_tokenizer")
            bm25_tokenizer_type = bm25_tokenizer
        else:
            checks.assert_instance_of(bm25_tokenizer, Tokenizer, arg_name="bm25_tokenizer")
            bm25_tokenizer_type = type(bm25_tokenizer)
        bm25_tokenize_kwargs = {}
        if kwargs:
            bm25_tokenize_arg_names = get_func_arg_names(bm25_tokenizer_type.tokenize)
            for k in bm25_tokenize_arg_names:
                if k in kwargs:
                    bm25_tokenize_kwargs[k] = kwargs.pop(k)
        if isinstance(bm25_tokenizer, type):
            if "splitter" not in kwargs:
                kwargs["splitter"] = cls.bm25_splitter
                if "lower" not in kwargs:
                    kwargs["lower"] = False
            if "stemmer" not in kwargs and check_installed("Stemmer"):
                import Stemmer

                kwargs["stemmer"] = Stemmer.Stemmer("english")
            bm25_tokenizer = bm25_tokenizer(**kwargs)
        return bm25_tokenizer, bm25_tokenize_kwargs

    def resolve_bm25_retriever(
        cls,
        bm25_retriever: tp.Optional[tp.MaybeType[BM25T]] = None,
        **kwargs,
    ) -> tp.Tuple[BM25T, tp.Kwargs]:
        """Return a tuple containing a resolved instance of `bm25s.BM25` and retrieval keyword arguments.

        Args:
            bm25_retriever (Optional[BM25]): BM25 retriever instance or type.
            **kwargs: Keyword arguments for initializing `bm25_retriever` and retrieval.

        Returns:
            Tuple[BM25T, Kwargs]: Resolved BM25 retriever and the retrieval keyword arguments.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("bm25s")

        from bm25s import BM25

        bm25_retriever = cls.resolve_setting(bm25_retriever, "bm25_retriever")
        kwargs = cls.resolve_setting(kwargs, "bm25_retriever_kwargs", merge=True)

        if bm25_retriever is None:
            bm25_retriever = BM25
        if isinstance(bm25_retriever, type):
            checks.assert_subclass_of(bm25_retriever, BM25, arg_name="bm25_retriever")
            bm25_retriever_type = bm25_retriever
        else:
            checks.assert_instance_of(bm25_retriever, BM25, arg_name="bm25_retriever")
            bm25_retriever_type = type(bm25_retriever)
        bm25_retrieve_kwargs = {}
        if kwargs:
            bm25_retrieve_arg_names = get_func_arg_names(bm25_retriever_type.retrieve)
            for k in bm25_retrieve_arg_names:
                if k in kwargs:
                    bm25_retrieve_kwargs[k] = kwargs.pop(k)
        if isinstance(bm25_retriever, type):
            bm25_retriever = bm25_retriever(**kwargs)
        return bm25_retriever, bm25_retrieve_kwargs

    def embed_documents(
        self,
        documents: tp.Iterable[StoreDocument],
        refresh: bool = False,
        refresh_documents: tp.Optional[bool] = None,
        refresh_embeddings: tp.Optional[bool] = None,
        return_embeddings: bool = False,
        return_documents: bool = False,
        with_fallback: bool = False,
    ) -> tp.Optional[tp.EmbeddedDocuments]:
        """Embed documents by optionally refreshing stored documents and embeddings.

        Without refreshing, persisted objects from the respective stores are used.

        Args:
            documents (Iterable[StoreDocument]): Collection of documents to embed.
            refresh (bool): Flag to refresh both documents and embeddings.
            refresh_documents (Optional[bool]): Flag to refresh documents; defaults to `refresh`.
            refresh_embeddings (Optional[bool]): Flag to refresh embeddings; defaults to `refresh`.
            return_embeddings (bool): Flag indicating whether to return embeddings.
            return_documents (bool): If True, include original document objects in the output.
            with_fallback (bool): If True, raise `FallbackError` if new embeddings are needed.

        Returns:
            Optional[EmbeddedDocuments]: Embedded documents or embeddings based on the specified return flags.

                Returns None if both return flags are False.
        """
        if refresh_documents is None:
            refresh_documents = refresh
        if refresh_embeddings is None:
            refresh_embeddings = refresh
        with self.doc_store, self.emb_store:
            documents = list(documents)
            documents_to_split = []
            document_splits = {}
            for document in documents:
                refresh_document = (
                    refresh_documents
                    or refresh_embeddings
                    or document.id_ not in self.doc_store
                    or document.id_ not in self.emb_store
                )
                if not refresh_document:
                    obj = self.emb_store[document.id_]
                    if obj.child_ids:
                        for child_id in obj.child_ids:
                            if child_id not in self.doc_store or child_id not in self.emb_store:
                                refresh_document = True
                                break
                if refresh_document:
                    if with_fallback:
                        raise FallbackError("Some documents need to be refreshed")
                    documents_to_split.append(document)
            if documents_to_split:
                from vectorbtpro.utils.pbar import ProgressBar

                pbar_kwargs = merge_dicts(dict(prefix="split_documents"), self.pbar_kwargs)
                with ProgressBar(
                    total=len(documents_to_split),
                    show_progress=self.show_progress,
                    **pbar_kwargs,
                ) as pbar:
                    for document in documents_to_split:
                        document_splits[document.id_] = document.split()
                        pbar.update()

            obj_contents = {}
            for document in documents:
                if refresh_documents or document.id_ not in self.doc_store:
                    self.doc_store[document.id_] = document
                if document.id_ in document_splits:
                    document_chunks = document_splits[document.id_]
                    obj = StoreEmbedding(document.id_)
                    for document_chunk in document_chunks:
                        if document_chunk.id_ != document.id_:
                            if refresh_documents or document_chunk.id_ not in self.doc_store:
                                self.doc_store[document_chunk.id_] = document_chunk
                            if refresh_embeddings or document_chunk.id_ not in self.emb_store:
                                child_obj = StoreEmbedding(document_chunk.id_, parent_id=document.id_)
                                self.emb_store[child_obj.id_] = child_obj
                            else:
                                child_obj = self.emb_store[document_chunk.id_]
                            obj.child_ids.append(child_obj.id_)
                    if refresh_documents or refresh_embeddings or document.id_ not in self.emb_store:
                        self.emb_store[obj.id_] = obj
                else:
                    obj = self.emb_store[document.id_]
                if not obj.embedding:
                    if obj.child_ids:
                        for child_id in obj.child_ids:
                            child_obj = self.emb_store[child_id]
                            if not child_obj.embedding:
                                child_document = self.doc_store[child_id]
                                content = child_document.get_content(for_embed=True)
                                if content:
                                    obj_contents[child_id] = content
                                    if with_fallback:
                                        raise FallbackError("Some documents need to be embedded")
                    else:
                        content = document.get_content(for_embed=True)
                        if content:
                            obj_contents[obj.id_] = content
                            if with_fallback:
                                raise FallbackError("Some documents need to be embedded")

            if obj_contents:
                if self.embeddings is None:
                    if with_fallback:
                        raise FallbackError("Embeddings engine is not set")
                    raise ValueError("Embeddings engine is not set")
                total = 0
                for batch in self.embeddings.iter_embedding_batches(list(obj_contents.values())):
                    batch_keys = list(obj_contents.keys())[total : total + len(batch)]
                    obj_embeddings = dict(zip(batch_keys, batch))
                    for obj_id, embedding in obj_embeddings.items():
                        obj = self.emb_store[obj_id]
                        new_obj = obj.replace(embedding=embedding)
                        self.emb_store[new_obj.id_] = new_obj
                    total += len(batch)

            if return_embeddings or return_documents:
                embeddings = []
                for document in documents:
                    obj = self.emb_store[document.id_]
                    if obj.embedding:
                        if return_documents:
                            embeddings.append(EmbeddedDocument(document, embedding=obj.embedding))
                        else:
                            embeddings.append(obj.embedding)
                    elif obj.child_ids:
                        child_embeddings = []
                        for child_id in obj.child_ids:
                            child_obj = self.emb_store[child_id]
                            if child_obj.embedding:
                                if return_documents:
                                    child_document = self.doc_store[child_id]
                                    child_embeddings.append(
                                        EmbeddedDocument(child_document, embedding=child_obj.embedding)
                                    )
                                else:
                                    child_embeddings.append(child_obj.embedding)
                            else:
                                if return_documents:
                                    child_document = self.doc_store[child_id]
                                    child_embeddings.append(EmbeddedDocument(child_document))
                                else:
                                    child_embeddings.append(None)
                        if return_documents:
                            embeddings.append(EmbeddedDocument(document, child_documents=child_embeddings))
                        else:
                            embeddings.append(child_embeddings)
                    else:
                        if return_documents:
                            embeddings.append(EmbeddedDocument(document))
                        else:
                            embeddings.append(None)

                return embeddings

    def compute_score(
        self,
        emb1: tp.Union[tp.MaybeIterable[tp.List[float]], np.ndarray],
        emb2: tp.Union[tp.MaybeIterable[tp.List[float]], np.ndarray],
    ) -> tp.Union[float, np.ndarray]:
        """Compute similarity or distance scores between embeddings.

        Compute scores between embedding vectors using the configured scoring function.
        Supported functions include "cosine", "euclidean", and "dot". Alternatively, a callable
        metric can be supplied that accepts two arrays and returns a 2-dimensional ndarray.

        Args:
            emb1 (Union[MaybeIterable[List[float]], ndarray]): First embedding or collection of embeddings.
            emb2 (Union[MaybeIterable[List[float]], ndarray]): Second embedding or collection of embeddings.

        Returns:
            Union[float, ndarray]: Computed score or score matrix between the embeddings.
        """
        emb1 = np.asarray(emb1)
        emb2 = np.asarray(emb2)
        emb1_single = emb1.ndim == 1
        emb2_single = emb2.ndim == 1
        if emb1_single:
            emb1 = emb1.reshape(1, -1)
        if emb2_single:
            emb2 = emb2.reshape(1, -1)

        if isinstance(self.score_func, str):
            if self.score_func.lower() == "cosine":
                emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
                emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
                emb1_norm = np.nan_to_num(emb1_norm)
                emb2_norm = np.nan_to_num(emb2_norm)
                score_matrix = np.dot(emb1_norm, emb2_norm.T)
            elif self.score_func.lower() == "euclidean":
                diff = emb1[:, np.newaxis, :] - emb2[np.newaxis, :, :]
                distances = np.linalg.norm(diff, axis=2)
                score_matrix = np.divide(1, distances, where=distances != 0, out=np.full_like(distances, np.inf))
            elif self.score_func.lower() == "dot":
                score_matrix = np.dot(emb1, emb2.T)
            else:
                raise ValueError(f"Invalid score_func: {self.score_func!r}")
        else:
            score_matrix = self.score_func(emb1, emb2)

        if emb1_single and emb2_single:
            return float(score_matrix[0, 0])
        if emb1_single or emb2_single:
            return score_matrix.flatten()
        return score_matrix

    def score_documents(
        self,
        query: str,
        documents: tp.Optional[tp.Iterable[StoreDocument]] = None,
        refresh: bool = False,
        refresh_documents: tp.Optional[bool] = None,
        refresh_embeddings: tp.Optional[bool] = None,
        return_chunks: bool = False,
        return_documents: bool = False,
        with_fallback: bool = False,
    ) -> tp.ScoredDocuments:
        """Score documents by relevance to a query.

        Optionally refresh and embed documents before scoring their relevance to a query.
        If no documents are provided, the document store is used. When `return_chunks` is True,
        document chunks are scored instead of parent documents. The query is embedded and compared
        against document embeddings to compute relevance scores.

        Args:
            query (str): Query string for scoring relevance.
            documents (Optional[Iterable[StoreDocument]]): Collection of documents to score.

                If None, documents from the document store are used.
            refresh (bool): Flag to refresh both documents and embeddings.
            refresh_documents (Optional[bool]): Flag to refresh documents; defaults to `refresh`.
            refresh_embeddings (Optional[bool]): Flag to refresh embeddings; defaults to `refresh`.
            return_chunks (bool): Whether to return document chunks.
            return_documents (bool): If True, include original document objects in the output.
            with_fallback (bool): If True, raise `FallbackError` if new embeddings are needed.

        Returns:
            ScoredDocuments: Collection of documents with their computed relevance scores.
        """
        with self.doc_store, self.emb_store:
            if documents is None:
                if self.doc_store is None:
                    raise ValueError("Must provide at least documents or doc_store")
                documents = self.doc_store.values()
                documents_provided = False
            else:
                documents_provided = True
            documents = list(documents)
            if not documents:
                return []
            self.embed_documents(
                documents,
                refresh=refresh,
                refresh_documents=refresh_documents,
                refresh_embeddings=refresh_embeddings,
                with_fallback=with_fallback,
            )
            if return_chunks:
                document_chunks = []
                for document in documents:
                    obj = self.emb_store[document.id_]
                    if obj.child_ids:
                        for child_id in obj.child_ids:
                            document_chunk = self.doc_store[child_id]
                            document_chunks.append(document_chunk)
                    elif not obj.parent_id or obj.parent_id not in self.doc_store:
                        document_chunk = self.doc_store[obj.id_]
                        document_chunks.append(document_chunk)
                documents = document_chunks
            elif not documents_provided:
                document_parents = []
                for document in documents:
                    obj = self.emb_store[document.id_]
                    if not obj.parent_id or obj.parent_id not in self.doc_store:
                        document_parent = self.doc_store[obj.id_]
                        document_parents.append(document_parent)
                documents = document_parents

            obj_embeddings = {}
            for document in documents:
                obj = self.emb_store[document.id_]
                if obj.embedding:
                    obj_embeddings[obj.id_] = obj.embedding
                elif obj.child_ids:
                    for child_id in obj.child_ids:
                        child_obj = self.emb_store[child_id]
                        if child_obj.embedding:
                            obj_embeddings[child_id] = child_obj.embedding
            if obj_embeddings:
                if self.embeddings is None:
                    if with_fallback:
                        raise FallbackError("Embeddings engine is not set")
                    raise ValueError("Embeddings engine is not set")
                query_embedding = self.embeddings.get_embedding(query)
                scores = self.compute_score(query_embedding, list(obj_embeddings.values()))
                obj_scores = dict(zip(obj_embeddings.keys(), scores))
            else:
                obj_scores = {}

            scores = []
            for document in documents:
                obj = self.emb_store[document.id_]
                child_scores = []
                if obj.child_ids:
                    for child_id in obj.child_ids:
                        if child_id in obj_scores:
                            child_score = obj_scores[child_id]
                            if return_documents:
                                child_document = self.doc_store[child_id]
                                child_scores.append(ScoredDocument(child_document, score=child_score))
                            else:
                                child_scores.append(child_score)
                    if child_scores:
                        if return_documents:
                            doc_score = self.score_agg_func([document.score for document in child_scores])
                        else:
                            doc_score = self.score_agg_func(child_scores)
                    else:
                        doc_score = float("nan")
                else:
                    if obj.id_ in obj_scores:
                        doc_score = obj_scores[obj.id_]
                    else:
                        doc_score = float("nan")
                if return_documents:
                    scores.append(ScoredDocument(document, score=doc_score, child_documents=child_scores))
                else:
                    scores.append(doc_score)
            return scores

    SPLIT_PATTERN = re.compile(r"(?<=[a-z])(?=[A-Z])|_")
    """Regular expression pattern used by `DocumentRanker.bm25_splitter` to split text at 
    transitions between lowercase and uppercase letters or underscores."""

    TOKEN_PATTERN = re.compile(r"(?u)\b\w{2,}\b")
    """Regular expression pattern used by `DocumentRanker.bm25_splitter` to extract tokens with 
    at least two characters."""

    @classmethod
    def bm25_splitter(cls, text: str) -> tp.List[str]:
        """Return a list of lowercase tokens extracted from the input text using BM25 tokenization.

        Args:
            text (str): Text to tokenize.

        Returns:
            list[str]: Lowercase tokens extracted from the input text.
        """
        spaced_text = cls.SPLIT_PATTERN.sub(" ", text)
        tokens = cls.TOKEN_PATTERN.findall(spaced_text)
        return [token.lower() for token in tokens]

    def bm25_score_documents(
        self,
        query: str,
        documents: tp.Optional[tp.Iterable[StoreDocument]] = None,
        refresh: bool = False,
        refresh_documents: tp.Optional[bool] = None,
        return_chunks: bool = False,
        return_documents: bool = False,
    ) -> tp.ScoredDocuments:
        """Return BM25 relevance scores for documents matching a query.

        Args:
            query (str): Query string for relevance scoring.
            documents (Optional[Iterable[StoreDocument]]): Collection of documents to score.

                If None, documents from the document store are used.
            refresh (bool): Flag to refresh both documents and embeddings.
            refresh_documents (Optional[bool]): Flag to refresh documents; defaults to `refresh`.
            return_chunks (bool): Whether to return document chunks.
            return_documents (bool): If True, include original document objects in the output.

        Returns:
            ScoredDocuments: Computed BM25 scores for each document, as either numeric scores or
                `ScoredDocument` objects.
        """
        with self.doc_store, self.emb_store:
            if refresh_documents is None:
                refresh_documents = refresh
            if documents is None:
                if self.doc_store is None:
                    raise ValueError("Must provide at least documents or doc_store")
                documents = self.doc_store.values()
            documents = list(documents)

            if return_chunks:
                documents_to_split = []
                document_splits = {}
                for document in documents:
                    refresh_document = (
                        refresh_documents or document.id_ not in self.doc_store or document.id_ not in self.emb_store
                    )
                    if not refresh_document:
                        obj = self.emb_store[document.id_]
                        if obj.child_ids:
                            for child_id in obj.child_ids:
                                if child_id not in self.doc_store or child_id not in self.emb_store:
                                    refresh_document = True
                                    break
                    if refresh_document:
                        documents_to_split.append(document)
                if documents_to_split:
                    from vectorbtpro.utils.pbar import ProgressBar

                    pbar_kwargs = merge_dicts(dict(prefix="split_documents"), self.pbar_kwargs)
                    with ProgressBar(
                        total=len(documents_to_split),
                        show_progress=self.show_progress,
                        **pbar_kwargs,
                    ) as pbar:
                        for document in documents_to_split:
                            document_splits[document.id_] = document.split()
                            pbar.update()

                for document in documents:
                    if refresh_documents or document.id_ not in self.doc_store:
                        self.doc_store[document.id_] = document
                    if document.id_ in document_splits:
                        document_chunks = document_splits[document.id_]
                        obj = StoreEmbedding(document.id_)
                        for document_chunk in document_chunks:
                            if document_chunk.id_ != document.id_:
                                if refresh_documents or document_chunk.id_ not in self.doc_store:
                                    self.doc_store[document_chunk.id_] = document_chunk
                                if document_chunk.id_ not in self.emb_store:
                                    child_obj = StoreEmbedding(document_chunk.id_, parent_id=document.id_)
                                    self.emb_store[child_obj.id_] = child_obj
                                else:
                                    child_obj = self.emb_store[document_chunk.id_]
                                obj.child_ids.append(child_obj.id_)
                        if refresh_documents or document.id_ not in self.emb_store:
                            self.emb_store[obj.id_] = obj

                document_chunks = []
                for document in documents:
                    obj = self.emb_store[document.id_]
                    if obj.child_ids:
                        for child_id in obj.child_ids:
                            document_chunk = self.doc_store[child_id]
                            document_chunks.append(document_chunk)
                    elif not obj.parent_id or obj.parent_id not in self.doc_store:
                        document_chunk = self.doc_store[obj.id_]
                        document_chunks.append(document_chunk)
                documents = document_chunks

            bm25_tokenizer = self.bm25_tokenizer
            bm25_retriever = self.bm25_retriever
            bm25_tokenize_kwargs = dict(self.bm25_tokenize_kwargs)
            bm25_retrieve_kwargs = dict(self.bm25_retrieve_kwargs)
            if (
                refresh_documents
                or not bm25_tokenizer.get_vocab_dict()
                or not hasattr(bm25_retriever, "scores")
                or not bm25_retriever.scores
                or bm25_retriever.scores["num_docs"] != len(documents)
            ):
                texts = []
                for document in documents:
                    content = document.get_content(for_embed=True)
                    if not content:
                        content = ""
                    texts.append(content)
                tokenized_documents = bm25_tokenizer.tokenize(
                    texts,
                    return_as="string",
                    **bm25_tokenize_kwargs,
                )
                bm25_retriever.index(tokenized_documents, show_progress=False)
            if "update_vocab" in bm25_tokenize_kwargs:
                del bm25_tokenize_kwargs["update_vocab"]
            if "show_progress" in bm25_tokenize_kwargs:
                del bm25_tokenize_kwargs["show_progress"]
            tokenized_queries = bm25_tokenizer.tokenize(
                [query],
                return_as="string",
                update_vocab=False,
                show_progress=False,
                **bm25_tokenize_kwargs,
            )
            indices, scores = bm25_retriever.retrieve(
                tokenized_queries,
                k=len(documents),
                sorted=False,
                **bm25_retrieve_kwargs,
            )
            obj_scores = {}
            for i in range(scores.shape[1]):
                obj_scores[documents[indices[0, i]].id_] = scores[0, i]

            scores = []
            for document in documents:
                if return_chunks:
                    obj = self.emb_store[document.id_]
                    child_scores = []
                    if obj.child_ids:
                        for child_id in obj.child_ids:
                            if child_id in obj_scores:
                                child_score = obj_scores[child_id]
                                if return_documents:
                                    child_document = self.doc_store[child_id]
                                    child_scores.append(ScoredDocument(child_document, score=child_score))
                                else:
                                    child_scores.append(child_score)
                        if child_scores:
                            if return_documents:
                                doc_score = self.score_agg_func([document.score for document in child_scores])
                            else:
                                doc_score = self.score_agg_func(child_scores)
                        else:
                            doc_score = float("nan")
                    else:
                        if obj.id_ in obj_scores:
                            doc_score = obj_scores[obj.id_]
                        else:
                            doc_score = float("nan")
                else:
                    if document.id_ in obj_scores:
                        doc_score = obj_scores[document.id_]
                    else:
                        doc_score = float("nan")
                    child_scores = []
                if return_documents:
                    scores.append(ScoredDocument(document, score=doc_score, child_documents=child_scores))
                else:
                    scores.append(doc_score)
            return scores

    @classmethod
    def resolve_top_k(cls, scores: tp.Iterable[float], top_k: tp.TopKLike = None) -> tp.Optional[int]:
        """Resolve the `top_k` value from sorted scores.

        Args:
            scores (Iterable[float]): Sorted document scores.
            top_k (TopKLike): Parameter specifying the `top_k` selection method, which can be an integer,
                a float percentage, a string ('elbow' or 'kmeans'), or a callable.

        Returns:
            Optional[int]: Resolved `top_k` value, or None if `top_k` is not provided.
        """
        if top_k is None:
            return None
        scores = np.asarray(scores)
        scores = scores[~np.isnan(scores)]

        if isinstance(top_k, str):
            if top_k.lower() == "elbow":
                if scores.size == 0:
                    return 0
                diffs = np.diff(scores)
                top_k = np.argmax(-diffs) + 1
            elif top_k.lower() == "kmeans":
                from sklearn.cluster import KMeans

                kmeans = KMeans(n_clusters=2, random_state=0).fit(scores.reshape(-1, 1))
                high_score_cluster = np.argmax(kmeans.cluster_centers_)
                top_k_indices = np.where(kmeans.labels_ == high_score_cluster)[0]
                top_k = max(top_k_indices) + 1
            else:
                raise ValueError(f"Invalid top_k: {top_k!r}")
        elif callable(top_k):
            top_k = top_k(scores)
        if checks.is_float(top_k):
            top_k = int(top_k * len(scores))
        return top_k

    @classmethod
    def top_k_from_cutoff(cls, scores: tp.Iterable[float], cutoff: tp.Optional[float] = None) -> tp.Optional[int]:
        """Determine the number of top documents based on a cutoff threshold from sorted scores.

        Args:
            scores (Iterable[float]): Sorted document scores.
            cutoff (Optional[float]): Score threshold to filter documents.

        Returns:
            Optional[int]: Count of scores greater than or equal to the cutoff, or None if cutoff is None.
        """
        if cutoff is None:
            return None
        scores = np.asarray(scores)
        scores = scores[~np.isnan(scores)]
        return len(scores[scores >= cutoff])

    @classmethod
    def extract_doc_scores(cls, scored_documents: tp.List[ScoredDocument]) -> tp.List[float]:
        """Recursively extract scores from a list of scored documents.

        Args:
            scored_documents (List[ScoredDocument]): Documents with existing scores.

        Returns:
            List[float]: Scores extracted from each document and its child documents.
        """
        scores = []
        for document in scored_documents:
            scores.append(document.score)
            if document.child_documents:
                scores.extend(cls.extract_doc_scores(document.child_documents))
        return scores

    @classmethod
    def normalize_doc_scores(cls, scores: tp.Iterable[float]) -> np.ndarray:
        """Normalize a collection of scores using min-max scaling.

        Args:
            scores (Iterable[float]): Iterable of scores to normalize.

        Returns:
            ndarray: Array of normalized scores.
        """
        scores = np.array(scores, dtype=float)
        min_score, max_score = np.nanmin(scores), np.nanmax(scores)
        return (scores - min_score) / (max_score - min_score) if max_score != min_score else scores - min_score

    @classmethod
    def replace_doc_scores(
        cls,
        scored_documents: tp.List[ScoredDocument],
        new_scores: tp.List[float],
    ) -> tp.List[ScoredDocument]:
        """Recursively replace scores in documents with new scores.

        Args:
            scored_documents (List[ScoredDocument]): Documents with existing scores.
            new_scores (List[float]): New scores to assign, consumed in order.

        Returns:
            List[ScoredDocument]: Updated documents with replaced scores.
        """
        new_scored_documents = []
        for i in range(len(scored_documents)):
            doc = scored_documents[i]
            document = doc.document
            score = new_scores.pop(0)
            if doc.child_documents:
                child_documents = cls.replace_doc_scores(doc.child_documents, new_scores)
            else:
                child_documents = []
            new_scored_documents.append(ScoredDocument(document, score=score, child_documents=child_documents))
        return new_scored_documents

    @classmethod
    def normalize_scored_documents(cls, scored_documents: tp.List[ScoredDocument]) -> tp.List[ScoredDocument]:
        """Normalize the scores of scored documents using min-max scaling.

        Args:
            scored_documents (List[ScoredDocument]): Documents with existing scores.

        Returns:
            List[ScoredDocument]: Documents with normalized scores.
        """
        scores = cls.extract_doc_scores(scored_documents)
        new_scores = cls.normalize_doc_scores(scores).tolist()
        return cls.replace_doc_scores(scored_documents, new_scores)

    @classmethod
    def extract_doc_pair_scores(
        cls,
        emb_scored_documents: tp.List[ScoredDocument],
        bm25_scored_documents: tp.List[ScoredDocument],
    ) -> tp.List[tp.Tuple[float, float]]:
        """Recursively extract paired scores from embedding and BM25 scored documents.

        Args:
            emb_scored_documents (List[ScoredDocument]): Documents scored using embeddings.
            bm25_scored_documents (List[ScoredDocument]): Documents scored using BM25.

        Returns:
            List[Tuple[float, float]]: Pairs of scores from corresponding documents and their child documents.
        """

        def _score(doc):
            return doc.score if doc is not None else float("nan")

        def _children(doc):
            return doc.child_documents if doc is not None else []

        emb_map = {}
        bm25_map = {}
        order = []
        for d in emb_scored_documents:
            doc_id = d.document.id_
            emb_map[doc_id] = d
            order.append(doc_id)
        for d in bm25_scored_documents:
            doc_id = d.document.id_
            bm25_map[doc_id] = d
            if doc_id not in emb_map:
                order.append(doc_id)

        pair_scores = []
        for doc_id in order:
            emb_doc = emb_map.get(doc_id, None)
            bm25_doc = bm25_map.get(doc_id, None)
            pair_scores.append((_score(emb_doc), _score(bm25_doc)))
            if _children(emb_doc) or _children(bm25_doc):
                child_pair_scores = cls.extract_doc_pair_scores(_children(emb_doc), _children(bm25_doc))
                pair_scores.extend(child_pair_scores)
        return pair_scores

    def fuse_doc_pair_scores(self, doc_pair_scores: tp.Iterable[tp.Tuple[float, float]]) -> np.ndarray:
        """Fuse paired (embedding, BM25) scores with Reciprocal-Rank Fusion (RRF).

        Args:
            doc_pair_scores (Iterable[Tuple[float, float]]): Paired scores (embedding, BM25) to fuse.

        Returns:
            ndarray: Array of fused scores.
        """
        pair_scores = np.asarray(doc_pair_scores, dtype=float)
        emb_scores = pair_scores[:, 0]
        bm25_scores = pair_scores[:, 1]

        emb_tmp = np.where(np.isnan(emb_scores), -np.inf, emb_scores)
        bm25_tmp = np.where(np.isnan(bm25_scores), -np.inf, bm25_scores)
        emb_order = np.argsort(-emb_tmp, kind="mergesort")
        bm25_order = np.argsort(-bm25_tmp, kind="mergesort")

        emb_rank = np.empty(len(pair_scores), dtype=np.int32)
        bm25_rank = np.empty(len(pair_scores), dtype=np.int32)
        emb_rank[emb_order] = np.arange(1, len(pair_scores) + 1)
        bm25_rank[bm25_order] = np.arange(1, len(pair_scores) + 1)

        new_emb_scores = (1 - self.rrf_bm25_weight) / (self.rrf_k + emb_rank)
        new_bm25_scores = self.rrf_bm25_weight / (self.rrf_k + bm25_rank)
        return new_emb_scores + new_bm25_scores

    @classmethod
    def replace_doc_pair_scores(
        cls,
        emb_scored_documents: tp.List[ScoredDocument],
        bm25_scored_documents: tp.List[ScoredDocument],
        new_scores: tp.List[float],
    ) -> tp.List[ScoredDocument]:
        """Recursively replace scores in paired embedding and BM25 documents with new scores.

        Args:
            emb_scored_documents (List[ScoredDocument]): Documents scored using embeddings.
            bm25_scored_documents (List[ScoredDocument]): Documents scored using BM25.
            new_scores (List[float]): New scores to assign, consumed in order.

        Returns:
            List[ScoredDocument]: Updated documents with replaced paired scores.
        """

        def _children(doc):
            return doc.child_documents if doc is not None else []

        emb_map = {}
        bm25_map = {}
        order = []
        for d in emb_scored_documents:
            doc_id = d.document.id_
            emb_map[doc_id] = d
            order.append(doc_id)
        for d in bm25_scored_documents:
            doc_id = d.document.id_
            bm25_map[doc_id] = d
            if doc_id not in emb_map:
                order.append(doc_id)

        scored_documents = []
        for doc_id in order:
            emb_doc = emb_map.get(doc_id, None)
            bm25_doc = bm25_map.get(doc_id, None)
            if emb_doc is not None:
                document = emb_doc.document
            else:
                document = bm25_doc.document
            score = new_scores.pop(0)
            if _children(emb_doc) or _children(bm25_doc):
                child_documents = cls.replace_doc_pair_scores(_children(emb_doc), _children(bm25_doc), new_scores)
            else:
                child_documents = []
            scored_documents.append(ScoredDocument(document, score=score, child_documents=child_documents))
        return scored_documents

    def fuse_scored_documents(
        self,
        emb_scored_documents: tp.List[ScoredDocument],
        bm25_scored_documents: tp.List[ScoredDocument],
    ) -> tp.List[ScoredDocument]:
        """Fuse embedding and BM25 scored documents by merging and updating their scores.

        Args:
            emb_scored_documents (List[ScoredDocument]): Documents scored using embeddings.
            bm25_scored_documents (List[ScoredDocument]): Documents scored using BM25.

        Returns:
            List[ScoredDocument]: Fused scored documents with updated scores.
        """
        doc_pair_scores = self.extract_doc_pair_scores(emb_scored_documents, bm25_scored_documents)
        new_scores = self.fuse_doc_pair_scores(doc_pair_scores).tolist()
        return self.replace_doc_pair_scores(emb_scored_documents, bm25_scored_documents, new_scores)

    def rank_documents(
        self,
        query: str,
        documents: tp.Optional[tp.Iterable[StoreDocument]] = None,
        top_k: tp.TopKLike = None,
        min_top_k: tp.TopKLike = None,
        max_top_k: tp.TopKLike = None,
        cutoff: tp.Optional[float] = None,
        refresh: bool = False,
        refresh_documents: tp.Optional[bool] = None,
        refresh_embeddings: tp.Optional[bool] = None,
        return_chunks: bool = False,
        return_scores: bool = False,
    ) -> tp.RankedDocuments:
        """Rank documents based on their relevance to a query.

        The method retrieves scored documents using embedding and BM25 strategies (or both in hybrid mode),
        fuses and normalizes their scores, and then sorts them to identify the most relevant documents.
        Top-k parameters and score cutoff are resolved using `DocumentRanker.resolve_top_k` and
        `DocumentRanker.top_k_from_cutoff`.

        Args:
            query (str): Query string to evaluate document relevance.
            documents (Optional[Iterable[StoreDocument]]): Collection of documents to rank.

                If None, documents from the document store are used.
            top_k (TopKLike): Number or percentage of top documents to return, or a method to determine it.
            min_top_k (TopKLike): Minimum limit for determining top documents.
            max_top_k (TopKLike): Maximum limit for determining top documents.
            cutoff (Optional[float]): Score threshold to filter documents.
            refresh (bool): Flag to refresh both documents and embeddings.
            refresh_documents (Optional[bool]): Flag to refresh documents; defaults to `refresh`.
            refresh_embeddings (Optional[bool]): Flag to refresh embeddings; defaults to `refresh`.
            return_chunks (bool): Whether to return document chunks.
            return_scores (bool): Whether to return scored documents with their scores.

        Returns:
            RankedDocuments: Documents ranked by relevance to the query.
        """
        if documents is not None:
            documents = list(documents)
        if self.search_method in ("embeddings", "hybrid", "embeddings_fallback", "hybrid_fallback"):
            try:
                emb_scored_documents = self.score_documents(
                    query,
                    documents=documents,
                    refresh=refresh,
                    refresh_documents=refresh_documents,
                    refresh_embeddings=refresh_embeddings,
                    return_chunks=return_chunks,
                    return_documents=True,
                    with_fallback=self.search_method in ("embeddings_fallback", "hybrid_fallback"),
                )
            except FallbackError as e:
                warn(f'Fallback triggered: "{e}"')
                emb_scored_documents = None
        else:
            emb_scored_documents = None
        if self.search_method in ("bm25", "hybrid") or (
            emb_scored_documents is None and self.search_method in ("embeddings_fallback", "hybrid_fallback")
        ):
            bm25_scored_documents = self.bm25_score_documents(
                query,
                documents=documents,
                refresh=refresh,
                refresh_documents=refresh_documents,
                return_chunks=return_chunks,
                return_documents=True,
            )
        else:
            bm25_scored_documents = None
        if emb_scored_documents is not None and bm25_scored_documents is not None:
            scored_documents = self.fuse_scored_documents(emb_scored_documents, bm25_scored_documents)
        elif emb_scored_documents is not None:
            scored_documents = emb_scored_documents
        elif bm25_scored_documents is not None:
            scored_documents = bm25_scored_documents
        else:
            raise NotImplementedError
        if self.normalize_scores:
            scored_documents = self.normalize_scored_documents(scored_documents)
        scored_documents = sorted(scored_documents, key=lambda x: (not np.isnan(x.score), x.score), reverse=True)
        scores = [document.score for document in scored_documents]

        int_top_k = top_k is not None and checks.is_int(top_k)
        top_k = self.resolve_top_k(scores, top_k=top_k)
        min_top_k = self.resolve_top_k(scores, top_k=min_top_k)
        max_top_k = self.resolve_top_k(scores, top_k=max_top_k)
        cutoff = self.top_k_from_cutoff(scores, cutoff=cutoff)
        if not int_top_k and min_top_k is not None and min_top_k > top_k:
            top_k = min_top_k
        if not int_top_k and max_top_k is not None and max_top_k < top_k:
            top_k = max_top_k
        if cutoff is not None and min_top_k is not None and min_top_k > cutoff:
            cutoff = min_top_k
        if cutoff is not None and max_top_k is not None and max_top_k < cutoff:
            cutoff = max_top_k
        if top_k is None:
            top_k = len(scores)
        if cutoff is None:
            cutoff = len(scores)
        top_k = min(top_k, cutoff)
        if top_k == 0:
            raise ValueError("No documents selected after ranking. Change top_k or cutoff.")
        scored_documents = scored_documents[:top_k]
        if return_scores:
            return scored_documents
        return [document.document for document in scored_documents]


def embed_documents(
    documents: tp.Iterable[StoreDocument],
    refresh: bool = False,
    refresh_documents: tp.Optional[bool] = None,
    refresh_embeddings: tp.Optional[bool] = None,
    return_embeddings: bool = False,
    return_documents: bool = False,
    doc_ranker: tp.Optional[tp.MaybeType[DocumentRanker]] = None,
    **kwargs,
) -> tp.Optional[tp.EmbeddedDocuments]:
    """Embed the provided documents using a `DocumentRanker`.

    Args:
        documents (Iterable[StoreDocument]): Collection of documents to embed.
        refresh (bool): Flag to refresh both documents and embeddings.
        refresh_documents (Optional[bool]): Flag to refresh documents; defaults to `refresh`.
        refresh_embeddings (Optional[bool]): Flag to refresh embeddings; defaults to `refresh`.
        return_embeddings (bool): Flag indicating whether to return embeddings.
        return_documents (bool): If True, include original document objects in the output.
        doc_ranker (Optional[MaybeType[DocumentRanker]]): `DocumentRanker` class or instance.
        **kwargs: Keyword arguments to initialize or update `doc_ranker`.

    Returns:
        Optional[EmbeddedDocuments]: Embedded documents output.
    """
    if documents is not None and not documents:
        return []
    if doc_ranker is None:
        doc_ranker = DocumentRanker
    if isinstance(doc_ranker, type):
        checks.assert_subclass_of(doc_ranker, DocumentRanker, "doc_ranker")
        doc_ranker = doc_ranker(**kwargs)
    else:
        checks.assert_instance_of(doc_ranker, DocumentRanker, "doc_ranker")
        if kwargs:
            doc_ranker = doc_ranker.replace(**kwargs)
    return doc_ranker.embed_documents(
        documents,
        refresh=refresh,
        refresh_documents=refresh_documents,
        refresh_embeddings=refresh_embeddings,
        return_embeddings=return_embeddings,
        return_documents=return_documents,
    )


def rank_documents(
    query: str,
    documents: tp.Optional[tp.Iterable[StoreDocument]] = None,
    top_k: tp.TopKLike = None,
    min_top_k: tp.TopKLike = None,
    max_top_k: tp.TopKLike = None,
    cutoff: tp.Optional[float] = None,
    refresh: bool = False,
    refresh_documents: tp.Optional[bool] = None,
    refresh_embeddings: tp.Optional[bool] = None,
    return_chunks: bool = False,
    return_scores: bool = False,
    doc_ranker: tp.Optional[tp.MaybeType[DocumentRanker]] = None,
    **kwargs,
) -> tp.RankedDocuments:
    """Rank documents based on their relevance to a query using a `DocumentRanker`.

    Args:
        query (str): Query string for ranking.
        documents (Optional[Iterable[StoreDocument]]): Collection of documents to rank.

            If None, documents from the document store are used.
        top_k (TopKLike): Number or percentage of top documents to return, or a method to determine it.
        min_top_k (TopKLike): Minimum limit for determining top documents.
        max_top_k (TopKLike): Maximum limit for determining top documents.
        cutoff (Optional[float]): Score threshold to filter documents.
        refresh (bool): Flag to refresh both documents and embeddings.
        refresh_documents (Optional[bool]): Flag to refresh documents; defaults to `refresh`.
        refresh_embeddings (Optional[bool]): Flag to refresh embeddings; defaults to `refresh`.
        return_chunks (bool): Whether to return document chunks.
        return_scores (bool): Whether to return scored documents with their scores.
        doc_ranker (Optional[MaybeType[DocumentRanker]]): `DocumentRanker` class or instance.
        **kwargs: Keyword arguments to initialize or update `doc_ranker`.

    Returns:
        RankedDocuments: Ranked documents based on the query relevance.
    """
    if documents is not None and not documents:
        return []
    if doc_ranker is None:
        doc_ranker = DocumentRanker
    if isinstance(doc_ranker, type):
        checks.assert_subclass_of(doc_ranker, DocumentRanker, "doc_ranker")
        doc_ranker = doc_ranker(**kwargs)
    else:
        checks.assert_instance_of(doc_ranker, DocumentRanker, "doc_ranker")
        if kwargs:
            doc_ranker = doc_ranker.replace(**kwargs)
    return doc_ranker.rank_documents(
        query,
        documents=documents,
        top_k=top_k,
        min_top_k=min_top_k,
        max_top_k=max_top_k,
        cutoff=cutoff,
        refresh=refresh,
        refresh_documents=refresh_documents,
        refresh_embeddings=refresh_embeddings,
        return_chunks=return_chunks,
        return_scores=return_scores,
    )


RankableT = tp.TypeVar("RankableT", bound="Rankable")


class Rankable(HasSettings):
    """Abstract class representing an entity that supports embedding and ranking operations.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and its sub-configuration `chat`.
    """

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat"]

    def embed(
        self: RankableT,
        refresh: bool = False,
        refresh_documents: tp.Optional[bool] = None,
        refresh_embeddings: tp.Optional[bool] = None,
        return_embeddings: bool = False,
        return_documents: bool = False,
        **kwargs,
    ) -> tp.Optional[RankableT]:
        """Embed the instance's documents.

        !!! abstract
            This method should be overridden in a subclass.

        Args:
            refresh (bool): Flag to refresh both documents and embeddings.
            refresh_documents (Optional[bool]): Flag to refresh documents; defaults to `refresh`.
            refresh_embeddings (Optional[bool]): Flag to refresh embeddings; defaults to `refresh`.
            return_embeddings (bool): Flag indicating whether to return embeddings.
            return_documents (bool): If True, include original document objects in the output.
            **kwargs: Additional keyword arguments.

        Returns:
            Optional[Rankable]: Updated instance with embedded documents, if available.
        """
        raise NotImplementedError

    def rank(
        self: RankableT,
        query: str,
        top_k: tp.TopKLike = None,
        min_top_k: tp.TopKLike = None,
        max_top_k: tp.TopKLike = None,
        cutoff: tp.Optional[float] = None,
        refresh: bool = False,
        refresh_documents: tp.Optional[bool] = None,
        refresh_embeddings: tp.Optional[bool] = None,
        return_chunks: bool = False,
        return_scores: bool = False,
        **kwargs,
    ) -> RankableT:
        """Rank documents based on their relevance to a provided query.

        !!! abstract
            This method should be overridden in a subclass.

        Args:
            query (str): Query string to evaluate document relevance.
            top_k (TopKLike): Number or percentage of top documents to return, or a method to determine it.
            min_top_k (TopKLike): Minimum limit for determining top documents.
            max_top_k (TopKLike): Maximum limit for determining top documents.
            cutoff (Optional[float]): Score threshold to filter documents.
            refresh (bool): Flag to refresh both documents and embeddings.
            refresh_documents (Optional[bool]): Flag to refresh documents; defaults to `refresh`.
            refresh_embeddings (Optional[bool]): Flag to refresh embeddings; defaults to `refresh`.
            return_chunks (bool): Whether to return document chunks.
            return_scores (bool): Whether to return scored documents with their scores.
            **kwargs: Additional keyword arguments.

        Returns:
            Rankable: Updated instance with ranked documents.
        """
        raise NotImplementedError


class Contextable(HasSettings):
    """Abstract class that provides functionality to generate a textual context.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and its sub-configuration `chat`.
    """

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat"]

    def to_context(self, *args, **kwargs) -> str:
        """Convert the instance into a textual context.

        !!! abstract
            This method should be overridden in a subclass.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Textual context representation.
        """
        raise NotImplementedError

    def count_tokens(
        self,
        to_context_kwargs: tp.KwargsLike = None,
        tokenizer: tp.TokenizerLike = None,
        tokenizer_kwargs: tp.KwargsLike = None,
    ) -> int:
        """Count the number of tokens in the generated context.
        tokenizer_kwargs (KwargsLike): Keyword arguments to initialize or update `tokenizer`.

        Args:
            to_context_kwargs (KwargsLike): Keyword arguments for `Contextable.to_context`.
            tokenizer (TokenizerLike): Identifier, subclass, or instance of
                `vectorbtpro.knowledge.tokenization.Tokenizer`.

                Resolved using `vectorbtpro.knowledge.tokenization.resolve_tokenizer`.

        Returns:
            int: Number of tokens in the context.
        """
        to_context_kwargs = self.resolve_setting(to_context_kwargs, "to_context_kwargs", merge=True)
        tokenizer = self.resolve_setting(tokenizer, "tokenizer", default=None)
        tokenizer_kwargs = self.resolve_setting(tokenizer_kwargs, "tokenizer_kwargs", default=None, merge=True)

        context = self.to_context(**to_context_kwargs)
        tokenizer = resolve_tokenizer(tokenizer)
        if isinstance(tokenizer, type):
            tokenizer = tokenizer(**tokenizer_kwargs)
        elif tokenizer_kwargs:
            tokenizer = tokenizer.replace(**tokenizer_kwargs)
        return len(tokenizer.encode(context))

    def create_chat(
        self,
        to_context_kwargs: tp.KwargsLike = None,
        completions: tp.CompletionsLike = None,
        **kwargs,
    ) -> tp.Completions:
        """Create a chat interface using the generated context.

        Args:
            to_context_kwargs (KwargsLike): Keyword arguments for `Contextable.to_context`.
            completions (CompletionsLike): Identifier, subclass, or instance of
                `vectorbtpro.knowledge.completions.Completions`.

                Resolved using `vectorbtpro.knowledge.completions.resolve_completions`.
            **kwargs: Keyword arguments to initialize or update `completions`.

        Returns:
            Completions: Instance of `vectorbtpro.knowledge.completions.Completions`
                configured with the generated context.

        Examples:
            ```pycon
            >>> chat = asset.create_chat()

            >>> chat.get_completion("What's the value under 'xyz'?")
            The value under 'xyz' is 123.

            >>> chat.get_completion("Are you sure?")
            Yes, I am sure. The value under 'xyz' is 123 for the entry where `s` is "EFG".
            ```
        """
        to_context_kwargs = self.resolve_setting(to_context_kwargs, "to_context_kwargs", merge=True)
        context = self.to_context(**to_context_kwargs)
        completions = resolve_completions(completions=completions)
        if isinstance(completions, type):
            completions = completions(context=context, **kwargs)
        else:
            completions = completions.replace(context=context, **kwargs)
        return completions

    @hybrid_method
    def chat(
        cls_or_self,
        message: str,
        chat_history: tp.Optional[tp.ChatHistory] = None,
        *,
        return_chat: bool = False,
        **kwargs,
    ) -> tp.MaybeChatOutput:
        """Chat with a language model using the instance as context.

        !!! note
            Context is recalculated each time this method is invoked. For multiple turns,
            it's more efficient to use `Contextable.create_chat`.

        Args:
            message (str): Message to send to the language model.
            chat_history (Optional[ChatHistory]): Chat history, a list of dictionaries with defined roles.
            return_chat (bool): Flag indicating whether to return both the completion and the chat instance.
            **kwargs: Keyword arguments for `Contextable.create_chat`.

        Returns:
            MaybeChatOutput: Completion response or a tuple of the response and the chat instance.

        Examples:
            ```pycon
            >>> asset.chat("What's the value under 'xyz'?")
            The value under 'xyz' is 123.

            >>> chat_history = []
            >>> asset.chat("What's the value under 'xyz'?", chat_history=chat_history)
            The value under 'xyz' is 123.

            >>> asset.chat("Are you sure?", chat_history=chat_history)
            Yes, I am sure. The value under 'xyz' is 123 for the entry where `s` is "EFG".
            ```
        """
        if isinstance(cls_or_self, type):
            args, kwargs = get_forward_args(super().chat, locals())
            return super().chat(*args, **kwargs)

        completions = cls_or_self.create_chat(chat_history=chat_history, **kwargs)
        if return_chat:
            return completions.get_completion(message), completions
        return completions.get_completion(message)


class RankContextable(Rankable, Contextable):
    """Abstract class combining `Rankable` and `Contextable` functionalities.

    This abstract class integrates ranking with contextual chat processing by applying
    ranking methods to chat queries when ranking parameters are provided.
    """

    @hybrid_method
    def chat(
        cls_or_self,
        message: str,
        chat_history: tp.Optional[tp.ChatHistory] = None,
        *,
        incl_past_queries: tp.Optional[bool] = None,
        rank: tp.Optional[bool] = None,
        top_k: tp.TopKLike = None,
        min_top_k: tp.TopKLike = None,
        max_top_k: tp.TopKLike = None,
        cutoff: tp.Optional[float] = None,
        return_chunks: tp.Optional[bool] = None,
        rank_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeChatOutput:
        """Return the chat output with optional ranking applied.

        If `rank` is True, or if `rank` is None and any ranking parameter (`top_k`, `min_top_k`,
        `max_top_k`, `cutoff`, or `return_chunks`) is specified, process the query using
        `Rankable.rank` before delegating to `Contextable.chat`.

        Args:
            message (str): Message to send to the language model.
            chat_history (Optional[ChatHistory]): Chat history, a list of dictionaries with defined roles.
            incl_past_queries (Optional[bool]): Whether to include past queries in the ranking process.
            rank (Optional[bool]): Flag indicating whether to apply ranking.
            top_k (TopKLike): Number or percentage of top documents to return, or a method to determine it.
            min_top_k (TopKLike): Minimum limit for determining top documents.
            max_top_k (TopKLike): Maximum limit for determining top documents.
            cutoff (Optional[float]): Score threshold to filter documents.
            return_chunks (Optional[bool]): Whether to return document chunks.
            rank_kwargs (KwargsLike): Keyword arguments for `Rankable.rank`.
            **kwargs: Keyword arguments for `Contextable.chat`.

        Returns:
            MaybeChatOutput: Completion response or a tuple of the response and the chat instance.
        """
        if isinstance(cls_or_self, type):
            args, kwargs = get_forward_args(super().chat, locals())
            return super().chat(*args, **kwargs)

        incl_past_queries = cls_or_self.resolve_setting(incl_past_queries, "incl_past_queries")
        rank = cls_or_self.resolve_setting(rank, "rank")
        rank_kwargs = cls_or_self.resolve_setting(rank_kwargs, "rank_kwargs", merge=True)
        def_top_k = rank_kwargs.pop("top_k")
        if top_k is None:
            top_k = def_top_k
        def_min_top_k = rank_kwargs.pop("min_top_k")
        if min_top_k is None:
            min_top_k = def_min_top_k
        def_max_top_k = rank_kwargs.pop("max_top_k")
        if max_top_k is None:
            max_top_k = def_max_top_k
        def_cutoff = rank_kwargs.pop("cutoff")
        if cutoff is None:
            cutoff = def_cutoff
        def_return_chunks = rank_kwargs.pop("return_chunks")
        if return_chunks is None:
            return_chunks = def_return_chunks
        if rank or (rank is None and (top_k or min_top_k or max_top_k or cutoff or return_chunks)):
            if incl_past_queries and chat_history is not None:
                queries = []
                for message_dct in chat_history:
                    if "role" in message_dct and message_dct["role"] == "user":
                        queries.append(message_dct["content"])
                queries.append(message)
                if len(queries) > 1:
                    query = "\n\n".join(queries)
                else:
                    query = queries[0]
            else:
                query = message
            _cls_or_self = cls_or_self.rank(
                query,
                top_k=top_k,
                min_top_k=min_top_k,
                max_top_k=max_top_k,
                cutoff=cutoff,
                return_chunks=return_chunks,
                **rank_kwargs,
            )
        else:
            _cls_or_self = cls_or_self
        return Contextable.chat.__func__(_cls_or_self, message, chat_history, **kwargs)
