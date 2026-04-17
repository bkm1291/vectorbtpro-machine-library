# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing classes and utilities for embeddings."""

import inspect
import sys
import time

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts, Configured
from vectorbtpro.utils.parsing import get_func_arg_names, get_func_kwargs
from vectorbtpro.utils.pbar import ProgressBar

if tp.TYPE_CHECKING:
    from openai import OpenAI as OpenAIT
else:
    OpenAIT = "openai.OpenAI"
if tp.TYPE_CHECKING:
    from google.genai import Client as GenAIClientT
else:
    GenAIClientT = "google.genai.Client"
if tp.TYPE_CHECKING:
    from huggingface_hub import InferenceClient as InferenceClientT
else:
    InferenceClientT = "huggingface_hub.InferenceClient"
if tp.TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding as BaseEmbeddingT
else:
    BaseEmbeddingT = "llama_index.core.embeddings.BaseEmbedding"
if tp.TYPE_CHECKING:
    from ollama import Client as OllamaClientT
else:
    OllamaClientT = "ollama.Client"

__all__ = [
    "Embeddings",
    "OpenAIEmbeddings",
    "GeminiEmbeddings",
    "HFInferenceEmbeddings",
    "LiteLLMEmbeddings",
    "LlamaIndexEmbeddings",
    "OllamaEmbeddings",
    "embed",
]


class Embeddings(Configured):
    """Abstract class for embedding providers.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and
        its sub-configurations `chat` and `chat.embeddings_config`.

    Args:
        batch_size (Optional[int]): Batch size for processing queries.

            Use None to disable batching.
        show_progress (Optional[bool]): Flag indicating whether to display the progress bar.
        pbar_kwargs (Kwargs): Keyword arguments for configuring the progress bar.
        template_context (Kwargs): Additional context for template substitution.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the class."""

    _expected_keys_mode: tp.ExpectedKeysMode = "disable"

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat", "knowledge.chat.embeddings_config"]

    def __init__(
        self,
        batch_size: tp.Optional[int] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            batch_size=batch_size,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            template_context=template_context,
            **kwargs,
        )

        batch_size = self.resolve_setting(batch_size, "batch_size")
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        self._batch_size = batch_size
        self._show_progress = show_progress
        self._pbar_kwargs = pbar_kwargs
        self._template_context = template_context

    @property
    def batch_size(self) -> tp.Optional[int]:
        """Batch size used for processing queries.

        Use None to disable batching.

        Returns:
            Optional[int]: Batch size.
        """
        return self._batch_size

    @property
    def show_progress(self) -> tp.Optional[bool]:
        """Whether to display a progress bar.

        Returns:
            Optional[bool]: True if progress bar is shown, False otherwise.
        """
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `vectorbtpro.utils.pbar.ProgressBar`.

        Returns:
            Kwargs: Keyword arguments for the progress bar.
        """
        return self._pbar_kwargs

    @property
    def template_context(self) -> tp.Kwargs:
        """Additional context for template substitution.

        Returns:
            Kwargs: Dictionary of context variables for template substitution.
        """
        return self._template_context

    @property
    def model(self) -> tp.Optional[str]:
        """Model identifier.

        Returns:
            Optional[str]: Model identifier; None by default.
        """
        return None

    def get_embedding(self, query: str) -> tp.List[float]:
        """Return the embedding vector for the given query.

        !!! abstract
            This method should be overridden in a subclass.

        Args:
            query (str): Query text.

        Returns:
            List[float]: Embedding vector.
        """
        raise NotImplementedError

    def get_embedding_batch(self, batch: tp.List[str]) -> tp.List[tp.List[float]]:
        """Return a batch of embedding vectors for a list of queries.

        Args:
            batch (List[str]): List of query texts.

        Returns:
            List[List[float]]: List containing an embedding vector for each query.
        """
        return [self.get_embedding(query) for query in batch]

    def iter_embedding_batches(self, queries: tp.List[str]) -> tp.Iterator[tp.List[tp.List[float]]]:
        """Return an iterator over batches of embeddings.

        Args:
            queries (List[str]): List of query texts.

        Returns:
            Iterator[List[List[float]]]: Iterator yielding batches of embedding vectors.
        """
        from vectorbtpro.utils.pbar import ProgressBar

        if self.batch_size is not None:
            batches = [queries[i : i + self.batch_size] for i in range(0, len(queries), self.batch_size)]
        else:
            batches = [queries]
        pbar_kwargs = merge_dicts(dict(prefix="get_embeddings"), self.pbar_kwargs)
        with ProgressBar(total=len(queries), show_progress=self.show_progress, **pbar_kwargs) as pbar:
            for batch in batches:
                yield self.get_embedding_batch(batch)
                pbar.update(len(batch))

    def get_embeddings(self, queries: tp.List[str]) -> tp.List[tp.List[float]]:
        """Return embeddings for multiple queries.

        Args:
            queries (List[str]): List of query texts.

        Returns:
            List[List[float]]: List containing an embedding vector for each query.
        """
        return [embedding for batch in self.iter_embedding_batches(queries) for embedding in batch]


class OpenAIEmbeddings(Embeddings):
    """Embeddings class for OpenAI.

    !!! info
        For default settings, see `chat.embeddings_configs.openai` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): OpenAI model identifier.
        client_kwargs (KwargsLike): Keyword arguments for `openai.OpenAI`.
        embeddings_kwargs (KwargsLike): Keyword arguments for `openai.resources.embeddings.Embeddings.create`.
        **kwargs: Keyword arguments for `Embeddings` or used as `client_kwargs` or `embeddings_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "openai"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.openai"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        embeddings_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            embeddings_kwargs=embeddings_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("openai")
        from openai import OpenAI

        openai_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = openai_config.pop("model", None)
        def_client_kwargs = openai_config.pop("client_kwargs", None)
        def_embeddings_kwargs = openai_config.pop("embeddings_kwargs", None)

        if model is None:
            model = def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = self.get_init_arg_names()
        for k in list(openai_config.keys()):
            if k in init_arg_names:
                openai_config.pop(k)

        client_arg_names = set(get_func_arg_names(OpenAI.__init__))
        _client_kwargs = {}
        _embeddings_kwargs = {}
        for k, v in openai_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _embeddings_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        embeddings_kwargs = merge_dicts(_embeddings_kwargs, def_embeddings_kwargs, embeddings_kwargs)
        client = OpenAI(**client_kwargs)

        self._model = model
        self._client = client
        self._embeddings_kwargs = embeddings_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> OpenAIT:
        """OpenAI client instance.

        Returns:
            OpenAI: OpenAI client instance.
        """
        return self._client

    @property
    def embeddings_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `openai.resources.embeddings.Embeddings.create`.

        Returns:
            Kwargs: Keyword arguments for creating embeddings.
        """
        return self._embeddings_kwargs

    def get_embedding(self, query: str) -> tp.List[float]:
        response = self.client.embeddings.create(input=query, model=self.model, **self.embeddings_kwargs)
        return response.data[0].embedding

    def get_embedding_batch(self, batch: tp.List[str]) -> tp.List[tp.List[float]]:
        response = self.client.embeddings.create(input=batch, model=self.model, **self.embeddings_kwargs)
        return [embedding.embedding for embedding in response.data]


class GeminiEmbeddings(Embeddings):
    """Embeddings class for Google GenAI (Gemini).

    !!! info
        For default settings, see `chat.embeddings_configs.gemini` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): Gemini model identifier.
        client_kwargs (KwargsLike): Keyword arguments for `google.genai.Client`.
        embeddings_kwargs (KwargsLike): Keyword arguments for `google.genai.Client.models.embed_content`.
        **kwargs: Keyword arguments for `Embeddings` or used as `client_kwargs` or `embeddings_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "gemini"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.gemini"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        embeddings_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            embeddings_kwargs=embeddings_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("google.genai")
        from google.genai import Client

        gemini_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = gemini_config.pop("model", None)
        def_client_kwargs = gemini_config.pop("client_kwargs", None)
        def_embeddings_kwargs = gemini_config.pop("embeddings_kwargs", None)

        if model is None:
            model = def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = self.get_init_arg_names()
        for k in list(gemini_config.keys()):
            if k in init_arg_names:
                gemini_config.pop(k)

        client_arg_names = set(get_func_arg_names(Client.__init__))
        _client_kwargs = {}
        _embeddings_kwargs = {}
        for k, v in gemini_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _embeddings_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        embeddings_kwargs = merge_dicts(_embeddings_kwargs, def_embeddings_kwargs, embeddings_kwargs)

        client = Client(**client_kwargs)

        self._model = model
        self._client = client
        self._embeddings_kwargs = embeddings_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> GenAIClientT:
        """Gemini client instance.

        Returns:
            Client: Gemini client instance.
        """
        return self._client

    @property
    def embeddings_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `google.genai.Client.models.embed_content`.

        Returns:
            Kwargs: Keyword arguments for generating embeddings.
        """
        return self._embeddings_kwargs

    def normalize_embedding(self, embedding: tp.List[float]) -> tp.List[float]:
        """Normalize a single embedding vector.

        Args:
            embedding (List[float]): Embedding vector to normalize.

        Returns:
            List[float]: Normalized embedding vector.
        """
        embedding_values_np = np.array(embedding)
        normed_embedding = embedding_values_np / np.linalg.norm(embedding_values_np)
        return normed_embedding.tolist()

    def get_embedding(self, query: str) -> tp.List[float]:
        from google.genai.errors import ClientError

        attempted = False
        while True:
            try:
                response = self.client.models.embed_content(model=self.model, contents=query, **self.embeddings_kwargs)
                return self.normalize_embedding(response.embeddings[0].values)
            except ClientError as e:
                if e.code == 429 and not attempted:
                    time.sleep(60)
                    attempted = True
                else:
                    raise e

    def get_embedding_batch(self, batch: tp.List[str]) -> tp.List[tp.List[float]]:
        from google.genai.errors import ClientError

        attempted = False
        while True:
            try:
                response = self.client.models.embed_content(model=self.model, contents=batch, **self.embeddings_kwargs)
                return list(map(lambda x: self.normalize_embedding(x.values), response.embeddings))
            except ClientError as e:
                if e.code == 429 and not attempted:
                    time.sleep(60)
                    attempted = True
                else:
                    raise e


class HFInferenceEmbeddings(Embeddings):
    """Embeddings class for HuggingFace Inference.

    !!! info
        For default settings, see `chat.embeddings_configs.hf_inference` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): HuggingFace model identifier.
        client_kwargs (KwargsLike): Keyword arguments for `huggingface_hub.InferenceClient`.
        feature_extraction_kwargs (KwargsLike): Keyword arguments for `huggingface_hub.InferenceClient.feature_extraction`.
        **kwargs: Keyword arguments for `Embeddings` or used as `client_kwargs` or `feature_extraction_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "hf_inference"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.hf_inference"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        feature_extraction_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            feature_extraction_kwargs=feature_extraction_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("huggingface_hub")
        from huggingface_hub import InferenceClient

        hf_inference_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = hf_inference_config.pop("model", None)
        def_client_kwargs = hf_inference_config.pop("client_kwargs", None)
        def_feature_extraction_kwargs = hf_inference_config.pop("feature_extraction_kwargs", None)

        if model is None:
            model = def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = self.get_init_arg_names()
        for k in list(hf_inference_config.keys()):
            if k in init_arg_names:
                hf_inference_config.pop(k)

        client_arg_names = set(get_func_arg_names(InferenceClient.__init__))
        _client_kwargs = {}
        _feature_extraction_kwargs = {}
        for k, v in hf_inference_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _feature_extraction_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        feature_extraction_kwargs = merge_dicts(
            _feature_extraction_kwargs,
            def_feature_extraction_kwargs,
            feature_extraction_kwargs,
        )
        client = InferenceClient(model=model, **client_kwargs)

        self._model = model
        self._client = client
        self._feature_extraction_kwargs = feature_extraction_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> InferenceClientT:
        """HuggingFace Inference client instance.

        Returns:
            InferenceClient: HuggingFace Inference client instance.
        """
        return self._client

    @property
    def feature_extraction_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `huggingface_hub.InferenceClient.feature_extraction`.

        Returns:
            Kwargs: Keyword arguments for feature extraction.
        """
        return self._feature_extraction_kwargs

    def get_embedding(self, query: str) -> tp.List[float]:
        return self.client.feature_extraction(query, **self.feature_extraction_kwargs)[0].tolist()

    def get_embedding_batch(self, batch: tp.List[str]) -> tp.List[tp.List[float]]:
        return self.client.feature_extraction(batch, **self.feature_extraction_kwargs).tolist()


class LiteLLMEmbeddings(Embeddings):
    """Embeddings class for LiteLLM.

    !!! info
        For default settings, see `chat.embeddings_configs.litellm` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): LiteLLM model identifier.
        embedding_kwargs (KwargsLike): Keyword arguments for `litellm.embedding`.
        **kwargs: Keyword arguments for `Embeddings` or used as `embedding_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "litellm"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.litellm"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        embedding_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            model=model,
            embedding_kwargs=embedding_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("litellm")

        litellm_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = litellm_config.pop("model", None)
        def_embedding_kwargs = litellm_config.pop("embedding_kwargs", None)

        if model is None:
            model = def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = self.get_init_arg_names()
        for k in list(litellm_config.keys()):
            if k in init_arg_names:
                litellm_config.pop(k)
        embedding_kwargs = merge_dicts(litellm_config, def_embedding_kwargs, embedding_kwargs)

        self._model = model
        self._embedding_kwargs = embedding_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def embedding_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `litellm.embedding`.

        Returns:
            Kwargs: Keyword arguments for creating embeddings.
        """
        return self._embedding_kwargs

    def get_embedding(self, query: str) -> tp.List[float]:
        from litellm import embedding

        response = embedding(self.model, input=query, **self.embedding_kwargs)
        return response.data[0]["embedding"]

    def get_embedding_batch(self, batch: tp.List[str]) -> tp.List[tp.List[float]]:
        from litellm import embedding

        response = embedding(self.model, input=batch, **self.embedding_kwargs)
        return [embedding["embedding"] for embedding in response.data]


class LlamaIndexEmbeddings(Embeddings):
    """Embeddings class for LlamaIndex.

    This class initializes embeddings for LlamaIndex using a specified identifier or instance.
    It combines configuration from `vectorbtpro._settings.knowledge` with provided parameters.

    !!! info
        For default settings, see `chat.embeddings_configs.llama_index` in `vectorbtpro._settings.knowledge`.

    Args:
        embedding (Union[None, str, BaseEmbedding]): Embedding identifier or instance.

            If None, a default from settings is used.
        embedding_kwargs (KwargsLike): Keyword arguments for embedding initialization.
        **kwargs: Keyword arguments for `Embeddings` or used as `embedding_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "llama_index"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.llama_index"

    def __init__(
        self,
        embedding: tp.Union[None, str, tp.MaybeType[BaseEmbeddingT]] = None,
        embedding_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            embedding=embedding,
            embedding_kwargs=embedding_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("llama_index")
        from llama_index.core.embeddings import BaseEmbedding

        llama_index_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_embedding = llama_index_config.pop("embedding", None)
        def_embedding_kwargs = llama_index_config.pop("embedding_kwargs", None)

        if embedding is None:
            embedding = def_embedding
        if embedding is None:
            raise ValueError("Must provide an embedding name or path")
        init_arg_names = self.get_init_arg_names()
        for k in list(llama_index_config.keys()):
            if k in init_arg_names:
                llama_index_config.pop(k)

        if isinstance(embedding, str):
            import llama_index.embeddings
            from vectorbtpro.utils.module_ import search_package

            def _match_func(k, v):
                if isinstance(v, type) and issubclass(v, BaseEmbedding):
                    if "." in embedding:
                        if k.endswith(embedding):
                            return True
                    else:
                        if k.split(".")[-1].lower() == embedding.lower():
                            return True
                        if k.split(".")[-1].replace("Embedding", "").lower() == embedding.lower().replace("_", ""):
                            return True
                return False

            found_embedding = search_package(
                llama_index.embeddings,
                _match_func,
                path_attrs=True,
                return_first=True,
            )
            if found_embedding is None:
                raise ValueError(f"Embedding {embedding!r} not found")
            embedding = found_embedding
        if isinstance(embedding, type):
            checks.assert_subclass_of(embedding, BaseEmbedding, arg_name="embedding")
            embedding_name = embedding.__name__.replace("Embedding", "").lower()
            module_name = embedding.__module__
        else:
            checks.assert_instance_of(embedding, BaseEmbedding, arg_name="embedding")
            embedding_name = type(embedding).__name__.replace("Embedding", "").lower()
            module_name = type(embedding).__module__
        embedding_configs = llama_index_config.pop("embedding_configs", {})
        if embedding_name in embedding_configs:
            llama_index_config = merge_dicts(llama_index_config, embedding_configs[embedding_name])
        elif module_name in embedding_configs:
            llama_index_config = merge_dicts(llama_index_config, embedding_configs[module_name])
        embedding_kwargs = merge_dicts(llama_index_config, def_embedding_kwargs, embedding_kwargs)
        model_name = embedding_kwargs.get("model_name", None)
        if model_name is None:
            func_kwargs = get_func_kwargs(type(embedding).__init__)
            model_name = func_kwargs.get("model_name", None)
        if isinstance(embedding, type):
            embedding = embedding(**embedding_kwargs)
        elif len(kwargs) > 0:
            raise ValueError("Cannot apply config to already initialized embedding")

        self._model = model_name
        self._embedding = embedding

    @property
    def model(self) -> tp.Optional[str]:
        return self._model

    @property
    def embedding(self) -> BaseEmbeddingT:
        """Underlying embedding instance.

        Returns:
            BaseEmbedding: Embedding instance.
        """
        return self._embedding

    def get_embedding(self, query: str) -> tp.List[float]:
        return self.embedding.get_text_embedding(query)

    def get_embedding_batch(self, batch: tp.List[str]) -> tp.List[tp.List[float]]:
        return [embedding for embedding in self.embedding.get_text_embedding_batch(batch)]


class OllamaEmbeddings(Embeddings):
    """Embeddings class for Ollama.

    !!! info
        For default settings, see `chat.embeddings_configs.ollama` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): Ollama model identifier.

            Pulls the model if not already available locally.
        client_kwargs (KwargsLike): Keyword arguments for `ollama.Client`.
        embed_kwargs (KwargsLike): Keyword arguments for `ollama.Client.embed`.
        **kwargs: Keyword arguments for `Embeddings` or used as `client_kwargs` or `embed_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "ollama"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.ollama"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        embed_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            embed_kwargs=embed_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ollama")
        from ollama import Client

        ollama_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = ollama_config.pop("model", None)
        def_client_kwargs = ollama_config.pop("client_kwargs", None)
        def_embed_kwargs = ollama_config.pop("embed_kwargs", None)

        if model is None:
            model = def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = self.get_init_arg_names()
        for k in list(ollama_config.keys()):
            if k in init_arg_names:
                ollama_config.pop(k)

        client_arg_names = set(get_func_arg_names(Client.__init__))
        _client_kwargs = {}
        _embed_kwargs = {}
        for k, v in ollama_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _embed_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        embed_kwargs = merge_dicts(_embed_kwargs, def_embed_kwargs, embed_kwargs)

        client = Client(**client_kwargs)
        model_installed = False
        for installed_model in client.list().models:
            if installed_model.model == model:
                model_installed = True
                break
        if not model_installed:
            pbar = None
            status = None
            for response in client.pull(model, stream=True):
                if pbar is not None and status is not None and response.status != status:
                    pbar.refresh()
                    pbar.exit()
                    pbar = None
                    status = None
                if response.completed is not None:
                    status = response.status
                    if pbar is None:
                        pbar = ProgressBar(total=response.total, show_progress=self.show_progress, **self.pbar_kwargs)
                        pbar.enter()
                    pbar.set_prefix(status)
                    pbar.update_to(response.completed)

        self._model = model
        self._client = client
        self._embed_kwargs = embed_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> OllamaClientT:
        """Ollama client instance.

        Returns:
            Client: Ollama client instance.
        """
        return self._client

    @property
    def embed_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `ollama.Client.embed`.

        Returns:
            Kwargs: Keyword arguments for generating embeddings.
        """
        return self._embed_kwargs

    def get_embedding(self, query: str) -> tp.List[float]:
        response = self.client.embed(model=self.model, input=query, **self.embed_kwargs)
        return response["embeddings"][0]

    def get_embedding_batch(self, batch: tp.List[str]) -> tp.List[tp.List[float]]:
        response = self.client.embed(model=self.model, input=batch, **self.embed_kwargs)
        return response["embeddings"]


def resolve_embeddings(embeddings: tp.EmbeddingsLike = None) -> tp.MaybeType[Embeddings]:
    """Return a subclass or instance of `Embeddings` based on the provided identifier or object.

    !!! info
        For default settings, see `chat` in `vectorbtpro._settings.knowledge`.

    Args:
        embeddings (EmbeddingsLike): Identifier, subclass, or instance of `Embeddings`.

            Supported identifiers:

            * "openai" for `OpenAIEmbeddings`
            * "gemini" for `GeminiEmbeddings`
            * "hf_inference" for `HFInferenceEmbeddings`
            * "litellm" for `LiteLLMEmbeddings`
            * "llama_index" for `LlamaIndexEmbeddings`
            * "ollama" for `OllamaEmbeddings`
            * "auto" to select the first available option

            If None, configuration from `vectorbtpro._settings` is used.

    Returns:
        Embeddings: Resolved embeddings subclass or instance.
    """
    if embeddings is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["chat"]
        embeddings = chat_cfg["embeddings"]
    if isinstance(embeddings, str):
        if embeddings.lower() == "auto":
            import os
            from vectorbtpro.utils.module_ import check_installed

            if check_installed("openai") and os.getenv("OPENAI_API_KEY"):
                embeddings = "openai"
            elif check_installed("google.genai") and os.getenv("GEMINI_API_KEY"):
                embeddings = "gemini"
            elif check_installed("huggingface_hub") and os.getenv("HF_TOKEN"):
                embeddings = "hf_inference"
            elif check_installed("openai"):
                embeddings = "openai"
            elif check_installed("google.genai"):
                embeddings = "gemini"
            elif check_installed("huggingface_hub"):
                embeddings = "hf_inference"
            elif check_installed("litellm"):
                embeddings = "litellm"
            elif check_installed("llama_index"):
                embeddings = "llama_index"
            elif check_installed("ollama"):
                embeddings = "ollama"
            else:
                raise ValueError(
                    "No embeddings available. "
                    "Please install one of the supported packages: "
                    "openai, "
                    "google-genai, "
                    "huggingface-hub, "
                    "litellm, "
                    "llama-index, "
                    "ollama."
                )
        if embeddings.lower() == "anthropic":
            raise ValueError("Anthropic does not provide embeddings. Please use a different embeddings provider.")
        curr_module = sys.modules[__name__]
        found_embeddings = None
        for name, cls in inspect.getmembers(curr_module, inspect.isclass):
            if name.endswith("Embeddings"):
                _short_name = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == embeddings.lower():
                    found_embeddings = cls
                    break
        if found_embeddings is None:
            raise ValueError(f"Invalid embeddings: {embeddings!r}")
        embeddings = found_embeddings
    if isinstance(embeddings, type):
        checks.assert_subclass_of(embeddings, Embeddings, arg_name="embeddings")
    else:
        checks.assert_instance_of(embeddings, Embeddings, arg_name="embeddings")
    return embeddings


def embed(query: tp.MaybeList[str], embeddings: tp.EmbeddingsLike = None, **kwargs) -> tp.MaybeList[tp.List[float]]:
    """Return embedding(s) for one or more queries.

    Args:
        query (MaybeList[str]): Query string or a list of query strings to embed.
        embeddings (EmbeddingsLike): Identifier, subclass, or instance of `Embeddings`.

            Resolved using `resolve_embeddings`.
        **kwargs: Keyword arguments to initialize or update `embeddings`.

    Returns:
        MaybeList[List[float]]: Embedding vector(s) corresponding to the input query or queries.
    """
    embeddings = resolve_embeddings(embeddings=embeddings)
    if isinstance(embeddings, type):
        embeddings = embeddings(**kwargs)
    elif kwargs:
        embeddings = embeddings.replace(**kwargs)
    if isinstance(query, str):
        return embeddings.get_embedding(query)
    return embeddings.get_embeddings(query)
