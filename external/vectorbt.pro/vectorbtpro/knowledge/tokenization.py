# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing classes and utilities for tokenization."""

import inspect
import sys

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import Configured
from vectorbtpro.utils.decorators import memoized_method

if tp.TYPE_CHECKING:
    from tiktoken import Encoding as EncodingT
else:
    EncodingT = "tiktoken.Encoding"

__all__ = [
    "Tokenizer",
    "TikTokenizer",
    "tokenize",
    "detokenize",
]


class Tokenizer(Configured):
    """Abstract class for tokenizers.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and
        its sub-configurations `chat` and `chat.tokenizer_config`.

    Args:
        template_context (KwargsLike): Additional context for template substitution.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the class."""

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat", "knowledge.chat.tokenizer_config"]

    def __init__(self, template_context: tp.KwargsLike = None, **kwargs) -> None:
        Configured.__init__(self, template_context=template_context, **kwargs)

        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        self._template_context = template_context

    @property
    def template_context(self) -> tp.Kwargs:
        """Additional context for template substitution.

        Returns:
            Kwargs: Dictionary of context variables for template substitution.
        """
        return self._template_context

    def encode(self, text: str) -> tp.Tokens:
        """Return a list of tokens corresponding to the given text.

        !!! abstract
            This method should be overridden in a subclass.

        Args:
            text (str): Text to encode.

        Returns:
            list: List of tokens representing the input text.
        """
        raise NotImplementedError

    def decode(self, tokens: tp.Tokens) -> str:
        """Return the text obtained by decoding the given list of tokens.

        !!! abstract
            This method should be overridden in a subclass.

        Args:
            tokens (list): List of tokens to decode.

        Returns:
            str: Decoded text.
        """
        raise NotImplementedError

    @memoized_method
    def encode_single(self, text: str) -> tp.Token:
        """Return a single token encoded from the given text.

        Args:
            text (str): Text to encode.

        Returns:
            Token: Single token representing the input text.

        Raises:
            ValueError: If the text contains multiple tokens.
        """
        tokens = self.encode(text)
        if len(tokens) > 1:
            raise ValueError("Text contains multiple tokens")
        return tokens[0]

    @memoized_method
    def decode_single(self, token: tp.Token) -> str:
        """Return the text decoded from the provided single token.

        Args:
            token: Token to decode.

        Returns:
            str: Decoded text.
        """
        return self.decode([token])

    def count_tokens(self, text: str) -> int:
        """Return the total number of tokens in the provided text.

        Args:
            text (str): Text for token counting.

        Returns:
            int: Number of tokens.
        """
        return len(self.encode(text))

    def count_tokens_in_messages(self, messages: tp.ChatMessages) -> int:
        """Return the total number of tokens across the provided messages.

        !!! abstract
            This method should be overridden in a subclass.

        Args:
            messages (ChatMessages): List of dictionaries representing the conversation history.

        Returns:
            int: Total token count.
        """
        raise NotImplementedError


class TikTokenizer(Tokenizer):
    """Tokenizer class for tiktoken.

    Encoding can be a model name, an encoding name, or an encoding object for tokenization.

    !!! info
        For default settings, see `chat.tokenizer_configs.tiktoken` in `vectorbtpro._settings.knowledge`.

    Args:
        encoding (Union[None, str, Encoding]): Encoding specification as a model name,
            encoding name, or encoding object.
        model (Optional[str]): Model identifier used to determine the encoding.
        tokens_per_message (Optional[int]): Number of tokens charged per message.
        tokens_per_name (Optional[int]): Additional token count for message names.
        **kwargs: Keyword arguments for `Tokenizer`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "tiktoken"

    _settings_path: tp.SettingsPath = "knowledge.chat.tokenizer_configs.tiktoken"

    def __init__(
        self,
        encoding: tp.Union[None, str, EncodingT] = None,
        model: tp.Optional[str] = None,
        tokens_per_message: tp.Optional[int] = None,
        tokens_per_name: tp.Optional[int] = None,
        **kwargs,
    ) -> None:
        Tokenizer.__init__(
            self,
            encoding=encoding,
            model=model,
            tokens_per_message=tokens_per_message,
            tokens_per_name=tokens_per_name,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("tiktoken")
        from tiktoken import Encoding, get_encoding, encoding_for_model

        encoding = self.resolve_setting(encoding, "encoding")
        model = self.resolve_setting(model, "model")
        tokens_per_message = self.resolve_setting(tokens_per_message, "tokens_per_message")
        tokens_per_name = self.resolve_setting(tokens_per_name, "tokens_per_name")

        if isinstance(encoding, str):
            if encoding.startswith("model_or_"):
                try:
                    if model is None:
                        raise KeyError
                    encoding = encoding_for_model(model)
                except KeyError:
                    encoding = encoding[len("model_or_") :]
                    encoding = get_encoding(encoding) if "k_base" in encoding else encoding_for_model(encoding)
            elif isinstance(encoding, str):
                encoding = get_encoding(encoding) if "k_base" in encoding else encoding_for_model(encoding)
        checks.assert_instance_of(encoding, Encoding, arg_name="encoding")

        self._encoding = encoding
        self._tokens_per_message = tokens_per_message
        self._tokens_per_name = tokens_per_name

    @property
    def encoding(self) -> EncodingT:
        """Token encoding object used for tokenization.

        Returns:
            Encoding: Encoding object.
        """
        return self._encoding

    @property
    def tokens_per_message(self) -> int:
        """Token count charged per message.

        Returns:
            int: Number of tokens charged per message.
        """
        return self._tokens_per_message

    @property
    def tokens_per_name(self) -> int:
        """Additional token count for message names.

        Returns:
            int: Number of tokens charged for message names.
        """
        return self._tokens_per_name

    def encode(self, text: str) -> tp.Tokens:
        return self.encoding.encode(text)

    def decode(self, tokens: tp.Tokens) -> str:
        return self.encoding.decode(tokens)

    def count_tokens_in_messages(self, messages: tp.ChatMessages) -> int:
        num_tokens = 0
        for message in messages:
            num_tokens += self.tokens_per_message
            for key, value in message.items():
                num_tokens += self.count_tokens(value)
                if key == "name":
                    num_tokens += self.tokens_per_name
        num_tokens += 3
        return num_tokens


def resolve_tokenizer(tokenizer: tp.TokenizerLike = None) -> tp.MaybeType[Tokenizer]:
    """Resolve a `Tokenizer` subclass or instance.

    !!! info
        For default settings, see `chat` in `vectorbtpro._settings.knowledge`.

    Args:
        tokenizer (TokenizerLike): Identifier, subclass, or instance of `Tokenizer`.

            Supported identifiers:

            * "tiktoken" for `TikTokenizer`

    Returns:
        Tokenizer: Resolved tokenizer type or instance.
    """
    if tokenizer is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["chat"]
        tokenizer = chat_cfg["tokenizer"]
    if isinstance(tokenizer, str):
        curr_module = sys.modules[__name__]
        found_tokenizer = None
        for name, cls in inspect.getmembers(curr_module, inspect.isclass):
            if name.endswith("Tokenizer"):
                _short_name = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == tokenizer.lower():
                    found_tokenizer = cls
                    break
        if found_tokenizer is None:
            raise ValueError(f"Invalid tokenizer: {tokenizer!r}")
        tokenizer = found_tokenizer
    if isinstance(tokenizer, type):
        checks.assert_subclass_of(tokenizer, Tokenizer, arg_name="tokenizer")
    else:
        checks.assert_instance_of(tokenizer, Tokenizer, arg_name="tokenizer")
    return tokenizer


def tokenize(text: str, tokenizer: tp.TokenizerLike = None, **kwargs) -> tp.Tokens:
    """Tokenize text using a resolved `Tokenizer`.

    Args:
        text (str): Text to tokenize.
        tokenizer (TokenizerLike): Identifier, subclass, or instance of `Tokenizer`.

            Resolved using `resolve_tokenizer`.
        **kwargs: Keyword arguments to initialize or update `tokenizer`.

    Returns:
        Tokens: List of tokens representing the input text.
    """
    tokenizer = resolve_tokenizer(tokenizer=tokenizer)
    if isinstance(tokenizer, type):
        tokenizer = tokenizer(**kwargs)
    elif kwargs:
        tokenizer = tokenizer.replace(**kwargs)
    return tokenizer.encode(text)


def detokenize(tokens: tp.Tokens, tokenizer: tp.TokenizerLike = None, **kwargs) -> str:
    """Detokenize tokens into text using a resolved `Tokenizer`.

    Args:
        tokens (Tokens): List of tokens to decode.
        tokenizer (TokenizerLike): Identifier, subclass, or instance of `Tokenizer`.

            Resolved using `resolve_tokenizer`.
        **kwargs: Keyword arguments to initialize or update `tokenizer`.

    Returns:
        str: Decoded text.
    """
    tokenizer = resolve_tokenizer(tokenizer=tokenizer)
    if isinstance(tokenizer, type):
        tokenizer = tokenizer(**kwargs)
    elif kwargs:
        tokenizer = tokenizer.replace(**kwargs)
    return tokenizer.decode(tokens)
