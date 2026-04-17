# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing classes and utilities for splitting documents."""

import ast
import inspect
import re
import sys

from vectorbtpro import _typing as tp
from vectorbtpro.knowledge.tokenization import Tokenizer, resolve_tokenizer
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts, flat_merge_dicts, Configured
from vectorbtpro.utils.parsing import get_func_arg_names, get_func_kwargs
from vectorbtpro.utils.template import CustomTemplate, SafeSub, RepFunc

if tp.TYPE_CHECKING:
    from llama_index.core.node_parser import NodeParser as NodeParserT
else:
    NodeParserT = "llama_index.core.node_parser.NodeParser"

__all__ = [
    "TextSplitter",
    "TokenSplitter",
    "SegmentSplitter",
    "SourceSplitter",
    "PythonSplitter",
    "MarkdownSplitter",
    "LlamaIndexSplitter",
    "split_text",
]


class TextSplitter(Configured):
    """Abstract class for text splitters.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and
        its sub-configurations `chat` and `chat.text_splitter_config`.

    Args:
        chunk_template (Optional[CustomTemplateLike]): Template used to format each text chunk.
        template_context (KwargsLike): Additional context for template substitution.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the text splitter class."""

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat", "knowledge.chat.text_splitter_config"]

    def __init__(
        self,
        chunk_template: tp.Optional[tp.CustomTemplateLike] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            chunk_template=chunk_template,
            template_context=template_context,
            **kwargs,
        )

        chunk_template = self.resolve_setting(chunk_template, "chunk_template")
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        self._chunk_template = chunk_template
        self._template_context = template_context

    @property
    def chunk_template(self) -> tp.Kwargs:
        """Template used for formatting text chunks.

        Can use the following context: `chunk_idx`, `chunk_start`, `chunk_end`, `chunk_text`, and `text`.

        The template can be a string, a function, or an instance of `vectorbtpro.utils.template.CustomTemplate`.

        Returns:
            Kwargs: Context mapping used for expression evaluation.
        """
        return self._chunk_template

    @property
    def template_context(self) -> tp.Kwargs:
        """Additional context for template substitution.

        Returns:
            Kwargs: Dictionary of context variables for template substitution.
        """
        return self._template_context

    def split(self, text: str) -> tp.TSSpanChunks:
        """Yield the start and end character indices for each text chunk in the given text.

        !!! abstract
            This method should be overridden in a subclass.

        Args:
            text (str): Input text to split.

        Yields:
            Tuple[int, int]: Tuple representing the start and end indices of a text chunk.
        """
        raise NotImplementedError

    def split_text(self, text: str) -> tp.TSTextChunks:
        """Yield formatted text chunks generated from the input text by applying the chunk template.

        The method substitutes the chunk template with context derived from each chunk's position and text.

        Args:
            text (str): Text to split.

        Yields:
            str: Formatted text chunk.
        """
        for chunk_idx, (chunk_start, chunk_end) in enumerate(self.split(text)):
            chunk_text = text[chunk_start:chunk_end]
            chunk_template = self.chunk_template
            if isinstance(chunk_template, str):
                chunk_template = SafeSub(chunk_template)
            elif checks.is_function(chunk_template):
                chunk_template = RepFunc(chunk_template)
            elif not isinstance(chunk_template, CustomTemplate):
                raise TypeError("Chunk template must be a string, function, or template")
            template_context = flat_merge_dicts(
                dict(
                    chunk_idx=chunk_idx,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    chunk_text=chunk_text,
                    text=text,
                ),
                self.template_context,
            )
            yield chunk_template.substitute(template_context, eval_id="chunk_template")


class TokenSplitter(TextSplitter):
    """Splitter class for tokens.

    !!! info
        For default settings, see `chat.text_splitter_configs.token` in `vectorbtpro._settings.knowledge`.

    Args:
        chunk_size (Optional[int]): Maximum number of tokens per chunk; None if disabled.
        chunk_overlap (Union[None, int, float]): Number or fraction of tokens
            overlapping between consecutive chunks.
        tokenizer (TokenizerLike): Identifier, subclass, or instance of
            `vectorbtpro.knowledge.tokenization.Tokenizer`.

            Resolved using `vectorbtpro.knowledge.tokenization.resolve_tokenizer`.
        tokenizer_kwargs (KwargsLike): Keyword arguments to initialize or update `tokenizer`.
        **kwargs: Keyword arguments for `TextSplitter`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "token"

    _settings_path: tp.SettingsPath = "knowledge.chat.text_splitter_configs.token"

    def __init__(
        self,
        chunk_size: tp.Optional[int] = None,
        chunk_overlap: tp.Union[None, int, float] = None,
        tokenizer: tp.TokenizerLike = None,
        tokenizer_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        TextSplitter.__init__(
            self,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            **kwargs,
        )

        chunk_size = self.resolve_setting(chunk_size, "chunk_size")
        chunk_overlap = self.resolve_setting(chunk_overlap, "chunk_overlap")
        tokenizer = self.resolve_setting(tokenizer, "tokenizer", default=None)
        tokenizer_kwargs = self.resolve_setting(tokenizer_kwargs, "tokenizer_kwargs", default=None, merge=True)

        tokenizer = resolve_tokenizer(tokenizer)
        if isinstance(tokenizer, type):
            tokenizer_kwargs = dict(tokenizer_kwargs)
            tokenizer_kwargs["template_context"] = merge_dicts(
                self.template_context, tokenizer_kwargs.get("template_context", None)
            )
            tokenizer = tokenizer(**tokenizer_kwargs)
        elif tokenizer_kwargs:
            tokenizer = tokenizer.replace(**tokenizer_kwargs)
        if chunk_size is not None:
            if checks.is_float(chunk_overlap):
                if 0 <= abs(chunk_overlap) <= 1:
                    chunk_overlap = chunk_overlap * chunk_size
                elif not chunk_overlap.is_integer():
                    raise ValueError("Floating number for chunk_overlap must be between 0 and 1")
                chunk_overlap = int(chunk_overlap)
            if chunk_overlap >= chunk_size:
                raise ValueError("Chunk overlap must be less than the chunk size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._tokenizer = tokenizer

    @property
    def chunk_size(self) -> tp.Optional[int]:
        """Maximum number of tokens per chunk.

        Returns:
            int: Maximum number of tokens allowed in each chunk; None if disabled.
        """
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Number of overlapping tokens between chunks.

        If specified as a float between 0 and 1, it is scaled by `TokenSplitter.chunk_size`.

        Returns:
            int: Number of overlapping tokens between chunks.
        """
        return self._chunk_overlap

    @property
    def tokenizer(self) -> Tokenizer:
        """`vectorbtpro.knowledge.tokenization.Tokenizer` instance used to tokenize input text.

        Returns:
            Tokenizer: Tokenizer instance used for encoding and decoding.
        """
        return self._tokenizer

    def split_into_tokens(self, text: str) -> tp.TSSpanChunks:
        """Yield start and end indices for each token in the given text.

        The method encodes the text into tokens and decodes each token to determine its character span.

        Args:
            text (str): Text to tokenize.

        Yields:
            Tuple[int, int]: Start and end indices of each token.
        """
        tokens = self.tokenizer.encode(text)
        last_end = 0
        for token in tokens:
            _text = self.tokenizer.decode_single(token)
            start = last_end
            end = start + len(_text)
            yield start, end
            last_end = end

    def split(self, text: str) -> tp.TSSpanChunks:
        if self.chunk_size is None:
            yield from self.split_into_tokens(text)

        tokens = list(self.split_into_tokens(text))
        total_tokens = len(tokens)
        if not tokens:
            return

        token_count = 0
        while token_count < total_tokens:
            chunk_tokens = tokens[token_count : token_count + self.chunk_size]
            chunk_start = chunk_tokens[0][0]
            chunk_end = chunk_tokens[-1][1]
            yield chunk_start, chunk_end

            if token_count + self.chunk_size >= total_tokens:
                break
            token_count += self.chunk_size - self.chunk_overlap


class SegmentSplitter(TokenSplitter):
    """Splitter class for segments based on specified separators.

    This class iteratively splits text by applying nested layers of separators.
    If a segment exceeds the allowed size and no valid previous chunk exists or the token
    count falls below the minimum, the next layer of separators is used. To split into tokens,
    set a separator to None; to split into individual characters, use an empty string.

    !!! info
        For default settings, see `chat.text_splitter_configs.segment` in `vectorbtpro._settings.knowledge`.

    Args:
        separators (List[List[Optional[str]]]): Nested list of separators grouped by layers used
            for splitting text.
        min_chunk_size (Union[int, float]): Minimum number of tokens required per chunk.

            If provided as a float between 0 and 1, it is interpreted relative to the chunk size.
        fixed_overlap (bool): Indicates whether fixed overlap is applied.
        **kwargs: Keyword arguments for `TokenSplitter`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "segment"

    _settings_path: tp.SettingsPath = "knowledge.chat.text_splitter_configs.segment"

    def __init__(
        self,
        separators: tp.MaybeList[tp.MaybeList[tp.Optional[str]]] = None,
        min_chunk_size: tp.Union[None, int, float] = None,
        fixed_overlap: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        TokenSplitter.__init__(
            self,
            separators=separators,
            min_chunk_size=min_chunk_size,
            fixed_overlap=fixed_overlap,
            **kwargs,
        )

        separators = self.resolve_setting(separators, "separators")
        min_chunk_size = self.resolve_setting(min_chunk_size, "min_chunk_size")
        fixed_overlap = self.resolve_setting(fixed_overlap, "fixed_overlap")

        if not isinstance(separators, list):
            separators = [separators]
        else:
            separators = list(separators)
        for layer in range(len(separators)):
            if not isinstance(separators[layer], list):
                separators[layer] = [separators[layer]]
            else:
                separators[layer] = list(separators[layer])
        if self.chunk_size is not None:
            if checks.is_float(min_chunk_size):
                if 0 <= abs(min_chunk_size) <= 1:
                    min_chunk_size = min_chunk_size * self.chunk_size
                elif not min_chunk_size.is_integer():
                    raise ValueError("Floating number for min_chunk_size must be between 0 and 1")
                min_chunk_size = int(min_chunk_size)

        self._separators = separators
        self._min_chunk_size = min_chunk_size
        self._fixed_overlap = fixed_overlap

    @property
    def separators(self) -> tp.List[tp.List[tp.Optional[str]]]:
        """Nested list of separators grouped by layers.

        Returns:
            List[List[Optional[str]]]: (Nested) list of separators used for splitting text.
        """
        return self._separators

    @property
    def min_chunk_size(self) -> int:
        """Minimum number of tokens per chunk. If provided as a float, it is interpreted relative to
        `SegmentSplitter.chunk_size`.

        Returns:
            int: Minimum number of tokens required per chunk.
        """
        return self._min_chunk_size

    @property
    def fixed_overlap(self) -> bool:
        """Whether fixed overlap is applied.

        Returns:
            bool: True if fixed overlap is applied, False otherwise.
        """
        return self._fixed_overlap

    def split_into_segments(self, text: str, separator: tp.Optional[str] = None) -> tp.TSSegmentChunks:
        """Split text into segments using the provided separator.

        If `separator` is None, split the text into tokens using `SegmentSplitter.split_into_tokens`.
        If `separator` is an empty string, split the text into individual characters; otherwise,
        split the text at each occurrence of `separator`.

        Args:
            text (str): Text to be split.
            separator (Optional[str]): Separator to insert between data items.

        Yields:
            Tuple[int, int, bool]: Tuple containing the segment's start index, end index, and
                a flag indicating if the segment is a separator.
        """
        if not separator:
            if separator is None:
                for start, end in self.split_into_tokens(text):
                    yield start, end, False
            else:
                for i in range(len(text)):
                    yield i, i + 1, False
        else:
            last_end = 0

            for match in re.finditer(separator, text):
                start, end = match.span()
                if start > last_end:
                    _text = text[last_end:start]
                    yield last_end, start, False

                _text = text[start:end]
                yield start, end, True
                last_end = end

            if last_end < len(text):
                _text = text[last_end:]
                yield last_end, len(text), False

    def split(self, text: str) -> tp.TSSpanChunks:
        if not text:
            yield 0, 0
            return None
        if self.chunk_size is None:
            yield 0, len(text)
            return None
        total_tokens = self.tokenizer.count_tokens(text)
        if total_tokens <= self.chunk_size:
            yield 0, len(text)
            return None

        layer = 0
        chunk_start = 0
        chunk_continue = 0
        chunk_tokens = []
        stable_token_count = 0
        stable_char_count = 0
        remaining_text = text
        overlap_segments = []
        token_offset_map = {}

        while remaining_text:
            if layer == 0:
                if chunk_continue:
                    curr_start = chunk_continue
                else:
                    curr_start = chunk_start
                curr_text = remaining_text
                curr_segments = list(overlap_segments)
                curr_tokens = list(chunk_tokens)
                curr_stable_token_count = stable_token_count
                curr_stable_char_count = stable_char_count
                sep_curr_segments = None
                sep_curr_tokens = None
                sep_curr_stable_token_count = None
                sep_curr_stable_char_count = None

            for separator in self.separators[layer]:
                segments = self.split_into_segments(curr_text, separator=separator)
                curr_text = ""
                finished = False

                for segment in segments:
                    segment_start = curr_start + segment[0]
                    segment_end = curr_start + segment[1]
                    segment_is_separator = segment[2]

                    if not curr_tokens:
                        segment_text = text[segment_start:segment_end]
                        new_curr_tokens = self.tokenizer.encode(segment_text)
                        new_curr_stable_token_count = 0
                        new_curr_stable_char_count = 0
                    elif not curr_stable_token_count:
                        chunk_text = text[chunk_start:segment_end]
                        new_curr_tokens = self.tokenizer.encode(chunk_text)
                        new_curr_stable_token_count = 0
                        new_curr_stable_char_count = 0
                        min_token_count = min(len(curr_tokens), len(new_curr_tokens))
                        for i in range(min_token_count):
                            if curr_tokens[i] == new_curr_tokens[i]:
                                new_curr_stable_token_count += 1
                                new_curr_stable_char_count += len(self.tokenizer.decode_single(curr_tokens[i]))
                            else:
                                break
                    else:
                        stable_tokens = curr_tokens[:curr_stable_token_count]
                        unstable_start = chunk_start + curr_stable_char_count
                        partial_text = text[unstable_start:segment_end]
                        partial_tokens = self.tokenizer.encode(partial_text)
                        new_curr_tokens = stable_tokens + partial_tokens
                        new_curr_stable_token_count = curr_stable_token_count
                        new_curr_stable_char_count = curr_stable_char_count
                        min_token_count = min(len(curr_tokens), len(new_curr_tokens))
                        for i in range(curr_stable_token_count, min_token_count):
                            if curr_tokens[i] == new_curr_tokens[i]:
                                new_curr_stable_token_count += 1
                                new_curr_stable_char_count += len(self.tokenizer.decode_single(curr_tokens[i]))
                            else:
                                break

                    if len(new_curr_tokens) > self.chunk_size:
                        if segment_is_separator:
                            if (
                                sep_curr_segments
                                and len(sep_curr_tokens) >= self.min_chunk_size
                                and not (self.chunk_overlap and len(sep_curr_tokens) <= self.chunk_overlap)
                            ):
                                curr_segments = list(sep_curr_segments)
                                curr_tokens = list(sep_curr_tokens)
                                curr_stable_token_count = sep_curr_stable_token_count
                                curr_stable_char_count = sep_curr_stable_char_count
                                segment_start = curr_segments[-1][0]
                                segment_end = curr_segments[-1][1]
                        curr_text = text[segment_start:segment_end]
                        curr_start = segment_start
                        finished = False
                        break
                    else:
                        curr_segments.append((segment_start, segment_end, segment_is_separator))
                        token_offset_map[segment_start] = len(curr_tokens)
                        curr_tokens = new_curr_tokens
                        curr_stable_token_count = new_curr_stable_token_count
                        curr_stable_char_count = new_curr_stable_char_count
                        if segment_is_separator:
                            sep_curr_segments = list(curr_segments)
                            sep_curr_tokens = list(curr_tokens)
                            sep_curr_stable_token_count = curr_stable_token_count
                            sep_curr_stable_char_count = curr_stable_char_count
                        finished = True

                if finished:
                    break

            if (
                curr_segments
                and len(curr_tokens) >= self.min_chunk_size
                and not (self.chunk_overlap and len(curr_tokens) <= self.chunk_overlap)
            ):
                chunk_start = curr_segments[0][0]
                chunk_end = curr_segments[-1][1]
                yield chunk_start, chunk_end

                if chunk_end == len(text):
                    break
                if self.chunk_overlap:
                    fixed_overlap = True
                    if not self.fixed_overlap:
                        for segment in curr_segments:
                            if not segment[2]:
                                token_offset = token_offset_map[segment[0]]
                                if token_offset > curr_stable_token_count:
                                    break
                                if len(curr_tokens) - token_offset <= self.chunk_overlap:
                                    chunk_tokens = curr_tokens[token_offset:]
                                    new_chunk_start = segment[0]
                                    chunk_offset = new_chunk_start - chunk_start
                                    chunk_start = new_chunk_start
                                    chunk_continue = chunk_end
                                    fixed_overlap = False
                                    break
                    if fixed_overlap:
                        chunk_tokens = curr_tokens[-self.chunk_overlap :]
                        token_offset = len(curr_tokens) - len(chunk_tokens)
                        new_chunk_start = chunk_end - len(self.tokenizer.decode(chunk_tokens))
                        chunk_offset = new_chunk_start - chunk_start
                        chunk_start = new_chunk_start
                        chunk_continue = chunk_end
                    stable_token_count = max(0, curr_stable_token_count - token_offset)
                    stable_char_count = max(0, curr_stable_char_count - chunk_offset)
                    overlap_segments = [(chunk_start, chunk_end, False)]
                    token_offset_map[chunk_start] = 0
                else:
                    chunk_tokens = []
                    chunk_start = chunk_end
                    chunk_continue = 0
                    stable_token_count = 0
                    stable_char_count = 0
                    overlap_segments = []
                    token_offset_map = {}

                if chunk_continue:
                    remaining_text = text[chunk_continue:]
                else:
                    remaining_text = text[chunk_start:]
                layer = 0
            else:
                layer += 1
                if layer == len(self.separators):
                    if curr_segments and curr_segments[-1][1] == len(text):
                        chunk_start = curr_segments[0][0]
                        chunk_end = curr_segments[-1][1]
                        yield chunk_start, chunk_end
                        break
                    remaining_tokens = self.tokenizer.encode(remaining_text)
                    if len(remaining_tokens) > self.chunk_size:
                        raise ValueError(
                            "Total number of tokens in the last chunk is greater than the chunk size. "
                            "Increase chunk_size or the separator granularity."
                        )
                    yield curr_start, len(text)
                    break


class SourceSplitter(TokenSplitter):
    """Splitter class for source code.

    This class is used to split source code into chunks by parsing the structure of the code.
    It divides nodes of the code into levels and performs splitting based on the specified chunk size and overlap.

    !!! info
        For default settings, see `chat.text_splitter_configs.source` in `vectorbtpro._settings.knowledge`.

    Args:
        uniform_chunks (Optional[bool]): Whether each chunk should start and end at the same base level.

            If nested chunks (with level > base) are present, includes them only if they fit as a whole.
        **kwargs: Keyword arguments for `TokenSplitter`.
    """

    _settings_path: tp.SettingsPath = "knowledge.chat.text_splitter_configs.source"

    def __init__(
        self,
        uniform_chunks: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        TokenSplitter.__init__(
            self,
            uniform_chunks=uniform_chunks,
            **kwargs,
        )

        uniform_chunks = self.resolve_setting(uniform_chunks, "uniform_chunks")

        self._uniform_chunks = uniform_chunks

    @property
    def uniform_chunks(self) -> bool:
        """Whether each chunk should start and end at the same base level.

        If nested chunks (with level > base) are present, includes them only if they fit as a whole.

        Returns:
            bool: True if uniform chunks are enabled, False otherwise.
        """
        return self._uniform_chunks

    def split_source(self, source: str) -> tp.TSSourceChunks:
        """Split the source code into chunks.

        !!! abstract
            This method should be overridden in a subclass.

        Args:
            source (str): Source code to be split.

        Yields:
            Tuple[str, int]: Tuple containing the source code chunk and its base level.
        """
        raise NotImplementedError

    def split_text(self, text: str, debug: bool = False) -> tp.TSTextChunks:
        source_nodes = list(self.split_source(text))

        if self.chunk_size is None:
            for code, _ in source_nodes:
                yield code
            return

        count_tokens = self.tokenizer.count_tokens
        max_chunk_tokens = self.chunk_size
        max_overlap_tokens = self.chunk_overlap

        total_nodes = len(source_nodes)
        current_node_index = 0
        last_overlap_start_idx = last_overlap_end_idx = None

        def _last_index_non_uniform(start_idx):
            used_tokens = 0
            idx = start_idx
            while idx < total_nodes:
                node_text = source_nodes[idx][0]
                node_tokens = count_tokens(node_text)
                if node_tokens > max_chunk_tokens:
                    return idx if idx == start_idx else idx - 1
                if used_tokens + node_tokens > max_chunk_tokens:
                    return idx - 1
                used_tokens += node_tokens
                idx += 1
            return idx - 1

        def _last_index_uniform(start_idx):
            base_level = source_nodes[start_idx][1]
            used_tokens = 0
            last_base_idx = start_idx - 1
            idx = start_idx
            while idx < total_nodes and source_nodes[idx][1] >= base_level:
                node_text, node_level = source_nodes[idx]
                node_tokens = count_tokens(node_text)
                if node_tokens > max_chunk_tokens:
                    return last_base_idx if last_base_idx >= start_idx else idx
                if used_tokens + node_tokens > max_chunk_tokens:
                    return last_base_idx
                used_tokens += node_tokens
                if node_level == base_level:
                    last_base_idx = idx
                idx += 1
                if idx == total_nodes or source_nodes[idx][1] < base_level:
                    return last_base_idx
            return idx - 1

        while current_node_index < total_nodes:
            chunk_end_idx = (
                _last_index_uniform(current_node_index)
                if self.uniform_chunks
                else _last_index_non_uniform(current_node_index)
            )

            if (
                last_overlap_start_idx is not None
                and current_node_index == last_overlap_start_idx
                and chunk_end_idx == last_overlap_end_idx
            ):
                current_node_index = chunk_end_idx + 1
                last_overlap_start_idx = last_overlap_end_idx = None
                continue

            node_slice = source_nodes[current_node_index : chunk_end_idx + 1]
            chunk_text = "".join(code for code, _ in node_slice)

            if debug:
                print("=" * 20, count_tokens(chunk_text), "=" * 20)
                for code, level in node_slice:
                    print("-" * 10, level, count_tokens(code), "-" * 10)
                    print(code, end="")

            yield chunk_text

            if max_overlap_tokens > 0:
                overlap_tokens = 0
                overlap_start_idx = chunk_end_idx
                while overlap_start_idx >= current_node_index:
                    node_tokens = count_tokens(source_nodes[overlap_start_idx][0])
                    if overlap_tokens + node_tokens > max_overlap_tokens:
                        break
                    overlap_tokens += node_tokens
                    overlap_start_idx -= 1
                overlap_start_idx += 1

                if chunk_end_idx - overlap_start_idx >= 1:
                    last_overlap_start_idx = overlap_start_idx
                    last_overlap_end_idx = chunk_end_idx
                    current_node_index = overlap_start_idx
                else:
                    last_overlap_start_idx = last_overlap_end_idx = None
                    current_node_index = chunk_end_idx + 1
            else:
                current_node_index = chunk_end_idx + 1


class PythonSplitter(SourceSplitter):
    """Splitter class for Python source code.

    This class is used to split Python source code using the `ast` module. All module-level statements
    become the zero level, which can be split into nested levels. The class supports splitting
    statements based on a whitelist and blacklist of statement types. It also allows for limiting
    the maximum statement level.

    !!! info
        For default settings, see `chat.text_splitter_configs.python` in `vectorbtpro._settings.knowledge`.

    Args:
        stmt_whitelist (Optional[Iterable[str]]): Statement types to include in the split.

            Effective only if `max_stmt_level` is met.
        stmt_blacklist (Optional[Iterable[str]]): Statement types to exclude from the split.
        max_stmt_level (Optional[int]): Maximum level of statements to include in the split.

            If None, all levels are included.
        **kwargs: Keyword arguments for `SourceSplitter`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "python"

    _settings_path: tp.SettingsPath = "knowledge.chat.text_splitter_configs.python"

    def __init__(
        self,
        stmt_whitelist: tp.Optional[tp.Iterable[str]] = None,
        stmt_blacklist: tp.Optional[tp.Iterable[str]] = None,
        max_stmt_level: tp.Optional[int] = None,
        **kwargs,
    ) -> None:
        SourceSplitter.__init__(
            self,
            stmt_whitelist=stmt_whitelist,
            stmt_blacklist=stmt_blacklist,
            max_stmt_level=max_stmt_level,
            **kwargs,
        )

        stmt_whitelist = self.resolve_setting(stmt_whitelist, "stmt_whitelist")
        stmt_blacklist = self.resolve_setting(stmt_blacklist, "stmt_blacklist")
        max_stmt_level = self.resolve_setting(max_stmt_level, "max_stmt_level")

        self._stmt_whitelist = tuple(stmt_whitelist or ())
        self._stmt_blacklist = tuple(stmt_blacklist or ())
        self._max_stmt_level = max_stmt_level

    @property
    def stmt_whitelist(self) -> tp.Tuple[str, ...]:
        """Statement types to include in the split.

        Effective only if `max_stmt_level` is met.

        Returns:
            Tuple[str, ...]: Tuple of statement types.
        """
        return self._stmt_whitelist

    @property
    def stmt_blacklist(self) -> tp.Tuple[str, ...]:
        """Statement types to exclude from the split.

        Returns:
            Tuple[str, ...]: Tuple of statement types.
        """
        return self._stmt_blacklist

    @property
    def max_stmt_level(self) -> tp.Optional[int]:
        """Maximum level of statements to include in the split.

        Returns:
            Optional[int]: Maximum statement level; None if all levels are included.
        """
        return self._max_stmt_level

    def should_split_stmt(self, stmt: ast.stmt, level: int) -> bool:
        """Check if the statement should be split based on its type and level.

        Args:
            stmt (ast.stmt): Statement to check.
            level (int): Level of the statement.

        Returns:
            bool: True if the statement should be split, False otherwise.
        """
        if self.max_stmt_level is not None and level >= self.max_stmt_level:
            return False
        if self.stmt_blacklist and checks.is_instance_of(stmt, self.stmt_blacklist):
            return False
        if self.stmt_whitelist and not checks.is_instance_of(stmt, self.stmt_whitelist):
            return False
        return True

    def split_source(self, source: str) -> tp.TSSourceChunks:
        lines = source.splitlines(keepends=True)
        tree = ast.parse(source, type_comments=True)

        def _stmt_span(node):
            start = min((d.lineno for d in getattr(node, "decorator_list", ())), default=node.lineno)
            end = getattr(node, "end_lineno", node.lineno)
            return start, end

        def _header_end(first_line):
            for idx in range(first_line - 1, len(lines)):
                code = lines[idx].split("#", 1)[0].rstrip()
                if code and code.endswith(":"):
                    return idx + 1
            return first_line

        def _split_block(body, start_line, end_line, level):
            body = list(body)
            if not body:
                yield (start_line, end_line, level)
                return

            cursor = start_line
            i = 0
            n = len(body)

            while i < n:
                stmt = body[i]
                stmt_start, stmt_end = _stmt_span(stmt)
                if stmt_start > cursor:
                    yield (cursor, stmt_start - 1, level)
                if (
                    isinstance(stmt, (ast.Assign, ast.AnnAssign))
                    and i + 1 < n
                    and isinstance(body[i + 1], ast.Expr)
                    and isinstance(body[i + 1].value, ast.Constant)
                    and isinstance(body[i + 1].value.value, str)
                ):
                    _, next_end = _stmt_span(body[i + 1])
                    stmt_end = next_end
                    i += 1
                if self.should_split_stmt(stmt, level) and isinstance(
                    stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    yield from _split_node(stmt, level + 1)
                else:
                    yield (stmt_start, stmt_end, level)
                cursor = stmt_end + 1
                i += 1
            if cursor <= end_line:
                yield (cursor, end_line, level)

        def _split_node(node, level):
            start, end = _stmt_span(node)
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                hdr_end = getattr(node.body[0], "end_lineno")
                body_stmts = node.body[1:]
            else:
                hdr_end = _header_end(start)
                body_stmts = node.body

            yield (start, hdr_end, level)
            yield from _split_block(body_stmts, hdr_end + 1, end, level)

        for s, e, lvl in _split_block(tree.body, 1, len(lines), 0):
            yield ("".join(lines[s - 1 : e]), lvl)


class MarkdownSplitter(SourceSplitter):
    """Splitter class for Markdown source code.

    This class is responsible for splitting Markdown source code into chunks
    based on headers and paragraphs. It uses a custom algorithm to identify headers
    and split the content accordingly.

    !!! info
        For default settings, see `chat.text_splitter_configs.markdown` in `vectorbtpro._settings.knowledge`.

    Args:
        split_by (Optional[str]): Method to split the source code.

            Options are "header" or "paragraph".
        max_section_level (Optional[int]): Maximum level of sections to include in the split.

            If None, all levels are included.
        **kwargs: Keyword arguments for `SourceSplitter`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "markdown"

    _settings_path: tp.SettingsPath = "knowledge.chat.text_splitter_configs.markdown"

    def __init__(
        self,
        split_by: tp.Optional[str] = None,
        max_section_level: tp.Optional[int] = None,
        **kwargs,
    ) -> None:
        SourceSplitter.__init__(
            self,
            split_by=split_by,
            max_section_level=max_section_level,
            **kwargs,
        )

        split_by = self.resolve_setting(split_by, "split_by")
        max_section_level = self.resolve_setting(max_section_level, "max_section_level")

        self._split_by = split_by
        self._max_section_level = max_section_level

    @property
    def split_by(self) -> str:
        """Method to split the source code.

        Options are "header" or "paragraph".

        Returns:
            str: Method used to split the source code.
        """
        return self._split_by

    @property
    def max_section_level(self) -> tp.Optional[int]:
        """Maximum level of sections to include in the split.

        Returns:
            Optional[int]: Maximum section level; None if all levels are included.
        """
        return self._max_section_level

    def should_split_section(self, section: str, level: int) -> bool:
        """Determine whether to split the given section.

        Args:
            section: Section to evaluate.
            level: Current level of the section.

        Returns:
            bool: True if the section should be split; False otherwise.
        """
        if self.max_section_level is not None and level >= self.max_section_level:
            return False
        return True

    def split_source(self, source: str) -> tp.TSSourceChunks:
        lines = source.splitlines(True)

        chunks, buf = [], []
        level = 0
        header_pending = para_started = False
        in_code = in_html = False
        fence = html_tag = ""
        code_closed = html_closed = False
        i, n = 0, len(lines)

        def _is_header(txt):
            return txt.lstrip().startswith("#")

        while i < n:
            line = lines[i]
            stripped = line.rstrip("\n")
            lstripped = stripped.lstrip()

            if in_code:
                buf.append(line)
                if re.match(r"\s*" + re.escape(fence), lstripped):
                    in_code = False
                    chunks.append(("".join(buf), level))
                    buf.clear()
                    code_closed = True
                i += 1
                continue

            if in_html:
                buf.append(line)
                if re.search(r"</" + html_tag + r"\s*>", lstripped, re.I):
                    in_html = False
                    chunks.append(("".join(buf), level))
                    buf.clear()
                    html_closed = True
                i += 1
                continue

            if re.match(r"\s*<div\b", lstripped, re.I):
                if header_pending and not para_started:
                    chunks.append(("".join(buf), level))
                    buf.clear()
                    header_pending = False
                buf.append(line)
                html_tag = "div"
                if re.search(r"</div\s*>", lstripped, re.I):
                    chunks.append(("".join(buf), level))
                    buf.clear()
                    html_closed = True
                else:
                    in_html = True
                i += 1
                continue

            if lstripped.startswith("```") or lstripped.startswith("~~~"):
                if header_pending and not para_started:
                    chunks.append(("".join(buf), level))
                    buf.clear()
                    header_pending = False
                buf.append(line)
                fence, in_code = lstripped[:3], True
                i += 1
                continue

            if _is_header(lstripped):
                if buf:
                    chunks.append(("".join(buf), level))
                    buf.clear()
                level = len(lstripped.split(" ")[0])
                buf.append(line)
                header_pending = True
                para_started = False
                i += 1
                continue

            if not lstripped:
                blanks = []
                while i < n and not lines[i].strip():
                    blanks.append(lines[i])
                    i += 1
                if code_closed or html_closed:
                    t, lvl = chunks[-1]
                    chunks[-1] = (t + "".join(blanks), lvl)
                    code_closed = html_closed = False
                    continue
                if i == n:
                    buf.extend(blanks)
                    break
                next_line = lines[i]
                prev_idx = i - 1
                while prev_idx >= 0 and not lines[prev_idx].strip():
                    prev_idx -= 1
                prev_line = lines[prev_idx] if prev_idx >= 0 else ""
                if (
                    len(prev_line) - len(prev_line.lstrip(" ")) >= 4
                    and len(next_line) - len(next_line.lstrip(" ")) >= 4
                ):
                    buf.extend(blanks)
                    continue
                buf.extend(blanks)
                if not (header_pending and not para_started):
                    chunks.append(("".join(buf), level))
                    buf.clear()
                    header_pending = para_started = False
                continue

            if header_pending and not para_started:
                para_started = True
            buf.append(line)
            code_closed = html_closed = False
            i += 1

        if buf:
            chunks.append(("".join(buf), level))

        split_by = self.split_by.lower()
        if split_by == "paragraph":
            final_chunks = []
            header_flag = [_is_header(c[0]) for c in chunks]
            i, m = 0, len(chunks)

            while i < m:
                text, lvl = chunks[i]
                if not header_flag[i]:
                    final_chunks.append((text, lvl))
                    i += 1
                    continue
                j = i + 1
                while j < m and not (header_flag[j] and chunks[j][1] <= lvl):
                    j += 1
                section_text = "".join(c[0] for c in chunks[i:j])
                if self.should_split_section(section_text, lvl):
                    final_chunks.append((text, lvl))
                    i += 1
                    continue
                final_chunks.append((text, lvl))
                k = i + 1
                while k < j:
                    ctext, clvl = chunks[k]
                    if header_flag[k] and clvl > lvl:
                        l = k + 1
                        while l < j and not (header_flag[l] and chunks[l][1] <= lvl):
                            l += 1
                        final_chunks.append(("".join(c[0] for c in chunks[k:l]), lvl))
                        k = l
                    else:
                        final_chunks.append((ctext, lvl))
                        k += 1
                i = j

            for chunk in final_chunks:
                yield chunk
            return

        if split_by == "header":
            sections, i, m = [], 0, len(chunks)
            while i < m:
                text, lvl = chunks[i]
                if not _is_header(text):
                    tail = text
                    i += 1
                    while i < m and not _is_header(chunks[i][0]):
                        tail += chunks[i][0]
                        i += 1
                    sections.append((tail, 0))
                    continue
                sec, header_lvl = text, lvl
                i += 1
                while i < m and not _is_header(chunks[i][0]):
                    sec += chunks[i][0]
                    i += 1
                sections.append((sec, header_lvl))

            final_chunks, i, s = [], 0, len(sections)
            while i < s:
                txt, lvl = sections[i]
                if self.should_split_section(txt, lvl):
                    final_chunks.append((txt, lvl))
                    i += 1
                    continue
                merged = txt
                i += 1
                while i < s and not (_is_header(sections[i][0]) and sections[i][1] <= lvl):
                    merged += sections[i][0]
                    i += 1
                final_chunks.append((merged, lvl))

            for chunk in final_chunks:
                yield chunk
            return

        raise ValueError(f"Invalid split_by: {self.split_by!r}")


class LlamaIndexSplitter(TextSplitter):
    """Splitter class based on a node parser from LlamaIndex that divides text into chunks using nodes.

    !!! info
        For default settings, see `chat.text_splitter_configs.llama_index` in `vectorbtpro._settings.knowledge`.

    Args:
        node_parser (Union[None, str, NodeParser]): Node parser to use,
            specified as a string key, class, or instance.
        node_parser_kwargs (KwargsLike): Keyword arguments to node parser initialization.
        **kwargs: Keyword arguments for `TextSplitter` or used as `node_parser_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "llama_index"

    _settings_path: tp.SettingsPath = "knowledge.chat.text_splitter_configs.llama_index"

    def __init__(
        self,
        node_parser: tp.Union[None, str, NodeParserT] = None,
        node_parser_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        TextSplitter.__init__(self, **kwargs)

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("llama_index")
        from llama_index.core.node_parser import NodeParser

        llama_index_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_node_parser = llama_index_config.pop("node_parser", None)
        def_node_parser_kwargs = llama_index_config.pop("node_parser_kwargs", None)

        if node_parser is None:
            node_parser = def_node_parser
        init_arg_names = set(get_func_arg_names(TextSplitter.__init__)) | set(get_func_arg_names(type(self).__init__))
        for k in list(llama_index_config.keys()):
            if k in init_arg_names:
                llama_index_config.pop(k)

        if isinstance(node_parser, str):
            import llama_index.core.node_parser
            from vectorbtpro.utils.module_ import search_package

            def _match_func(k, v):
                if isinstance(v, type) and issubclass(v, NodeParser):
                    if "." in node_parser:
                        if k.endswith(node_parser):
                            return True
                    else:
                        if k.split(".")[-1].lower() == node_parser.lower():
                            return True
                        if k.split(".")[-1].replace("Splitter", "").replace(
                            "NodeParser", ""
                        ).lower() == node_parser.lower().replace("_", ""):
                            return True
                return False

            found_node_parser = search_package(
                llama_index.core.node_parser,
                _match_func,
                path_attrs=True,
                return_first=True,
            )
            if found_node_parser is None:
                raise ValueError(f"Node parser {node_parser!r} not found")
            node_parser = found_node_parser
        if isinstance(node_parser, type):
            checks.assert_subclass_of(node_parser, NodeParser, arg_name="node_parser")
            node_parser_name = node_parser.__name__.replace("Splitter", "").replace("NodeParser", "").lower()
            module_name = node_parser.__module__
        else:
            checks.assert_instance_of(node_parser, NodeParser, arg_name="node_parser")
            node_parser_name = type(node_parser).__name__.replace("Splitter", "").replace("NodeParser", "").lower()
            module_name = type(node_parser).__module__
        node_parser_configs = llama_index_config.pop("node_parser_configs", {})
        if node_parser_name in node_parser_configs:
            llama_index_config = merge_dicts(llama_index_config, node_parser_configs[node_parser_name])
        elif module_name in node_parser_configs:
            llama_index_config = merge_dicts(llama_index_config, node_parser_configs[module_name])
        node_parser_kwargs = merge_dicts(llama_index_config, def_node_parser_kwargs, node_parser_kwargs)
        model_name = node_parser_kwargs.get("model_name", None)
        if model_name is None:
            func_kwargs = get_func_kwargs(type(node_parser).__init__)
            model_name = func_kwargs.get("model_name", None)
        if isinstance(node_parser, type):
            node_parser = node_parser(**node_parser_kwargs)
        elif len(kwargs) > 0:
            raise ValueError("Cannot apply config to already initialized node parser")

        self._model = model_name
        self._node_parser = node_parser

    @property
    def node_parser(self) -> NodeParserT:
        """LlamaIndex node parser instance used for splitting text.

        Returns:
            NodeParser: Node parser instance used for splitting text.
        """
        return self._node_parser

    def split_text(self, text: str) -> tp.TSTextChunks:
        from llama_index.core.schema import Document

        nodes = self.node_parser.get_nodes_from_documents([Document(text=text)])
        for node in nodes:
            yield node.text


def resolve_text_splitter(text_splitter: tp.TextSplitterLike = None) -> tp.MaybeType[TextSplitter]:
    """Resolve a `TextSplitter` subclass or instance.

    !!! info
        For default settings, see `chat` in `vectorbtpro._settings.knowledge`.

    Args:
        text_splitter (TextSplitterLike): Identifier, subclass, or instance of `TextSplitter`.

            Supported identifiers:

            * "token" for `TokenSplitter`
            * "segment" for `SegmentSplitter`
            * "llama_index" for `LlamaIndexSplitter`

    Returns:
        TextSplitter: Resolved text splitter subclass or instance.
    """
    if text_splitter is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["chat"]
        text_splitter = chat_cfg["text_splitter"]
    if isinstance(text_splitter, str):
        curr_module = sys.modules[__name__]
        found_text_splitter = None
        for name, cls in inspect.getmembers(curr_module, inspect.isclass):
            if name.endswith("Splitter"):
                _short_name = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == text_splitter.lower():
                    found_text_splitter = cls
                    break
        if found_text_splitter is None:
            raise ValueError(f"Invalid text_splitter: {text_splitter!r}")
        text_splitter = found_text_splitter
    if isinstance(text_splitter, type):
        checks.assert_subclass_of(text_splitter, TextSplitter, arg_name="text_splitter")
    else:
        checks.assert_instance_of(text_splitter, TextSplitter, arg_name="text_splitter")
    return text_splitter


def split_text(text: str, text_splitter: tp.TextSplitterLike = None, **kwargs) -> tp.List[str]:
    """Split text into chunks using a specified text splitter.

    Args:
        text (str): Input text to be split.
        text_splitter (TextSplitterLike): Identifier, subclass, or instance of `TextSplitter`.

            Resolved using `resolve_text_splitter`.
        **kwargs: Keyword arguments to initialize or update `text_splitter`.

    Returns:
        List[str]: List of text chunks.
    """
    text_splitter = resolve_text_splitter(text_splitter=text_splitter)
    if isinstance(text_splitter, type):
        text_splitter = text_splitter(**kwargs)
    elif kwargs:
        text_splitter = text_splitter.replace(**kwargs)
    return list(text_splitter.split_text(text))
