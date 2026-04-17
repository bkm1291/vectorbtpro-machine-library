# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing classes and utilities for completions."""

import inspect
import io
import json
import re
import sys
import textwrap
import time
from pathlib import Path

from vectorbtpro import _typing as tp
from vectorbtpro.knowledge.formatting import ContentFormatter, HTMLFileFormatter, resolve_formatter, ThoughtProcessor
from vectorbtpro.knowledge.tokenization import Tokenizer, TikTokenizer, resolve_tokenizer, tokenize
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts, flat_merge_dicts, has, get
from vectorbtpro.utils.formatting import dump, get_dump_language, head_and_tail
from vectorbtpro.utils.parsing import get_func_arg_names, get_func_kwargs
from vectorbtpro.utils.pbar import ProgressBar
from vectorbtpro.utils.refs import get_obj
from vectorbtpro.utils.template import CustomTemplate, SafeSub, RepFunc
from vectorbtpro.utils.warnings_ import warn

if tp.TYPE_CHECKING:
    from openai import OpenAI as OpenAIT, Stream as StreamT
    from openai.types.chat.chat_completion import ChatCompletion as ChatCompletionT
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as ChatCompletionChunkT
    from openai.types.responses import Response as ResponseT, ResponseStreamEvent as ResponseStreamEventT
else:
    OpenAIT = "openai.OpenAI"
    StreamT = "openai.Stream"
    ChatCompletionT = "openai.types.chat.chat_completion.ChatCompletion"
    ChatCompletionChunkT = "openai.types.chat.chat_completion_chunk.ChatCompletionChunk"
    ResponseT = "openai.types.responses.Response"
    ResponseStreamEventT = "openai.types.responses.ResponseStreamEvent"
if tp.TYPE_CHECKING:
    from anthropic import Client as AnthropicClientT, Stream as AnthropicStreamT
    from anthropic.types import Message as AnthropicMessageT, MessageStreamEvent as AnthropicMessageStreamEventT
else:
    AnthropicClientT = "anthropic.Client"
    AnthropicStreamT = "anthropic.Stream"
    AnthropicMessageT = "anthropic.types.Message"
    AnthropicMessageStreamEventT = "anthropic.types.MessageStreamEvent"
if tp.TYPE_CHECKING:
    from google.genai import Client as GenAIClientT
    from google.genai.types import Content as ContentT, GenerateContentResponse as GenerateContentResponseT
else:
    GenAIClientT = "google.genai.Client"
    ContentT = "google.genai.types.Content"
    GenerateContentResponseT = "google.genai.types.GenerateContentResponse"
if tp.TYPE_CHECKING:
    from huggingface_hub import (
        InferenceClient as InferenceClientT,
        ChatCompletionOutput as ChatCompletionOutputT,
        ChatCompletionStreamOutput as ChatCompletionStreamOutputT,
    )
else:
    InferenceClientT = "huggingface_hub.InferenceClient"
    ChatCompletionOutputT = "huggingface_hub.ChatCompletionOutput"
    ChatCompletionStreamOutputT = "huggingface_hub.ChatCompletionStreamOutput"
if tp.TYPE_CHECKING:
    from litellm import ModelResponse as ModelResponseT, CustomStreamWrapper as CustomStreamWrapperT
else:
    ModelResponseT = "litellm.ModelResponse"
    CustomStreamWrapperT = "litellm.CustomStreamWrapper"
if tp.TYPE_CHECKING:
    from llama_index.core.llms import LLM as LLMT, ChatResponse as ChatResponseT, ChatMessage as ChatMessageT
else:
    LLMT = "llama_index.core.llms.LLM"
    ChatResponseT = "llama_index.core.llms.ChatResponse"
    ChatMessageT = "llama_index.core.llms.ChatMessage"
if tp.TYPE_CHECKING:
    from ollama import Client as OllamaClientT, ChatResponse as OllamaChatResponseT
else:
    OllamaClientT = "ollama.Client"
    OllamaChatResponseT = "ollama.ChatResponse"

__all__ = [
    "Completions",
    "OpenAICompletions",
    "AnthropicCompletions",
    "GeminiCompletions",
    "HFInferenceCompletions",
    "LiteLLMCompletions",
    "LlamaIndexCompletions",
    "OllamaCompletions",
    "complete",
    "completed",
]


class Completions(ThoughtProcessor):
    """Abstract class for completion providers.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and
        its sub-configurations `chat` and `chat.completions_config`.

    Args:
        context (str): Context string to be used as a user message.
        chat_history (Optional[ChatHistory]): Chat history, a list of dictionaries with defined roles.

            After a response is generated, the assistant message is appended to this history.
        stream (Optional[bool]): Boolean indicating whether responses are streamed.

            In streaming mode, chunks are appended and displayed incrementally; otherwise,
            the entire message is displayed.
        max_tokens (Union[None, bool, int]): Maximum token limit configured for messages.

            If False, the limit is disabled.
        tokenizer (TokenizerLike): Identifier, subclass, or instance of `vectorbtpro.knowledge.tokenization.Tokenizer`.

            Resolved using `vectorbtpro.knowledge.tokenization.resolve_tokenizer`.
        tokenizer_kwargs (KwargsLike): Keyword arguments to initialize or update `tokenizer`.
        system_prompt (Optional[str]): System prompt that precedes the context prompt.

            This prompt is used to set the system's behavior or context for the conversation.
        system_as_user (Optional[bool]): Boolean indicating whether to use the user role for the system message.

            This is mainly used for experimental models where a dedicated system role is not available.
        context_template (Optional[str]): Context template requiring a 'context' variable.

            The template can be a string, a function, or an instance of `vectorbtpro.utils.template.CustomTemplate`.
        formatter (ContentFormatterLike): Identifier, subclass, or instance of
            `vectorbtpro.knowledge.formatting.ContentFormatter`.

            Resolved using `vectorbtpro.knowledge.formatting.resolve_formatter`.

            This formatter is used to format the content of the response.
        formatter_kwargs (KwargsLike): Keyword arguments to initialize or update `formatter`.
        minimal_format (Optional[bool]): Boolean indicating if the input is minimally formatted.
        quick_mode (Optional[bool]): Boolean indicating whether quick mode is enabled.
        tools (Optional[Tools]): Tools to be used in the conversation.

            If a string is provided, it must be either the name of a registered tool,
            "registry" to use all available tools from the registry,
            "mcp" to use the MCP tools from `vectorbtpro.mcp_server.tool_registry`,
            or "all" to use both all available tools and the MCP tools.

            If a list is provided, it must be a list of registered tool names or functions.
            Any unregistered functions will be added to the registry.
        tool_registry (Optional[Dict[str, Callable]]): Registry mapping tool names to functions for execution.
        max_tool_calls (Optional[int]): Maximum number of tool calls per request.
        tool_dump_kwargs (KwargsLike): Keyword arguments for dumping structured data from tools.

            See `vectorbtpro.utils.formatting.dump`.
        tool_request_template (Optional[CustomTemplateLike]): Template for tool requests.

            The template can be a string, a function, or an instance of `vectorbtpro.utils.template.CustomTemplate`.

            Allowed keys include `id`, `name`, `arguments`, `token_count`, and `payload`,
            as well as the keys from `template_context`.
        tool_response_template (Optional[CustomTemplateLike]): Template for tool responses.

            The template can be a string, a function, or an instance of `vectorbtpro.utils.template.CustomTemplate`.

            Allowed keys include `id`, `name`, `output`, `success`, `token_count`, and `payload`,
            as well as the keys from `template_context`.
        tool_display_format (Optional[str]): Format for displaying tool information.

            Supports the following options:

            * "none": don't display tool information.
            * "minimal": display calls but don't display tool request/response payloads.
            * "compact": display only head and tail of tool request/response payloads.
            * "full": display full tool request/response payloads.
        silence_warnings (Optional[bool]): Flag to suppress warning messages.
        show_progress (Optional[bool]): Flag indicating whether to display the progress bar.
        pbar_kwargs (Kwargs): Keyword arguments for configuring the progress bar.
        template_context (KwargsLike): Additional context for template substitution.
        **kwargs: Keyword arguments for `vectorbtpro.knowledge.formatting.ThoughtProcessor`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the class."""

    _expected_keys_mode: tp.ExpectedKeysMode = "disable"

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat", "knowledge.chat.completions_config"]

    def __init__(
        self,
        context: str = "",
        chat_history: tp.Optional[tp.ChatHistory] = None,
        stream: tp.Optional[bool] = None,
        max_tokens: tp.Union[None, bool, int] = None,
        tokenizer: tp.TokenizerLike = None,
        tokenizer_kwargs: tp.KwargsLike = None,
        system_prompt: tp.Optional[str] = None,
        system_as_user: tp.Optional[bool] = None,
        context_template: tp.Optional[str] = None,
        formatter: tp.ContentFormatterLike = None,
        formatter_kwargs: tp.KwargsLike = None,
        minimal_format: tp.Optional[bool] = None,
        quick_mode: tp.Optional[bool] = None,
        tools: tp.Optional[tp.Tools] = None,
        tool_registry: tp.Optional[tp.Dict[str, tp.Callable]] = None,
        max_tool_calls: tp.Optional[int] = None,
        tool_dump_kwargs: tp.KwargsLike = None,
        tool_request_template: tp.CustomTemplateLike = None,
        tool_response_template: tp.CustomTemplateLike = None,
        tool_display_format: tp.Optional[str] = None,
        silence_warnings: tp.Optional[bool] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        ThoughtProcessor.__init__(
            self,
            context=context,
            chat_history=chat_history,
            stream=stream,
            max_tokens=max_tokens,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            system_prompt=system_prompt,
            system_as_user=system_as_user,
            context_template=context_template,
            formatter=formatter,
            formatter_kwargs=formatter_kwargs,
            minimal_format=minimal_format,
            quick_mode=quick_mode,
            tools=tools,
            tool_registry=tool_registry,
            max_tool_calls=max_tool_calls,
            tool_dump_kwargs=tool_dump_kwargs,
            tool_request_template=tool_request_template,
            tool_response_template=tool_response_template,
            tool_display_format=tool_display_format,
            silence_warnings=silence_warnings,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            template_context=template_context,
            **kwargs,
        )

        if chat_history is None:
            chat_history = []
        stream = self.resolve_setting(stream, "stream")
        max_tokens_set = max_tokens is not None
        max_tokens = self.resolve_setting(max_tokens, "max_tokens")
        tokenizer = self.resolve_setting(tokenizer, "tokenizer", default=None)
        tokenizer_kwargs = self.resolve_setting(tokenizer_kwargs, "tokenizer_kwargs", default=None, merge=True)
        system_prompt = self.resolve_setting(system_prompt, "system_prompt")
        system_as_user = self.resolve_setting(system_as_user, "system_as_user")
        context_template = self.resolve_setting(context_template, "context_template")
        formatter = self.resolve_setting(formatter, "formatter", default=None)
        formatter_kwargs = self.resolve_setting(formatter_kwargs, "formatter_kwargs", default=None, merge=True)
        minimal_format = self.resolve_setting(minimal_format, "minimal_format", default=None)
        quick_mode = self.resolve_setting(quick_mode, "quick_mode")
        tools = self.resolve_setting(tools, "tools")
        tool_registry = self.resolve_setting(tool_registry, "tool_registry", merge=True)
        max_tool_calls = self.resolve_setting(max_tool_calls, "max_tool_calls")
        tool_dump_kwargs = self.resolve_setting(tool_dump_kwargs, "tool_dump_kwargs", merge=True)
        tool_request_template = self.resolve_setting(tool_request_template, "tool_request_template")
        tool_response_template = self.resolve_setting(tool_response_template, "tool_response_template")
        tool_display_format = self.resolve_setting(tool_display_format, "tool_display_format")
        silence_warnings = self.resolve_setting(silence_warnings, "silence_warnings")
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        if tools is None:
            tools = []
        elif not isinstance(tools, (list, dict)):
            tools = [tools]

        def _merge_mcp_into_registry():
            nonlocal tool_registry
            from vectorbtpro.mcp_server import tool_registry as mcp_tool_registry

            if not isinstance(tool_registry, dict):
                tool_registry = dict(tool_registry)
            added = []
            for name, func in mcp_tool_registry.items():
                if name not in tool_registry:
                    tool_registry[name] = func
                    added.append(name)
            return added

        new_tools = []
        for t in tools:
            if isinstance(tools, dict):
                name = t
                t = tools[t]
            else:
                name = None
            if isinstance(t, str):
                k = t.lower()
                if k == "registry":
                    new_tools.extend(tool_registry.keys())
                    continue
                if k in {"mcp", "all"}:
                    added = _merge_mcp_into_registry()
                    if k == "mcp":
                        new_tools.extend(added)
                    else:
                        new_tools.extend(tool_registry.keys())
                    continue
                if t in tool_registry:
                    new_tools.append(t)
                    continue
                t = get_obj(t)

            if inspect.isfunction(t) and not isinstance(t, str):
                if not name:
                    name = getattr(t, "__name__", None)
                if not name:
                    raise ValueError(f"Tool {t!r} is not registered")
                if name not in tool_registry:
                    tool_registry = dict(tool_registry)
                    tool_registry[name] = t
                new_tools.append(name)
            elif isinstance(t, str):
                if t not in tool_registry:
                    raise ValueError(f"Tool '{t}' is not found in tool_registry")
                new_tools.append(t)
            else:
                raise TypeError(f"Tool {t!r} must be a string or function")

        tools = list(dict.fromkeys(new_tools))

        tokenizer = resolve_tokenizer(tokenizer)
        formatter = resolve_formatter(formatter)

        self._context = context
        self._chat_history = chat_history
        self._stream = stream
        self._max_tokens_set = max_tokens_set
        self._max_tokens = max_tokens
        self._tokenizer = tokenizer
        self._tokenizer_kwargs = tokenizer_kwargs
        self._system_prompt = system_prompt
        self._system_as_user = system_as_user
        self._context_template = context_template
        self._formatter = formatter
        self._formatter_kwargs = formatter_kwargs
        self._minimal_format = minimal_format
        self._quick_mode = quick_mode
        self._tools = tools
        self._tool_registry = tool_registry
        self._max_tool_calls = max_tool_calls
        self._tool_dump_kwargs = tool_dump_kwargs
        self._tool_request_template = tool_request_template
        self._tool_response_template = tool_response_template
        self._tool_display_format = tool_display_format
        self._silence_warnings = silence_warnings
        self._show_progress = show_progress
        self._pbar_kwargs = pbar_kwargs
        self._template_context = template_context

    @property
    def context(self) -> str:
        """Context string to be used as a user message.

        Returns:
            str: Context string used for expression evaluation.
        """
        return self._context

    @property
    def chat_history(self) -> tp.ChatHistory:
        """Chat history, a list of dictionaries with defined roles.

        After a response is generated, the assistant message is appended to this history.

        Returns:
            ChatHistory: List of dictionaries representing the chat history.
        """
        return self._chat_history

    @property
    def stream(self) -> bool:
        """Boolean indicating whether responses are streamed.

        In streaming mode, chunks are appended and displayed incrementally; otherwise,
        the entire message is displayed.

        Returns:
            bool: True if streaming is enabled, False otherwise.
        """
        return self._stream

    @property
    def max_tokens_set(self) -> tp.Optional[int]:
        """Boolean indicating if `Completions.max_tokens` was explicitly provided by the user.

        Returns:
            Optional[int]: Maximum token limit set by the user; None if not set.
        """
        return self._max_tokens_set

    @property
    def max_tokens(self) -> tp.Union[bool, int]:
        """Maximum token limit configured for messages.

        Returns:
            Union[bool, int]: Maximum token limit; False if disabled.
        """
        return self._max_tokens

    @property
    def tokenizer(self) -> tp.MaybeType[Tokenizer]:
        """Subclass or instance of `vectorbtpro.knowledge.tokenization.Tokenizer`.

        Resolved using `vectorbtpro.knowledge.tokenization.resolve_tokenizer`.

        Returns:
            MaybeType[Tokenizer]: Resolved tokenizer instance or subclass.
        """
        return self._tokenizer

    @property
    def tokenizer_kwargs(self) -> tp.Kwargs:
        """Keyword arguments to initialize or update `Completions.tokenizer`.

        Returns:
            Kwargs: Keyword arguments for tokenizer initialization or update.
        """
        return self._tokenizer_kwargs

    @property
    def system_prompt(self) -> str:
        """System prompt that precedes the context prompt.

        This prompt is used to set the system's behavior or context for the conversation.

        Returns:
            str: System prompt.
        """
        return self._system_prompt

    @property
    def system_as_user(self) -> bool:
        """Boolean indicating whether to use the user role for the system message.

        This is mainly used for experimental models where a dedicated system role is not available.

        Returns:
            bool: True if the system message is treated as a user message, False otherwise.
        """
        return self._system_as_user

    @property
    def context_template(self) -> str:
        """Context prompt template requiring a 'context' variable.

        The template can be a string, a function, or an instance of `vectorbtpro.utils.template.CustomTemplate`.

        This prompt is used to provide context for the conversation.

        Returns:
            str: Context prompt template.
        """
        return self._context_template

    @property
    def formatter(self) -> tp.MaybeType[ContentFormatter]:
        """Content formatter subclass or instance.

        Resolved using `vectorbtpro.knowledge.formatting.resolve_formatter`.

        This formatter is used to format the content of the response.

        Returns:
            MaybeType[ContentFormatter]: Resolved content formatter instance or subclass.
        """
        return self._formatter

    @property
    def formatter_kwargs(self) -> tp.Kwargs:
        """Keyword arguments to initialize or update `Completions.formatter`.

        Returns:
            Kwargs: Keyword arguments for the content formatter.
        """
        return self._formatter_kwargs

    @property
    def minimal_format(self) -> bool:
        """Boolean indicating if the input is minimally formatted.

        Returns:
            bool: True if the input is minimally formatted, False otherwise.
        """
        return self._minimal_format

    @property
    def quick_mode(self) -> bool:
        """Boolean indicating whether quick mode is enabled.

        Returns:
            bool: True if quick mode is enabled, False otherwise.
        """
        return self._quick_mode

    @property
    def tools(self) -> tp.List[str]:
        """List of tool names to be used in the conversation.

        Each tool name must correspond to a registered tool in the tool registry.

        Returns:
            List[str]: List of tool names.
        """
        return self._tools

    @property
    def tool_registry(self) -> tp.Dict[str, tp.Callable]:
        """Registry mapping tool names to functions for execution.

        Returns:
            Mapping[str, Callable]: Tool registry if configured.
        """
        return self._tool_registry

    @property
    def max_tool_calls(self) -> tp.Optional[int]:
        """Maximum number of tool calls per request.

        Returns:
            Optional[int]: Max number of tool-call steps, or None if not set.
        """
        return self._max_tool_calls

    @property
    def tool_dump_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for dumping structured data from tools.

        See `vectorbtpro.utils.formatting.dump`.

        Returns:
            Kwargs: Dictionary of keyword arguments.
        """
        return self._tool_dump_kwargs

    @property
    def tool_request_template(self) -> tp.CustomTemplateLike:
        """Template for tool requests.

        The template can be a string, a function, or an instance of `vectorbtpro.utils.template.CustomTemplate`.

        Returns:
            CustomTemplateLike: Tool request template.
        """
        return self._tool_request_template

    @property
    def tool_response_template(self) -> tp.CustomTemplateLike:
        """Template for tool responses.

        The template can be a string, a function, or an instance of `vectorbtpro.utils.template.CustomTemplate`.

        Returns:
            CustomTemplateLike: Tool response template.
        """
        return self._tool_response_template

    @property
    def tool_display_format(self) -> str:
        """Display format for tool responses.

        Can be one of "none", "minimal", "compact", and "full".

        Returns:
            str: Tool display format.
        """
        return self._tool_display_format

    @property
    def silence_warnings(self) -> bool:
        """Boolean indicating whether warnings are suppressed.

        Returns:
            bool: True if warnings are suppressed, False otherwise.
        """
        return self._silence_warnings

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
        """Model name.

        Returns:
            Optional[str]: Model name if specified; otherwise, None.
        """
        return None

    def get_chat_response(self, messages: tp.ChatMessages, enable_tools: bool = True, **kwargs) -> tp.Any:
        """Return a chat response based on the provided messages.

        !!! abstract
            This method should be overridden in a subclass.

        Args:
            messages (ChatMessages): List representing the conversation history.
            enable_tools (bool): Whether to enable tool usage.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Chat response generated from the provided messages.
        """
        raise NotImplementedError

    def get_message_content(self, response: tp.Any) -> tp.Optional[str]:
        """Return the content extracted from a chat response.

        !!! abstract
            This method should be overridden in a subclass.

        Args:
            response (Any): Chat response object.

        Returns:
            Optional[str]: Content extracted from the chat response.
        """
        raise NotImplementedError

    def get_stream_response(self, messages: tp.ChatMessages, enable_tools: bool = True, **kwargs) -> tp.Any:
        """Return a streaming response generated from the provided messages.

        !!! abstract
            This method should be overridden in a subclass.

        Args:
            messages (ChatMessages): List representing the conversation history.
            enable_tools (bool): Whether to enable tool usage.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Streaming response generated from the provided messages.
        """
        raise NotImplementedError

    def get_delta_content(self, response_chunk: tp.Any) -> tp.Optional[str]:
        """Return the content extracted from a streaming response chunk.

        !!! abstract
            This method should be overridden in a subclass.

        Args:
            response_chunk (Any): Streaming response chunk object.

        Returns:
            Optional[str]: Content extracted from the streaming response chunk.
        """
        raise NotImplementedError

    def prepare_messages(self, message: str) -> tp.ChatMessages:
        """Return a list of chat messages formatted for a completion request.

        Args:
            message (str): User message to process.

        Returns:
            ChatMessages: List of dictionaries representing the conversation history.
        """
        context = self.context
        chat_history = self.chat_history
        max_tokens_set = self.max_tokens_set
        max_tokens = self.max_tokens
        tokenizer = self.tokenizer
        tokenizer_kwargs = self.tokenizer_kwargs
        system_prompt = self.system_prompt
        system_as_user = self.system_as_user
        context_template = self.context_template
        template_context = self.template_context
        silence_warnings = self.silence_warnings

        if isinstance(tokenizer, type):
            tokenizer_kwargs = dict(tokenizer_kwargs)
            tokenizer_kwargs["template_context"] = merge_dicts(
                template_context, tokenizer_kwargs.get("template_context", None)
            )
            if issubclass(tokenizer, TikTokenizer) and "model" not in tokenizer_kwargs:
                tokenizer_kwargs["model"] = self.model
            tokenizer = tokenizer(**tokenizer_kwargs)
        elif tokenizer_kwargs:
            tokenizer = tokenizer.replace(**tokenizer_kwargs)

        if context:
            if isinstance(context_template, str):
                context_template = SafeSub(context_template)
            elif checks.is_function(context_template):
                context_template = RepFunc(context_template)
            elif not isinstance(context_template, CustomTemplate):
                raise TypeError("Context prompt must be a string, function, or template")
            if max_tokens not in (None, False):
                if max_tokens is True:
                    raise ValueError("max_tokens cannot be True")
                empty_context_template = context_template.substitute(
                    flat_merge_dicts(dict(context=""), template_context),
                    eval_id="context_template",
                )
                empty_messages = [
                    dict(role="user" if system_as_user else "system", content=system_prompt),
                    dict(role="user", content=empty_context_template),
                    *chat_history,
                    dict(role="user", content=message),
                ]
                num_tokens = tokenizer.count_tokens_in_messages(empty_messages)
                max_context_tokens = max(0, max_tokens - num_tokens)
                encoded_context = tokenizer.encode(context)
                if len(encoded_context) > max_context_tokens:
                    context = tokenizer.decode(encoded_context[:max_context_tokens])
                    if not max_tokens_set and not silence_warnings:
                        warn(
                            f"Context is too long ({len(encoded_context)}). "
                            f"Truncating to {max_context_tokens} tokens."
                        )
            template_context = flat_merge_dicts(dict(context=context), template_context)
            context_template = context_template.substitute(template_context, eval_id="context_template")
            return [
                dict(role="user" if system_as_user else "system", content=system_prompt),
                dict(role="user", content=context_template),
                *chat_history,
                dict(role="user", content=message),
            ]
        else:
            return [
                dict(role="user" if system_as_user else "system", content=system_prompt),
                *chat_history,
                dict(role="user", content=message),
            ]

    @property
    def supports_tool_calling(self) -> bool:
        """Whether this provider supports tool calling.

        Returns:
            bool: True if tool calling is supported, False otherwise.
        """
        return False

    def function_to_tool_spec(
        self,
        func: tp.Callable,
        name: tp.Optional[str] = None,
        strict: bool = False,
        draft_2020_12: bool = False,
    ) -> tp.Kwargs:
        """Convert a typed Python callable into a minimal tool specification in JSON Schema format.

        Args:
            func (Callable): Callable with type-annotated parameters.
            name (Optional[str]): Name of the tool.
            strict (bool): If True, make the schema strict.
            draft_2020_12 (bool): If True, emit tuple arrays using JSON Schema 2020-12.

        Returns:
            Kwargs: Tool specification for the function.
        """
        import types as _types
        import typing as _typing
        from typing import Any, get_type_hints, get_origin, get_args, Annotated, Union, Literal
        from collections.abc import Sequence as ABCSequence, Mapping as ABCMapping
        from vectorbtpro.utils.module_ import check_installed

        try:
            from typing import (
                TypedDict as _TypedDict,
                Required as _Required,
                NotRequired as _NotRequired,
            )
        except Exception:
            try:
                from typing_extensions import (
                    TypedDict as _TypedDict,
                    Required as _Required,
                    NotRequired as _NotRequired,
                )
            except Exception:
                _TypedDict = _Required = _NotRequired = None

        if not callable(func):
            raise TypeError("Function must be callable")

        if name is None:
            name = func.__name__

        PRIMS = {str: "string", int: "integer", float: "number", bool: "boolean"}

        def _json_defaultable(v):
            try:
                json.dumps(v)
                return True
            except TypeError:
                return False

        _TD_META = getattr(_typing, "_TypedDictMeta", None)

        def _is_typeddict(tp_obj):
            if _TD_META is not None and isinstance(tp_obj, _TD_META):
                return True
            return isinstance(tp_obj, type) and hasattr(tp_obj, "__annotations__") and hasattr(tp_obj, "__total__")

        def _unwrap_required_markers(ann):
            if _Required is not None and get_origin(ann) is _Required:
                return get_args(ann)[0], True, False
            if _NotRequired is not None and get_origin(ann) is _NotRequired:
                return get_args(ann)[0], False, True
            return ann, None, None

        def _schema(tp):
            if tp is Any:
                return {}

            if tp in PRIMS:
                return {"type": PRIMS[tp]}

            if tp is type(None):
                return {"type": "null"}

            if _TypedDict is not None and _is_typeddict(tp):
                annotations = {}
                for base in reversed(getattr(tp, "__mro__", ())):
                    if _is_typeddict(base):
                        annotations.update(getattr(base, "__annotations__", {}) or {})
                required_keys = set(getattr(tp, "__required_keys__", set()))
                optional_keys = set(getattr(tp, "__optional_keys__", set()))
                props = {}
                if not required_keys and not optional_keys:
                    total = getattr(tp, "__total__", True)
                    inferred_required, inferred_optional = set(), set()
                    for key, ann in annotations.items():
                        base_ann, is_req, is_opt = _unwrap_required_markers(ann)
                        props[key] = _schema(base_ann)
                        if is_req is True:
                            inferred_required.add(key)
                        elif is_opt is True:
                            inferred_optional.add(key)
                        else:
                            (inferred_required if total else inferred_optional).add(key)
                    required_keys, optional_keys = inferred_required, inferred_optional
                else:
                    for key, ann in annotations.items():
                        base_ann, _, _ = _unwrap_required_markers(ann)
                        props[key] = _schema(base_ann)
                return {
                    "type": "object",
                    "properties": props,
                    "required": sorted(required_keys),
                    "additionalProperties": False,
                }

            origin = get_origin(tp)
            args = get_args(tp)

            union_type = getattr(_types, "UnionType", None)
            if origin in (Union, union_type):
                return {"anyOf": [_schema(a) for a in args]}

            if origin is Literal:
                vals = list(args)
                enum_types = {type(v) for v in vals if v is not None}
                out = {"enum": vals}
                if len(enum_types) == 1:
                    inferred = PRIMS.get(enum_types.pop(), None)
                    if inferred:
                        out["type"] = inferred
                return out

            if origin is Annotated and args:
                base, *meta = args
                out = _schema(base)
                for m in meta:
                    if isinstance(m, str):
                        out.setdefault("description", m)
                    elif isinstance(m, dict):
                        out.update(m)
                return out

            if origin in (list, ABCSequence):
                item_t = args[0] if args else Any
                return {"type": "array", "items": _schema(item_t) if item_t is not Any else {}}

            if origin is tuple:
                if len(args) == 2 and args[1] is Ellipsis:
                    return {"type": "array", "items": _schema(args[0])}
                if args:
                    item_schemas = [_schema(a) for a in args]
                    if draft_2020_12:
                        return {
                            "type": "array",
                            "prefixItems": item_schemas,
                            "items": False,
                        }
                    else:
                        return {
                            "type": "array",
                            "items": item_schemas,
                            "additionalItems": False,
                            "minItems": len(item_schemas),
                            "maxItems": len(item_schemas),
                        }
                return {"type": "array"}

            if origin in (dict, ABCMapping):
                if strict:
                    raise ValueError(
                        "Strict mode: mapping/dict parameters are not supported. "
                        "Use a TypedDict or fixed-object schema instead."
                    )
                key_t, val_t = (args + (Any, Any))[:2] if args else (Any, Any)
                if key_t not in (str, Any):
                    raise ValueError("Only dicts with string keys are supported")
                addl = _schema(val_t) if val_t is not Any else {}
                return {"type": "object", "additionalProperties": addl}

            raise ValueError(f"Unsupported annotation: {tp!r}")

        def _nullable(sch):
            if not isinstance(sch, dict):
                return sch
            if "anyOf" in sch:
                anyof = list(sch["anyOf"])
                if not any(isinstance(x, dict) and x.get("type") == "null" for x in anyof):
                    anyof.append({"type": "null"})
                return {**sch, "anyOf": anyof}
            return {"anyOf": [sch, {"type": "null"}]}

        def _strictify(schema, path="<root>"):
            if not isinstance(schema, dict):
                return schema
            if schema == {}:
                raise ValueError(
                    f"Strict mode: unresolved type at {path}. " "Annotate with a concrete type instead of 'Any'."
                )
            t = schema.get("type")

            if "anyOf" in schema:
                schema["anyOf"] = [_strictify(s, f"{path}/anyOf[{i}]") for i, s in enumerate(schema["anyOf"])]
                return schema

            if t == "object":
                props = dict(schema.get("properties", {}))
                pre_required = set(schema.get("required", []))
                for k, v in list(props.items()):
                    if k not in pre_required:
                        props[k] = _nullable(v)
                for k, v in list(props.items()):
                    props[k] = _strictify(v, f"{path}/properties/{k}")
                schema["properties"] = props
                schema["required"] = sorted(list(props.keys()))
                schema["additionalProperties"] = False
                return schema

            if t == "array":
                if "items" not in schema or schema["items"] == {}:
                    raise ValueError(f"Strict mode: array at {path or '<root>'} must specify a concrete item schema.")
                schema["items"] = _strictify(schema["items"], f"{path}/items")
                return schema

            return schema

        sig = inspect.signature(func)
        hints = get_type_hints(func, globalns=getattr(func, "__globals__", None), include_extras=True)

        desc = inspect.getdoc(func) or ""
        param_docs = {}
        if check_installed("docstring_parser"):
            from docstring_parser import parse

            try:
                docstring = parse(desc)
                desc = docstring.description
                if getattr(docstring, "deprecation", None):
                    if not desc.endswith("\n"):
                        desc += "\n"
                    desc += "\nDeprecated:\n"
                    d = docstring.deprecation.description
                    desc += textwrap.indent(d, " " * 4)
                if getattr(docstring, "raises", []):
                    if not desc.endswith("\n"):
                        desc += "\n"
                    desc += "\nRaises:\n"
                    for raise_doc in docstring.raises:
                        d = f"{raise_doc.type_name}: {raise_doc.description}"
                        desc += textwrap.indent(d, " " * 4)
                if getattr(docstring, "examples", []):
                    if not desc.endswith("\n"):
                        desc += "\n"
                    desc += "\nExamples:\n"
                    for example_doc in docstring.examples:
                        d = example_doc.description
                        desc += textwrap.indent(d, " " * 4)
                param_docs = {param.arg_name: param for param in docstring.params}
            except Exception:
                pass

        props = {}
        required = []

        for pname, param in sig.parameters.items():
            if pname in ("self", "cls", "cls_or_self"):
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                raise TypeError(f"{name} uses *args/**kwargs which cannot be expressed in JSON Schema")

            ann = hints.get(pname, Any)

            pdoc = param_docs.get(pname, None)
            if pdoc and ann is Any and pdoc.type_name:
                eval_globals = {}
                eval_globals.update(getattr(func, "__globals__", {}) or {})
                eval_globals.update(vars(_typing))
                try:
                    ann = eval(pdoc.type_name, eval_globals, {})
                except Exception:
                    pass
            if pdoc and pdoc.is_optional:
                ann = Union[ann, type(None)]

            sch = _schema(ann)

            if pdoc and pdoc.description:
                sch = dict(sch)
                sch["description"] = pdoc.description

            if param.default is inspect._empty:
                required.append(pname)
            else:
                if _json_defaultable(param.default):
                    sch = dict(sch)
                    sch["default"] = param.default

            props[pname] = sch

        parameters_schema = {"type": "object", "properties": props, "required": required}
        if strict:
            parameters_schema = _strictify(parameters_schema)
        else:
            parameters_schema["additionalProperties"] = False
        tool_spec = {
            "type": "function",
            "name": name,
            "description": desc,
            "parameters": parameters_schema,
        }
        if strict:
            tool_spec["strict"] = True
        return tool_spec

    def get_chat_tool_calls(self, response: tp.Any) -> tp.List[tp.Kwargs]:
        """Return tool call descriptors from a chat response.

        Each descriptor should at least include 'id' (if available), 'name', and 'arguments'.
        Arguments can be a dict or a JSON string; they will be normalized by `Completions.execute_tool_calls`.

        Args:
            response (Any): Chat response object.

        Returns:
            List[Kwargs]: List of extracted tool call descriptors.
        """
        return []

    def get_stream_tool_calls(self, response_chunks: tp.Iterator[tp.Any]) -> tp.List[tp.Kwargs]:
        """Return tool call descriptors from a streaming response.

        Each descriptor should at least include 'id' (if available), 'name', and 'arguments'.
        Arguments can be a dict or a JSON string; they will be normalized by `Completions.execute_tool_calls`.

        Args:
            response_chunks (Iterator[Any]): Iterator of streaming response chunk objects.

        Returns:
            List[Kwargs]: List of extracted tool call descriptors.
        """
        return []

    def get_tool_call_messages(self, tool_calls: tp.List[tp.Kwargs]) -> tp.List[tp.Kwargs]:
        """Return assistant messages built from tool calls.

        Args:
            tool_calls (List[Kwargs]): List of tool call descriptors.

        Returns:
            List[Kwargs]: Built assistant messages.
        """
        raise NotImplementedError

    def get_tool_result_messages(self, tool_results: tp.List[tp.Kwargs]) -> tp.List[tp.Kwargs]:
        """Return provider-specific tool result messages to append to the next request.

        Args:
            tool_results (List[Kwargs]): List of tool result descriptors.

        Returns:
            List[Kwargs]: List of built tool result messages.
        """
        raise NotImplementedError

    def normalize_tool_calls(self, tool_calls: tp.List[tp.Kwargs]) -> tp.List[tp.Kwargs]:
        """Normalize tool calls to a standard format.

        Args:
            tool_calls (List[Kwargs]): List of tool call descriptors.

        Returns:
            List[Kwargs]: List of normalized tool call descriptors.
        """

        def _normalize_arguments(arguments):
            if isinstance(arguments, dict):
                return arguments
            try:
                return _normalize_arguments(json.loads(arguments))
            except (json.JSONDecodeError, TypeError):
                try:
                    from vectorbtpro.utils.eval_ import evaluate

                    return _normalize_arguments(evaluate(arguments))
                except Exception:
                    return {"raw": arguments}

        normalized = []
        for tc in tool_calls:
            tc = dict(tc)
            id = tc.pop("id", None)
            name = tc.pop("name", None)
            arguments = _normalize_arguments(tc.pop("arguments", {}))
            normalized.append({"id": id, "name": name, "arguments": arguments, **tc})
        return normalized

    def execute_tool_calls(self, tool_calls: tp.List[tp.Kwargs]) -> tp.List[tp.Kwargs]:
        """Execute tool calls using the provided registry.

        Args:
            tool_calls (List[Kwargs]): List of tool call descriptors.

                Each descriptor should at least include 'id', 'name', and 'arguments' (dict).

        Returns:
            List[Kwargs]: List of tool results.

                Each result should include 'id', 'name', 'output' (string), and 'success' (boolean).
        """
        results = []
        for tc in tool_calls:
            name = tc["name"]
            call_id = tc["id"]
            if not name:
                out = "No tool name provided"
                success = False
            elif name not in self.tool_registry:
                out = f"Tool not found: {name!r}"
                success = False
            else:
                func = self.tool_registry[name]
                try:
                    out = func(**tc["arguments"])
                    success = True
                except Exception as e:
                    out = f"Tool execution error: {e!r}"
                    success = False
                if not isinstance(out, str):
                    try:
                        out = json.dumps(out, ensure_ascii=False)
                    except Exception:
                        out = str(out)
            results.append({"id": call_id, "name": name, "output": out, "success": success})
        return results

    @classmethod
    def format_payload(cls, payload: str, language: tp.Optional[str] = None) -> str:
        """Return a fenced Markdown code block that safely contains `payload`,
        even if `payload` contains backtick or tilde code fences.

        Args:
            payload (str): Raw payload string.
            language (Optional[str]): Programming language for syntax highlighting.

        Returns:
            str: Formatted payload string.
        """
        max_ticks = max((len(m.group(0)) for m in re.finditer(r"`+", payload)), default=0)
        max_tildes = max((len(m.group(0)) for m in re.finditer(r"~+", payload)), default=0)
        if max_ticks <= max_tildes:
            fence_char = "`"
            n = max(3, max_ticks + 1)
        else:
            fence_char = "~"
            n = max(3, max_tildes + 1)

        fence = fence_char * n
        info = f"{language.strip()}" if language else ""
        return f"{fence}{info}\n{payload}\n{fence}"

    def format_tool_calls(self, tool_calls: tp.List[tp.Kwargs]) -> tp.Optional[str]:
        """Format tool calls into a string representation.

        Args:
            tool_calls (List[Kwargs]): List of tool call descriptors.

        Returns:
            Optional[str]: Formatted string representation of the tool calls, or None if not displayed.
        """
        if self.tool_display_format.lower() == "none":
            return None

        out = []
        for tc in tool_calls:
            tc = dict(tc)
            tc["name"] = tc.get("name", "unknown")
            arguments = tc.get("arguments", {})
            tool_dump_kwargs = dict(self.tool_dump_kwargs)
            dump_engine = tool_dump_kwargs.pop("dump_engine", None) or "json"
            if not isinstance(arguments, str):
                arguments = dump(arguments, dump_engine=dump_engine, **tool_dump_kwargs)
            dump_language = get_dump_language(dump_engine)

            tc["token_count"] = len(tokenize(arguments))
            if arguments:
                if self.tool_display_format.lower() == "minimal":
                    tc["payload"] = ""
                elif self.tool_display_format.lower() == "compact":
                    head, tail = head_and_tail(arguments)
                    skipped = arguments[len(head) : -len(tail)]
                    n_lines_skipped = len(skipped.splitlines())
                    if skipped:
                        head = self.format_payload(head.strip(), language=dump_language)
                        tail = self.format_payload(tail.strip(), language=dump_language)
                        if n_lines_skipped == 1:
                            middle = f"\n\n*... ({len(skipped)} chars skipped) ...*\n\n"
                        else:
                            middle = f"\n\n*... ({n_lines_skipped} lines skipped) ...*\n\n"
                        tc["payload"] = head + middle + tail
                    else:
                        tc["payload"] = self.format_payload(arguments, language=dump_language)
                elif self.tool_display_format.lower() == "full":
                    tc["payload"] = self.format_payload(arguments, language=dump_language)
                else:
                    raise ValueError(f"Invalid tool display format: {self.tool_display_format!r}")
            else:
                tc["payload"] = ""
            tool_request_template = self.tool_request_template
            if isinstance(tool_request_template, str):
                tool_request_template = SafeSub(tool_request_template)
            elif checks.is_function(tool_request_template):
                tool_request_template = RepFunc(tool_request_template)
            elif not isinstance(tool_request_template, CustomTemplate):
                raise TypeError("Tool request template must be a string, function, or template")
            template_context = flat_merge_dicts(tc, self.template_context)
            tool_request = tool_request_template.substitute(template_context, eval_id="tool_request_template")
            if tool_request:
                out.append(tool_request)
        return self.process_thought(thought="\n\n".join(out))

    def format_tool_results(self, tool_results: tp.List[tp.Kwargs], prepend_newline: bool = True) -> tp.Optional[str]:
        """Format tool results into a string representation.

        Args:
            tool_results (List[Kwargs]): List of tool result descriptors.
            prepend_newline (bool): Flag to prepend a newline to the output.

        Returns:
            Optional[str]: Formatted string representation of the tool results, or None if not displayed.
        """
        if self.tool_display_format.lower() == "none":
            return None

        out = []
        for tr in tool_results:
            tr = dict(tr)
            tr["name"] = tr.get("name", "unknown")
            output = tr.get("output", "")
            tr["token_count"] = len(tokenize(output))
            if output:
                if self.tool_display_format.lower() == "minimal":
                    tr["payload"] = ""
                elif self.tool_display_format.lower() == "compact":
                    head, tail = head_and_tail(output)
                    skipped = output[len(head) : -len(tail)]
                    n_lines_skipped = len(skipped.splitlines())
                    if skipped:
                        head = self.format_payload(head.strip(), language="text")
                        tail = self.format_payload(tail.strip(), language="text")
                        if n_lines_skipped == 1:
                            middle = f"\n\n*... ({len(skipped)} chars skipped) ...*\n\n"
                        else:
                            middle = f"\n\n*... ({n_lines_skipped} lines skipped) ...*\n\n"
                        tr["payload"] = head + middle + tail
                    else:
                        tr["payload"] = self.format_payload(output, language="text")
                elif self.tool_display_format.lower() == "full":
                    tr["payload"] = self.format_payload(output, language="text")
                else:
                    raise ValueError(f"Invalid tool display format: {self.tool_display_format!r}")
            else:
                tr["payload"] = ""
            tool_response_template = self.tool_response_template
            if isinstance(tool_response_template, str):
                tool_response_template = SafeSub(tool_response_template)
            elif checks.is_function(tool_response_template):
                tool_response_template = RepFunc(tool_response_template)
            elif not isinstance(tool_response_template, CustomTemplate):
                raise TypeError("Tool response template must be a string, function, or template")
            template_context = flat_merge_dicts(tr, self.template_context)
            tool_response = tool_response_template.substitute(template_context, eval_id="tool_response_template")
            if tool_response:
                out.append(tool_response)
        if out and prepend_newline:
            out = [""] + out
        return self.process_thought(thought="\n\n".join(out))

    def get_completion(
        self,
        message: str,
        return_response: bool = False,
    ) -> tp.ChatOutput:
        """Return the formatted completion output for a provided message.

        Args:
            message (str): User message to generate a completion for.
            return_response (bool): Flag to return the last raw response along with the file path.

        Returns:
            ChatOutput: File path for the formatted output; if `return_response` is True,
                a tuple containing the file path and last raw response.
        """
        chat_history = self.chat_history
        stream = self.stream
        max_tool_calls = self.max_tool_calls
        formatter = self.formatter
        formatter_kwargs = self.formatter_kwargs
        template_context = self.template_context

        if isinstance(formatter, type):
            formatter_kwargs = dict(formatter_kwargs)
            if "minimal_format" not in formatter_kwargs:
                formatter_kwargs["minimal_format"] = self.minimal_format
            formatter_kwargs["template_context"] = merge_dicts(
                template_context, formatter_kwargs.get("template_context", None)
            )
            if issubclass(formatter, HTMLFileFormatter):
                if "page_title" not in formatter_kwargs:
                    formatter_kwargs["page_title"] = message
                if "cache_dir" not in formatter_kwargs:
                    chat_dir = self.get_setting("chat_dir", default=None)
                    if isinstance(chat_dir, CustomTemplate):
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
                        chat_dir = chat_dir.substitute(template_context, eval_id="chat_dir")
                    chat_dir = Path(chat_dir) / "html"
                    formatter_kwargs["dir_path"] = chat_dir
            formatter = formatter(**formatter_kwargs)
        elif formatter_kwargs:
            formatter = formatter.replace(**formatter_kwargs)
        with formatter:
            self.reset_thought_state()
            messages = self.prepare_messages(message)

            while True:
                enable_tools = max_tool_calls is None or max_tool_calls > 0
                if stream:
                    response = self.get_stream_response(messages, enable_tools=enable_tools)
                    response_chunks = []
                    content_chunks = []
                    for response_chunk in response:
                        new_content = self.get_delta_content(response_chunk)
                        if new_content:
                            formatter.append(new_content)
                            content_chunks.append(new_content)
                        response_chunks.append(response_chunk)
                    if content_chunks:
                        messages.append(dict(role="assistant", content="".join(content_chunks)))
                    tool_calls = self.get_stream_tool_calls(response_chunks)
                else:
                    response = self.get_chat_response(messages, enable_tools=enable_tools)
                    new_content = self.get_message_content(response)
                    if new_content:
                        formatter.append(new_content, complete=True)
                        messages.append(dict(role="assistant", content=new_content))
                    tool_calls = self.get_chat_tool_calls(response)

                if not self.supports_tool_calling or not self.tool_registry:
                    break
                if not tool_calls:
                    break

                tool_calls = self.normalize_tool_calls(tool_calls)
                new_content = self.format_tool_calls(tool_calls)
                if new_content:
                    formatter.append(new_content, complete=True)
                    prepend_newline = True
                else:
                    prepend_newline = False
                tool_results = self.execute_tool_calls(tool_calls)
                new_content = self.format_tool_results(tool_results, prepend_newline=prepend_newline)
                flushed_content = self.flush_thought()
                if flushed_content:
                    if not new_content:
                        new_content = flushed_content
                    else:
                        new_content += flushed_content
                if new_content:
                    formatter.append(new_content, complete=True)
                tool_call_messages = self.get_tool_call_messages(tool_calls)
                if tool_call_messages:
                    messages.extend(tool_call_messages)
                tool_result_messages = self.get_tool_result_messages(tool_results)
                if tool_result_messages:
                    messages.extend(tool_result_messages)
                if max_tool_calls is not None:
                    max_tool_calls -= 1

            content = formatter.content
            flushed_content = self.flush_thought()
            if flushed_content:
                content += flushed_content

        chat_history.append(dict(role="user", content=message))
        chat_history.append(dict(role="assistant", content=content))
        if isinstance(formatter, HTMLFileFormatter) and formatter.file_handle is not None:
            file_path = Path(formatter.file_handle.name)
        else:
            file_path = None
        if return_response:
            return file_path, response
        return file_path

    def get_completion_content(self, message: str) -> str:
        """Return the text content of a completion for a given message.

        Args:
            message (str): User message to complete.

        Returns:
            str: Generated completion text.
        """
        buf = io.StringIO()
        new_formatter = ContentFormatter(output_to=buf)
        new_completions = self.replace(formatter=new_formatter, formatter_kwargs={})
        return new_completions.get_completion(message)


class OpenAICompatibleCompletions(Completions):
    """Base class for OpenAI-compatible completion providers."""

    def get_message_content(self, response: tp.Any) -> tp.Optional[str]:
        choices = get(response, "choices", None) or []
        if choices:
            message = get(choices[0], "message", None)
            return get(message, "content", None)
        message = get(response, "message", None)
        if message:
            return get(message, "content", None)
        return None

    def get_delta_content(self, response_chunk: tp.Any) -> tp.Optional[str]:
        choices = get(response_chunk, "choices", None) or []
        if choices:
            delta = get(choices[0], "delta", None)
            return get(delta, "content", None)
        delta = get(response_chunk, "delta", None)
        if delta:
            return get(delta, "content", None)
        return None

    @property
    def supports_tool_calling(self) -> bool:
        return True

    def function_to_tool_spec(self, func: tp.Callable, *args, **kwargs) -> tp.Kwargs:
        tool_spec = Completions.function_to_tool_spec(self, func, *args, **kwargs)
        tool_spec = {"type": tool_spec.pop("type"), "function": tool_spec}
        return tool_spec

    def get_chat_tool_calls(self, response: tp.Any) -> tp.List[tp.Kwargs]:
        def _iter_messages(obj):
            choices = get(obj, "choices", None)
            if choices:
                for ch in choices or []:
                    msg = get(ch, "message", None)
                    if msg:
                        yield msg
            else:
                msg = get(obj, "message", None) or obj
                if msg:
                    yield msg

        def _extract_from_message(msg, ch_i=0):
            out = []
            for i, tc in enumerate(get(msg, "tool_calls", []) or []):
                fn = get(tc, "function", None)
                out.append(
                    {
                        "id": get(tc, "id", None) or f"tool_call_{ch_i}_{i}",
                        "name": get(fn, "name", None) if fn else None,
                        "arguments": get(fn, "arguments", None) if fn else None,
                        "legacy": False,
                    }
                )
            fc = get(msg, "function_call", None)
            if fc:
                out.append(
                    {
                        "id": get(fc, "id", None) or f"function_call_{ch_i}",
                        "name": get(fc, "name", None),
                        "arguments": get(fc, "arguments", None),
                        "legacy": True,
                    }
                )
            return out

        tool_calls = []
        for ch_i, msg in enumerate(_iter_messages(response)):
            tool_calls.extend(_extract_from_message(msg, ch_i))
        return tool_calls

    def get_stream_tool_calls(self, response_chunks: tp.Iterator[tp.Any]) -> tp.List[tp.Kwargs]:
        tool_call_mapping = {}
        fc_entry = None

        def _iter_deltas(obj):
            choices = get(obj, "choices", None)
            if choices:
                for ch in choices or []:
                    delta = get(ch, "delta", None)
                    if delta:
                        yield delta
            else:
                delta = get(obj, "delta", None)
                if delta:
                    yield delta

        def _alloc_index():
            i = 0
            while i in tool_call_mapping:
                i += 1
            return i

        def _extract_from_delta(delta):
            nonlocal fc_entry

            for tc in get(delta, "tool_calls", []) or []:
                idx = get(tc, "index", None)
                if idx is None:
                    idx = _alloc_index()
                entry = tool_call_mapping.setdefault(
                    idx,
                    {
                        "id": f"tool_call_{idx}",
                        "name": None,
                        "arguments": "",
                        "legacy": False,
                    },
                )
                tc_id = get(tc, "id", None)
                if tc_id:
                    entry["id"] = tc_id
                fn = get(tc, "function", None)
                if fn:
                    name = get(fn, "name", None)
                    if name:
                        entry["name"] = name
                    args = get(fn, "arguments", None)
                    if args:
                        entry["arguments"] += args

            fc = get(delta, "function_call", None)
            if fc:
                if fc_entry is None:
                    fc_entry = {
                        "id": get(fc, "id", None) or f"function_call_{len(tool_call_mapping)}",
                        "name": None,
                        "arguments": "",
                        "legacy": True,
                    }
                name = get(fc, "name", None)
                if name:
                    fc_entry["name"] = name
                args = get(fc, "arguments", None)
                if args:
                    fc_entry["arguments"] += args

        def _merge_final_message(msg):
            nonlocal fc_entry

            for i, tc in enumerate(get(msg, "tool_calls", []) or []):
                idx = get(tc, "index", None)
                if idx is None:
                    idx = i
                entry = tool_call_mapping.setdefault(
                    idx,
                    {
                        "id": f"tool_call_{idx}",
                        "name": None,
                        "arguments": "",
                        "legacy": False,
                    },
                )
                tc_id = get(tc, "id", None)
                if tc_id:
                    entry["id"] = tc_id
                fn = get(tc, "function", None)
                if fn:
                    name = get(fn, "name", None)
                    if name:
                        entry["name"] = name
                    args = get(fn, "arguments", None)
                    if args is not None:
                        entry["arguments"] = args

            fc = get(msg, "function_call", None)
            if fc:
                if fc_entry is None:
                    fc_entry = {
                        "id": get(fc, "id", None) or f"function_call_{len(tool_call_mapping)}",
                        "name": None,
                        "arguments": "",
                        "legacy": True,
                    }
                name = get(fc, "name", None)
                if name:
                    fc_entry["name"] = name
                args = get(fc, "arguments", None)
                if args is not None:
                    fc_entry["arguments"] = args

        for chunk in response_chunks:
            for delta in _iter_deltas(chunk):
                _extract_from_delta(delta)
            msg = get(chunk, "message", None)
            if msg:
                _merge_final_message(msg)

        tool_calls = [tool_call_mapping[i] for i in sorted(tool_call_mapping)]
        if fc_entry:
            tool_calls.append(fc_entry)
        return tool_calls

    @property
    def should_dump_arguments(self) -> bool:
        """Whether to dump tool call arguments to a string before sending.

        Returns:
            bool: True if tool call arguments should be dumped, False otherwise.
        """
        return True

    def get_tool_call_messages(self, tool_calls: tp.List[tp.Kwargs]) -> tp.List[tp.Kwargs]:
        legacy_messages = []
        norm_tool_calls = []
        for tc in tool_calls:
            arguments = tc["arguments"]
            if self.should_dump_arguments:
                arguments = json.dumps(arguments)
            if tc.get("legacy", False):
                legacy_messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "function_call": {
                            "name": tc["name"],
                            "arguments": arguments,
                        },
                    }
                )
            else:
                norm_tool_calls.append(
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": arguments,
                        },
                    }
                )
        if norm_tool_calls:
            new_messages = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": norm_tool_calls,
                }
            ]
        else:
            new_messages = []
        messages = legacy_messages + new_messages
        return messages

    def get_tool_result_messages(self, tool_results: tp.List[tp.Kwargs]) -> tp.List[tp.Kwargs]:
        messages = []
        for tr in tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tr["id"],
                    "name": tr["name"],
                    "content": tr["output"],
                }
            )
        return messages


class OpenAICompletions(OpenAICompatibleCompletions):
    """Completions class for OpenAI.

    !!! info
        For default settings, see `chat.completions_configs.openai` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): Identifier for the model to use.
        strict_schema (Optional[bool]): Whether to enforce strict schema validation.
        use_responses (bool): Whether to use the Responses API instead of the Completions API.

            Note that thought summarization is not supported in the Completions API.
        client_kwargs (KwargsLike): Keyword arguments for `openai.OpenAI`.
        responses_kwargs (KwargsLike): Keyword arguments for `openai.Responses.create`.
        completions_kwargs (KwargsLike): Keyword arguments for `openai.Completions.create`.
        **kwargs: Keyword arguments for `Completions` or used as `client_kwargs`,
            `responses_kwargs`, or `completions_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "openai"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.openai"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        strict_schema: tp.Optional[bool] = None,
        use_responses: tp.Optional[bool] = None,
        client_kwargs: tp.KwargsLike = None,
        responses_kwargs: tp.KwargsLike = None,
        completions_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        OpenAICompatibleCompletions.__init__(
            self,
            model=model,
            strict_schema=strict_schema,
            use_responses=use_responses,
            client_kwargs=client_kwargs,
            responses_kwargs=responses_kwargs,
            completions_kwargs=completions_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("openai")
        from openai import OpenAI

        openai_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = openai_config.pop("model", None)
        def_quick_model = openai_config.pop("quick_model", None)
        def_strict_schema = openai_config.pop("strict_schema", None)
        def_use_responses = openai_config.pop("use_responses", None)
        def_client_kwargs = openai_config.pop("client_kwargs", None)
        def_responses_kwargs = openai_config.pop("responses_kwargs", None)
        def_completions_kwargs = openai_config.pop("completions_kwargs", None)

        if model is None:
            model = def_quick_model if self.quick_mode else def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = self.get_init_arg_names()
        for k in list(openai_config.keys()):
            if k in init_arg_names:
                openai_config.pop(k)

        if strict_schema is None:
            strict_schema = def_strict_schema
        if use_responses is None:
            use_responses = def_use_responses

        client_arg_names = set(get_func_arg_names(OpenAI.__init__))
        _client_kwargs = {}
        _responses_kwargs = {}
        _completions_kwargs = {}
        for k, v in openai_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _responses_kwargs[k] = v
                _completions_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        responses_kwargs = merge_dicts(_responses_kwargs, def_responses_kwargs, responses_kwargs)
        completions_kwargs = merge_dicts(_completions_kwargs, def_completions_kwargs, completions_kwargs)
        client = OpenAI(**client_kwargs)

        self._model = model
        self._client = client
        self._strict_schema = strict_schema
        self._use_responses = use_responses
        self._responses_kwargs = responses_kwargs
        self._completions_kwargs = completions_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> OpenAIT:
        """OpenAI client instance used for API calls.

        Returns:
            OpenAI: OpenAI client instance.
        """
        return self._client

    @property
    def strict_schema(self) -> bool:
        """Whether to enforce strict schema validation.

        Returns:
            bool: True if strict schema validation is enforced, False otherwise.
        """
        return self._strict_schema

    @property
    def use_responses(self) -> bool:
        """Whether to use the Responses API instead of the Completions API.

        Returns:
            bool: True if the Responses API is used, False if the Completions API is used.
        """
        return self._use_responses

    @property
    def responses_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `openai.Responses.create`.

        Returns:
            Kwargs: Keyword arguments for the responses API call.
        """
        return self._responses_kwargs

    @property
    def completions_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `openai.Completions.create`.

        Returns:
            Kwargs: Keyword arguments for the completion API call.
        """
        return self._completions_kwargs

    def format_messages(self, messages: tp.ChatMessages) -> tp.Tuple[tp.List[tp.Dict[str, str]], tp.Optional[str]]:
        """Format messages.

        Args:
            messages (ChatMessages): List representing the conversation history.

        Returns:
            Tuple[List[Dict[str, str]], str]: List of message dictionaries and system instructions.
        """
        new_messages = []
        instructions = []
        for message in messages:
            if self.use_responses and isinstance(message, dict) and message.get("role", "user") == "system":
                instructions.append(message.get("content", ""))
            elif isinstance(message, str):
                new_messages.append(dict(role="user", content=message))
            else:
                new_messages.append(message)
        if instructions:
            instructions = "\n".join(instructions)
        else:
            instructions = None
        return new_messages, instructions

    def format_kwargs(self, messages: tp.ChatMessages, enable_tools: bool = True) -> tp.Kwargs:
        """Format keyword arguments for the API call.

        Args:
            messages (ChatMessages): List representing the conversation history.
            enable_tools (bool): Whether to enable tool usage.

        Returns:
            Kwargs: Keyword arguments for the API call.
        """
        if not enable_tools:
            messages = list(messages)
            messages += [{"role": "user", "content": "Write the final answer without calling tools."}]
        if self.use_responses:
            input, instructions = self.format_messages(messages)
            responses_kwargs = dict(self.responses_kwargs)
            responses_kwargs["tools"] = responses_kwargs.get("tools", []) + self.tools
            for i, tool in enumerate(responses_kwargs["tools"]):
                if isinstance(tool, str) and tool in self._tool_registry:
                    tool_spec = self.function_to_tool_spec(self._tool_registry[tool], name=tool)
                    responses_kwargs["tools"][i] = tool_spec
            if not enable_tools:
                responses_kwargs.pop("tools", None)
                responses_kwargs.pop("tool_choice", None)
            return dict(
                model=self.model,
                instructions=instructions,
                input=input,
                stream=self.stream,
                **responses_kwargs,
            )
        else:
            input, _ = self.format_messages(messages)
            completions_kwargs = dict(self.completions_kwargs)
            completions_kwargs["tools"] = completions_kwargs.get("tools", []) + self.tools
            for i, tool in enumerate(completions_kwargs["tools"]):
                if isinstance(tool, str) and tool in self._tool_registry:
                    tool_spec = self.function_to_tool_spec(self._tool_registry[tool], name=tool)
                    completions_kwargs["tools"][i] = tool_spec
            if not enable_tools:
                completions_kwargs.pop("tools", None)
                completions_kwargs.pop("tool_choice", None)
            return dict(
                messages=messages,
                model=self.model,
                stream=self.stream,
                **completions_kwargs,
            )

    def get_chat_response(self, messages: tp.ChatMessages, enable_tools: bool = True) -> ChatCompletionT:
        if self.use_responses:
            return self.client.responses.create(**self.format_kwargs(messages, enable_tools=enable_tools))
        else:
            return self.client.chat.completions.create(**self.format_kwargs(messages, enable_tools=enable_tools))

    def get_message_content(self, response: tp.Union[ChatCompletionT, ResponseT]) -> tp.Optional[str]:
        if self.use_responses:
            out = None
            for output in get(response, "output", []) or []:
                if get(output, "type", None) == "reasoning":
                    for summary in get(output, "summary", []) or []:
                        if get(summary, "type", None) == "summary_text":
                            thought = self.process_thought(thought=get(summary, "text", None), flush=True)
                            if thought is not None:
                                if out is None:
                                    out = ""
                                out += thought
                if get(output, "type", None) == "message":
                    for content in get(output, "content", []) or []:
                        if get(content, "type", None) == "output_text":
                            text = self.process_thought(content=get(content, "text", None), flush=True)
                            if text is not None:
                                if out is None:
                                    out = ""
                                out += text
            return out
        else:
            return OpenAICompatibleCompletions.get_message_content(self, response)

    def get_stream_response(self, messages: tp.ChatMessages, enable_tools: bool = True) -> StreamT:
        if self.use_responses:
            return self.client.responses.create(**self.format_kwargs(messages, enable_tools=enable_tools))
        else:
            return self.client.chat.completions.create(**self.format_kwargs(messages, enable_tools=enable_tools))

    def get_delta_content(
        self,
        response_chunk: tp.Union[ChatCompletionChunkT, ResponseStreamEventT],
    ) -> tp.Optional[str]:
        if self.use_responses:
            if get(response_chunk, "type", None) == "response.reasoning_summary_text.delta":
                return self.process_thought(thought=get(response_chunk, "delta", None))
            if get(response_chunk, "type", None) == "response.output_text.delta":
                return self.process_thought(content=get(response_chunk, "delta", None))
            return self.flush_thought()
        else:
            return OpenAICompatibleCompletions.get_delta_content(self, response_chunk)

    def function_to_tool_spec(self, func: tp.Callable, *args, **kwargs) -> tp.Kwargs:
        if self.use_responses:
            return Completions.function_to_tool_spec(self, func, *args, strict=self.strict_schema, **kwargs)
        else:
            return OpenAICompatibleCompletions.function_to_tool_spec(self, func, *args, **kwargs)

    def get_chat_tool_calls(self, response: tp.Union[ChatCompletionT, ResponseT]) -> tp.List[tp.Kwargs]:
        if self.use_responses:
            tool_calls = []
            output = get(response, "output", None) or []
            for idx, item in enumerate(output):
                if get(item, "type", None) == "function_call":
                    fc = get(item, "function_call", None) or item
                    tool_calls.append(
                        {
                            "id": get(fc, "call_id", None) or get(fc, "id", None) or f"function_call_{idx}",
                            "name": get(fc, "name", None),
                            "arguments": get(fc, "parsed_arguments", None) or get(fc, "arguments", None),
                        }
                    )
            return tool_calls
        else:
            return OpenAICompatibleCompletions.get_chat_tool_calls(self, response)

    def get_stream_tool_calls(
        self,
        response_chunks: tp.Iterator[tp.Union[ChatCompletionChunkT, ResponseStreamEventT]],
    ) -> tp.List[tp.Kwargs]:
        if self.use_responses:
            tool_call_mapping = {}
            for response_chunk in response_chunks:
                output_index = get(response_chunk, "output_index", None)
                if output_index is not None:
                    if get(response_chunk, "type", None) == "response.output_item.added":
                        item = get(response_chunk, "item", None)
                        if item:
                            if get(item, "type", None) == "function_call":
                                fc = get(item, "function_call", None) or item
                                tool_call_mapping[output_index] = {
                                    "id": get(fc, "call_id", None)
                                    or get(fc, "id", None)
                                    or f"function_call_{len(tool_call_mapping)}",
                                    "name": get(fc, "name", None),
                                    "arguments": "",
                                }
                    elif get(response_chunk, "type", None) == "response.function_call_arguments.delta":
                        delta = get(response_chunk, "delta", None)
                        if delta:
                            tool_call_mapping[output_index]["arguments"] += delta
            return [tool_call_mapping[i] for i in sorted(tool_call_mapping)]
        else:
            return OpenAICompatibleCompletions.get_stream_tool_calls(self, response_chunks)

    def get_tool_call_messages(self, tool_calls: tp.List[tp.Kwargs]) -> tp.List[tp.Kwargs]:
        if self.use_responses:
            messages = []
            for tc in tool_calls:
                messages.append(
                    {
                        "type": "function_call",
                        "call_id": tc["id"],
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    }
                )
            return messages
        else:
            return OpenAICompatibleCompletions.get_tool_call_messages(self, tool_calls)

    def get_tool_result_messages(self, tool_results: tp.List[tp.Kwargs]) -> tp.List[tp.Kwargs]:
        if self.use_responses:
            messages = []
            for tr in tool_results:
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tr["id"],
                        "output": tr["output"],
                    }
                )
            return messages
        else:
            return OpenAICompatibleCompletions.get_tool_result_messages(self, tool_results)


class AnthropicCompletions(Completions):
    """Completions class for Anthropic (Claude).

    !!! info
        For default settings, see `chat.completions_configs.anthropic` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): Anthropic model identifier.
        client_type (Union[None, str, type]): Anthropic client type.

            Supported values:

            * "anthropic": `anthropic.Anthropic`
            * "bedrock": `anthropic.AnthropicBedrock`
            * "vertex": `anthropic.AnthropicVertex`
            * type: Custom Anthropic client class
        client_kwargs (KwargsLike): Keyword arguments for `client_type`
        messages_kwargs (KwargsLike): Keyword arguments for `anthropic.Client.messages.create`.
        **kwargs: Keyword arguments for `Completions` or used as `client_kwargs` or `messages_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "anthropic"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.anthropic"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_type: tp.Union[None, str, type] = None,
        client_kwargs: tp.KwargsLike = None,
        messages_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Completions.__init__(
            self,
            model=model,
            client_type=client_type,
            client_kwargs=client_kwargs,
            messages_kwargs=messages_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("anthropic")

        anthropic_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = anthropic_config.pop("model", None)
        def_client_type = anthropic_config.pop("client_type", None)
        def_quick_model = anthropic_config.pop("quick_model", None)
        def_client_kwargs = anthropic_config.pop("client_kwargs", None)
        def_messages_kwargs = anthropic_config.pop("messages_kwargs", None)

        if model is None:
            model = def_quick_model if self.quick_mode else def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = self.get_init_arg_names()
        for k in list(anthropic_config.keys()):
            if k in init_arg_names:
                anthropic_config.pop(k)

        if client_type is None:
            client_type = def_client_type
        if isinstance(client_type, str) and client_type.lower() == "anthropic":
            from anthropic import Anthropic

            client_type = Anthropic
        elif isinstance(client_type, str) and client_type.lower() == "bedrock":
            from anthropic import AnthropicBedrock

            client_type = AnthropicBedrock
        elif isinstance(client_type, str) and client_type.lower() == "vertex":
            from anthropic import AnthropicVertex

            client_type = AnthropicVertex
        elif not isinstance(client_type, type):
            raise ValueError(f"Invalid client_type: {client_type!r}")

        client_arg_names = set(get_func_arg_names(client_type.__init__))
        _client_kwargs = {}
        _messages_kwargs = {}
        for k, v in anthropic_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _messages_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        messages_kwargs = merge_dicts(_messages_kwargs, def_messages_kwargs, messages_kwargs)

        client = client_type(**client_kwargs)

        self._model = model
        self._client = client
        self._messages_kwargs = messages_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> AnthropicClientT:
        """Anthropic client instance.

        Returns:
            Anthropic: Anthropic client instance.
        """
        return self._client

    @property
    def messages_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `anthropic.Client.messages.create`.

        Returns:
            Kwargs: Keyword arguments for message creation.
        """
        return self._messages_kwargs

    def format_messages(self, messages: tp.ChatMessages) -> tp.Tuple[tp.List[tp.Dict[str, str]], tp.Optional[str]]:
        """Format messages to Anthropic format.

        Args:
            messages (ChatMessages): List representing the conversation history.

        Returns:
            Tuple[List[Dict[str, str]], str]: List of message dictionaries and system message.
        """
        new_messages = []
        system_messages = []
        for message in messages:
            if isinstance(message, dict) and message.get("role", "user") == "system":
                system_messages.append(message.get("content", ""))
            elif isinstance(message, str):
                new_messages.append(dict(role="user", content=message))
            else:
                new_messages.append(message)
        if system_messages:
            system_message = "\n".join(system_messages)
        else:
            system_message = None
        return new_messages, system_message

    def format_kwargs(self, messages: tp.ChatMessages, enable_tools: bool = True) -> tp.Kwargs:
        """Format keyword arguments for the API call.

        Args:
            messages (ChatMessages): List representing the conversation history.
            enable_tools (bool): Whether to enable tool usage.

        Returns:
            Kwargs: Keyword arguments for the API call.
        """
        if not enable_tools:
            messages = list(messages)
            messages += [{"role": "user", "content": "Write the final answer without calling tools."}]
        formatted_messages, system_message = self.format_messages(messages)
        messages_kwargs = dict(self.messages_kwargs)
        if system_message:
            messages_kwargs["system"] = system_message
        messages_kwargs["tools"] = messages_kwargs.get("tools", []) + self.tools
        for i, tool in enumerate(messages_kwargs["tools"]):
            if isinstance(tool, str) and tool in self._tool_registry:
                messages_kwargs["tools"][i] = self.function_to_tool_spec(self._tool_registry[tool], name=tool)
        if not enable_tools:
            messages_kwargs.pop("tools", None)
            messages_kwargs.pop("tool_choice", None)
        return dict(
            model=self.model,
            messages=formatted_messages,
            stream=self.stream,
            **messages_kwargs,
        )

    def get_chat_response(self, messages: tp.ChatMessages, enable_tools: bool = True) -> AnthropicMessageT:
        return self.client.messages.create(**self.format_kwargs(messages, enable_tools=enable_tools))

    def get_message_content(self, response: AnthropicMessageT) -> tp.Optional[str]:
        from anthropic.types import ThinkingBlock, TextBlock

        content = None
        for block in response.content:
            if isinstance(block, ThinkingBlock):
                thinking = self.process_thought(thought=block.thinking, flush=True)
                if thinking is not None:
                    if content is None:
                        content = ""
                    content += thinking
            elif isinstance(block, TextBlock):
                text = self.process_thought(content=block.text, flush=True)
                if text is not None:
                    if content is None:
                        content = ""
                    content += text
            else:
                out = self.flush_thought()
                if out is not None:
                    if content is None:
                        content = ""
                    content += out
        return content

    def get_stream_response(self, messages: tp.ChatMessages, enable_tools: bool = True) -> AnthropicStreamT:
        return self.client.messages.create(**self.format_kwargs(messages, enable_tools=enable_tools))

    def get_delta_content(self, response_chunk: AnthropicMessageStreamEventT) -> tp.Optional[str]:
        from anthropic.types import RawContentBlockDeltaEvent, ThinkingDelta, TextDelta

        if isinstance(response_chunk, RawContentBlockDeltaEvent) and isinstance(response_chunk.delta, ThinkingDelta):
            return self.process_thought(thought=response_chunk.delta.thinking)
        if isinstance(response_chunk, RawContentBlockDeltaEvent) and isinstance(response_chunk.delta, TextDelta):
            return self.process_thought(content=response_chunk.delta.text)
        return self.flush_thought()

    @property
    def supports_tool_calling(self) -> bool:
        return True

    def function_to_tool_spec(self, func: tp.Callable, *args, **kwargs) -> tp.Kwargs:
        tool_spec = Completions.function_to_tool_spec(self, func, *args, **kwargs)
        return {
            "name": tool_spec["name"],
            "description": tool_spec["description"],
            "input_schema": tool_spec["parameters"],
        }

    def get_chat_tool_calls(self, response: AnthropicMessageT) -> tp.List[tp.Kwargs]:
        from anthropic.types import ToolUseBlock

        tool_calls = []
        for block in response.content:
            if isinstance(block, ToolUseBlock):
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    }
                )
        return tool_calls

    def get_stream_tool_calls(self, response_chunks: tp.Iterator[AnthropicMessageStreamEventT]) -> tp.List[tp.Kwargs]:
        from anthropic.types import (
            RawMessageStartEvent,
            RawContentBlockStartEvent,
            RawContentBlockDeltaEvent,
            RawContentBlockStopEvent,
            ToolUseBlock,
            InputJSONDelta,
        )

        results = []
        active = {}
        for response_chunk in response_chunks:
            if isinstance(response_chunk, RawMessageStartEvent):
                active.clear()
                continue
            if isinstance(response_chunk, RawContentBlockStartEvent):
                cb = response_chunk.content_block
                if isinstance(cb, ToolUseBlock):
                    active[response_chunk.index] = {
                        "id": cb.id,
                        "name": cb.name,
                        "arguments": "",
                    }
                continue
            if isinstance(response_chunk, RawContentBlockDeltaEvent):
                d = response_chunk.delta
                if isinstance(d, InputJSONDelta):
                    st = active.get(response_chunk.index)
                    if st is not None:
                        st["arguments"] += d.partial_json
                continue
            if isinstance(response_chunk, RawContentBlockStopEvent):
                st = active.pop(response_chunk.index, None)
                if st is not None:
                    results.append(st)
                continue
        return results

    def get_tool_call_messages(self, tool_calls: tp.List[tp.Kwargs]) -> tp.List[tp.Kwargs]:
        content_blocks = []
        for tc in tool_calls:
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["arguments"],
                }
            )
        return [
            {
                "role": "assistant",
                "content": content_blocks,
            }
        ]

    def get_tool_result_messages(self, tool_results: tp.List[tp.Kwargs]) -> tp.List[tp.Kwargs]:
        content_blocks = []
        for tr in tool_results:
            content_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tr["id"],
                    "content": tr["output"],
                }
            )
        return [
            {
                "role": "user",
                "content": content_blocks,
            }
        ]


class GeminiCompletions(Completions):
    """Completions class for Google GenAI (Gemini).

    !!! info
        For default settings, see `chat.completions_configs.gemini` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): Gemini model identifier.
        client_kwargs (KwargsLike): Keyword arguments for `google.genai.Client`.
        completions_kwargs (KwargsLike): Keyword arguments for `google.genai.Client.models.generate_content`.
        **kwargs: Keyword arguments for `Completions` or used as `client_kwargs` or `completions_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "gemini"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.gemini"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        completions_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Completions.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            completions_kwargs=completions_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("google.genai")
        from google.genai import Client

        gemini_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = gemini_config.pop("model", None)
        def_quick_model = gemini_config.pop("quick_model", None)
        def_client_kwargs = gemini_config.pop("client_kwargs", None)
        def_completions_kwargs = gemini_config.pop("completions_kwargs", None)

        if model is None:
            model = def_quick_model if self.quick_mode else def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = self.get_init_arg_names()
        for k in list(gemini_config.keys()):
            if k in init_arg_names:
                gemini_config.pop(k)

        client_arg_names = set(get_func_arg_names(Client.__init__))
        _client_kwargs = {}
        _completions_kwargs = {}
        for k, v in gemini_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _completions_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        completions_kwargs = merge_dicts(_completions_kwargs, def_completions_kwargs, completions_kwargs)

        client = Client(**client_kwargs)

        self._model = model
        self._client = client
        self._completions_kwargs = completions_kwargs

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
    def completions_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `google.genai.Client.models.generate_content`.

        Returns:
            Kwargs: Keyword arguments for content generation.
        """
        return self._completions_kwargs

    def format_messages(self, messages: tp.ChatMessages) -> tp.Tuple[tp.List[ContentT], tp.List[str]]:
        """Format messages to Gemini format.

        Args:
            messages (ChatMessages): List representing the conversation history.

        Returns:
            Union[List[Content], List[str]]: List of `google.genai.types.Content` objects and system instructions.
        """
        from google.genai.types import Content, Part

        contents = []
        system_instruction = []
        for message in messages:
            if isinstance(message, dict) and message.get("role", "user") == "system":
                system_instruction.append(message.get("content", ""))
            elif isinstance(message, dict) and message.get("role", "user") == "assistant":
                content = Content(role="model", parts=[Part.from_text(text=message.get("content", ""))])
                contents.append(content)
            elif isinstance(message, dict) and message.get("role", "user") == "user":
                content = Content(role="user", parts=[Part.from_text(text=message.get("content", ""))])
                contents.append(content)
            elif isinstance(message, str):
                content = Content(role="user", parts=[Part.from_text(text=message)])
                contents.append(content)
            else:
                contents.append(message)
        return contents, system_instruction

    def format_kwargs(self, messages: tp.ChatMessages, enable_tools: bool = True) -> tp.Kwargs:
        """Format keyword arguments for the API call.

        Args:
            messages (ChatMessages): List representing the conversation history.
            enable_tools (bool): Whether to enable tool usage.

        Returns:
            Kwargs: Keyword arguments for the API call.
        """
        from google.genai.types import GenerateContentConfig, AutomaticFunctionCallingConfig, Tool

        if not enable_tools:
            messages = list(messages)
            messages += [{"role": "user", "content": "Write the final answer without calling tools."}]
        formatted_messages, system_instruction = self.format_messages(messages)
        completions_kwargs = dict(self.completions_kwargs)
        config = dict(completions_kwargs.pop("config", {}))
        if system_instruction:
            config["system_instruction"] = system_instruction
        if "automatic_function_calling" not in config:
            config["automatic_function_calling"] = AutomaticFunctionCallingConfig(disable=True)
        tools = config.get("tools", [])
        if not isinstance(tools, list):
            tools = [tools]
        tools = tools + self.tools
        for i, tool in enumerate(tools):
            if isinstance(tool, str) and tool in self._tool_registry:
                tool = self.function_to_tool_spec(self._tool_registry[tool], name=tool)
            if not isinstance(tool, Tool):
                tool = Tool(function_declarations=[tool])
            tools[i] = tool
        config["tools"] = tools
        if not enable_tools:
            config.pop("tools", None)
            config.pop("tool_config", None)
        return dict(
            model=self.model,
            contents=formatted_messages,
            config=GenerateContentConfig(**config),
            **completions_kwargs,
        )

    def get_chat_response(self, messages: tp.ChatMessages, enable_tools: bool = True) -> GenerateContentResponseT:
        from google.genai.errors import ClientError

        kwargs = self.format_kwargs(messages, enable_tools=enable_tools)
        attempted = False
        while True:
            try:
                return self.client.models.generate_content(**kwargs)
            except ClientError as e:
                if e.code == 429 and not attempted:
                    time.sleep(60)
                    attempted = True
                else:
                    raise e

    def get_message_content(self, response: GenerateContentResponseT) -> tp.Optional[str]:
        content = None
        for part in response.candidates[0].content.parts:
            if get(part, "thought", False):
                text = self.process_thought(thought=part.text, flush=True)
                if text is not None:
                    if content is None:
                        content = ""
                    content += text
            else:
                text = self.process_thought(content=part.text, flush=True)
                if text is not None:
                    if content is None:
                        content = ""
                    content += text
        return content

    def get_stream_response(
        self,
        messages: tp.ChatMessages,
        enable_tools: bool = True,
    ) -> tp.Iterator[GenerateContentResponseT]:
        from google.genai.errors import ClientError

        kwargs = self.format_kwargs(messages, enable_tools=enable_tools)
        attempted = False
        while True:
            try:
                return self.client.models.generate_content_stream(**kwargs)
            except ClientError as e:
                if e.code == 429 and not attempted:
                    time.sleep(60)
                    attempted = True
                else:
                    raise e

    def get_delta_content(self, response_chunk: GenerateContentResponseT) -> tp.Optional[str]:
        content = None
        for part in response_chunk.candidates[0].content.parts:
            if get(part, "thought", False):
                text = self.process_thought(thought=part.text)
                if text is not None:
                    if content is None:
                        content = ""
                    content += text
            else:
                text = self.process_thought(content=part.text)
                if text is not None:
                    if content is None:
                        content = ""
                    content += text
        return content

    @property
    def supports_tool_calling(self) -> bool:
        return True

    def function_to_tool_spec(self, func: tp.Callable, *args, **kwargs) -> tp.Kwargs:
        from google.genai.types import FunctionDeclaration

        return FunctionDeclaration.from_callable(client=self.client, callable=func)

    def get_chat_tool_calls(self, response: GenerateContentResponseT) -> tp.List[tp.Kwargs]:
        tool_calls = []
        for fc in get(response, "function_calls", []) or []:
            name = get(fc, "name", None)
            args = get(fc, "args", None)
            if args is None and get(fc, "function_call", None):
                name = name or get(fc.function_call, "name", None)
                args = get(fc.function_call, "args", None)
            tool_calls.append(
                {
                    "id": f"function_call_{len(tool_calls)}",
                    "name": name,
                    "arguments": args,
                }
            )
        return tool_calls

    def get_stream_tool_calls(self, response_chunks: tp.Iterator[GenerateContentResponseT]) -> tp.List[tp.Kwargs]:
        tool_calls = []
        for response_chunk in response_chunks:
            for fc in get(response_chunk, "function_calls", []) or []:
                name = get(fc, "name", None)
                args = get(fc, "args", None)
                if args is None and get(fc, "function_call", None):
                    name = name or get(fc.function_call, "name", None)
                    args = get(fc.function_call, "args", None)
                tool_calls.append(
                    {
                        "id": f"function_call_{len(tool_calls)}",
                        "name": name,
                        "arguments": args,
                    }
                )
        return tool_calls

    def get_tool_call_messages(self, tool_calls: tp.List[tp.Kwargs]) -> tp.List[tp.Kwargs]:
        from google.genai.types import Content, Part

        parts = []
        for tc in tool_calls:
            parts.append(
                Part.from_function_call(
                    name=tc["name"],
                    args=tc["arguments"],
                )
            )
        return [Content(role="model", parts=parts)]

    def get_tool_result_messages(self, tool_results: tp.List[tp.Kwargs]) -> tp.List[tp.Kwargs]:
        from google.genai.types import Content, Part

        parts = []
        for tr in tool_results:
            parts.append(
                Part.from_function_response(
                    name=tr["name"],
                    response=dict(output=tr["output"]),
                )
            )
        return [Content(role="tool", parts=parts)]


class HFInferenceCompletions(OpenAICompatibleCompletions):
    """Completions class for HuggingFace Inference.

    !!! info
        For default settings, see `chat.completions_configs.hf_inference` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): HuggingFace model identifier.
        client_kwargs (KwargsLike): Keyword arguments for `huggingface_hub.InferenceClient`.
        chat_completion_kwargs (KwargsLike): Keyword arguments for `huggingface_hub.InferenceClient.chat_completion`.
        **kwargs: Keyword arguments for `Completions` or used as `client_kwargs` or `chat_completion_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "hf_inference"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.hf_inference"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        chat_completion_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        OpenAICompatibleCompletions.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            chat_completion_kwargs=chat_completion_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("huggingface_hub")
        from huggingface_hub import InferenceClient

        hf_inference_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = hf_inference_config.pop("model", None)
        def_quick_model = hf_inference_config.pop("quick_model", None)
        def_client_kwargs = hf_inference_config.pop("client_kwargs", None)
        def_chat_completion_kwargs = hf_inference_config.pop("chat_completion_kwargs", None)

        if model is None:
            model = def_quick_model if self.quick_mode else def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = self.get_init_arg_names()
        for k in list(hf_inference_config.keys()):
            if k in init_arg_names:
                hf_inference_config.pop(k)

        client_arg_names = set(get_func_arg_names(InferenceClient.__init__))
        _client_kwargs = {}
        _chat_completion_kwargs = {}
        for k, v in hf_inference_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _chat_completion_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        chat_completion_kwargs = merge_dicts(
            _chat_completion_kwargs, def_chat_completion_kwargs, chat_completion_kwargs
        )
        client = InferenceClient(model=model, **client_kwargs)

        self._model = model
        self._client = client
        self._chat_completion_kwargs = chat_completion_kwargs

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
    def chat_completion_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `huggingface_hub.InferenceClient.chat_completion`.

        Returns:
            Kwargs: Keyword arguments for chat completion.
        """
        return self._chat_completion_kwargs

    def format_kwargs(self, messages: tp.ChatMessages, enable_tools: bool = True) -> tp.Kwargs:
        """Format keyword arguments for the API call.

        Args:
            messages (ChatMessages): List representing the conversation history.
            enable_tools (bool): Whether to enable tool usage.

        Returns:
            Kwargs: Keyword arguments for the API call.
        """
        if not enable_tools:
            messages = list(messages)
            messages += [{"role": "user", "content": "Write the final answer without calling tools."}]
        chat_completion_kwargs = dict(self.chat_completion_kwargs)
        chat_completion_kwargs["tools"] = chat_completion_kwargs.get("tools", []) + self.tools
        for i, tool in enumerate(chat_completion_kwargs["tools"]):
            if isinstance(tool, str) and tool in self._tool_registry:
                chat_completion_kwargs["tools"][i] = self.function_to_tool_spec(self._tool_registry[tool], name=tool)
        if not enable_tools:
            chat_completion_kwargs.pop("tools", None)
            chat_completion_kwargs.pop("tool_choice", None)
        return dict(
            messages=messages,
            model=self.model,
            stream=self.stream,
            **chat_completion_kwargs,
        )

    def get_chat_response(self, messages: tp.ChatMessages, enable_tools: bool = True) -> ChatCompletionOutputT:
        return self.client.chat_completion(**self.format_kwargs(messages, enable_tools=enable_tools))

    def get_message_content(self, response: ChatCompletionOutputT) -> tp.Optional[str]:
        message = response.choices[0].message
        if has(message, "thinking"):
            thought = get(message, "thinking")
        elif has(message, "reasoning"):
            thought = get(message, "reasoning")
        elif has(message, "reasoning_content"):
            thought = get(message, "reasoning_content")
        else:
            thought = None
        content = message.content
        return self.process_thought(thought=thought, content=content, flush=True)

    def get_stream_response(self, messages: tp.ChatMessages, enable_tools: bool = True) -> ChatCompletionStreamOutputT:
        return self.client.chat_completion(**self.format_kwargs(messages, enable_tools=enable_tools))

    def get_delta_content(self, response_chunk: ChatCompletionStreamOutputT) -> tp.Optional[str]:
        delta = response_chunk.choices[0].delta
        if has(delta, "thinking"):
            thought = get(delta, "thinking")
        elif has(delta, "reasoning"):
            thought = get(delta, "reasoning")
        elif has(delta, "reasoning_content"):
            thought = get(delta, "reasoning_content")
        else:
            thought = None
        content = delta.content
        return self.process_thought(thought=thought, content=content)


class LiteLLMCompletions(OpenAICompatibleCompletions):
    """Completions class for LiteLLM.

    !!! info
        For default settings, see `chat.completions_configs.litellm` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): Identifier for the model to use.
        completion_kwargs (KwargsLike): Keyword arguments for `litellm.completion`.
        **kwargs: Keyword arguments for `Completions` or used as `completion_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "litellm"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.litellm"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        completion_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        OpenAICompatibleCompletions.__init__(
            self,
            model=model,
            completion_kwargs=completion_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("litellm")

        super_arg_names = self.get_init_arg_names()
        for k in list(kwargs.keys()):
            if k in super_arg_names:
                kwargs.pop(k)
        litellm_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = litellm_config.pop("model", None)
        def_quick_model = litellm_config.pop("quick_model", None)
        def_completion_kwargs = litellm_config.pop("completion_kwargs", None)

        if model is None:
            model = def_quick_model if self.quick_mode else def_model
        if model is None:
            raise ValueError("Must provide a model")
        completion_kwargs = merge_dicts(litellm_config, def_completion_kwargs, completion_kwargs)

        self._model = model
        self._completion_kwargs = completion_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def completion_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `litellm.completion`.

        Returns:
            Kwargs: Keyword arguments for the completion API call.
        """
        return self._completion_kwargs

    def format_kwargs(self, messages: tp.ChatMessages, enable_tools: bool = True) -> tp.Kwargs:
        """Format keyword arguments for the API call.

        Args:
            messages (ChatMessages): List representing the conversation history.
            enable_tools (bool): Whether to enable tool usage.

        Returns:
            Kwargs: Keyword arguments for the API call.
        """
        if not enable_tools:
            messages = list(messages)
            messages += [{"role": "user", "content": "Write the final answer without calling tools."}]
        completion_kwargs = dict(self.completion_kwargs)
        completion_kwargs["tools"] = completion_kwargs.get("tools", []) + self.tools
        for i, tool in enumerate(completion_kwargs["tools"]):
            if isinstance(tool, str) and tool in self._tool_registry:
                completion_kwargs["tools"][i] = self.function_to_tool_spec(self._tool_registry[tool], name=tool)
        if not enable_tools:
            completion_kwargs.pop("tools", None)
            completion_kwargs.pop("tool_choice", None)
        return dict(
            messages=messages,
            model=self.model,
            stream=self.stream,
            **completion_kwargs,
        )

    def get_chat_response(self, messages: tp.ChatMessages, enable_tools: bool = True) -> ModelResponseT:
        from litellm import completion

        return completion(**self.format_kwargs(messages, enable_tools=enable_tools))

    def get_message_content(self, response: ModelResponseT) -> tp.Optional[str]:
        message = response.choices[0].message
        reasoning_content = get(message, "reasoning_content", None)
        return self.process_thought(thought=reasoning_content, content=message.content, flush=True)

    def get_stream_response(self, messages: tp.ChatMessages, enable_tools: bool = True) -> CustomStreamWrapperT:
        from litellm import completion

        return completion(**self.format_kwargs(messages, enable_tools=enable_tools))

    def get_delta_content(self, response_chunk: ModelResponseT) -> tp.Optional[str]:
        delta = response_chunk.choices[0].delta
        reasoning_content = get(delta, "reasoning_content", None)
        return self.process_thought(thought=reasoning_content, content=delta.content)


class LlamaIndexCompletions(Completions):
    """Completions class for LlamaIndex.

    LLM can be provided via `llm`, which can be either the name of the class (case doesn't matter),
    the path or its suffix to the class (case matters), or a subclass or an instance of
    `llama_index.core.llms.LLM`.

    !!! info
        For default settings, see `chat.completions_configs.llama_index` in `vectorbtpro._settings.knowledge`.

    Args:
        llm (Union[None, str, MaybeType[LLM]]): Identifier, class path, subclass, or instance of
            `llama_index.core.llms.LLM`.
        llm_kwargs (KwargsLike): Additional parameters for LLM initialization.
        **kwargs: Keyword arguments for `Completions` or used as `llm_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "llama_index"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.llama_index"

    def __init__(
        self,
        llm: tp.Union[None, str, tp.MaybeType[LLMT]] = None,
        llm_kwargs: tp.KwargsLike = None,
        chat_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Completions.__init__(
            self,
            llm=llm,
            llm_kwargs=llm_kwargs,
            chat_kwargs=chat_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("llama_index")
        from llama_index.core.llms import LLM

        llama_index_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_llm = llama_index_config.pop("llm", None)
        def_llm_kwargs = llama_index_config.pop("llm_kwargs", None)
        def_chat_kwargs = llama_index_config.pop("chat_kwargs", None)

        if llm is None:
            llm = def_llm
        if llm is None:
            raise ValueError("Must provide an LLM name or path")
        init_arg_names = self.get_init_arg_names()
        for k in list(llama_index_config.keys()):
            if k in init_arg_names:
                llama_index_config.pop(k)

        if isinstance(llm, str):
            import llama_index.llms
            from vectorbtpro.utils.module_ import search_package

            def _match_func(k, v):
                if isinstance(v, type) and issubclass(v, LLM):
                    if "." in llm:
                        if k.endswith(llm):
                            return True
                    else:
                        if k.split(".")[-1].lower() == llm.lower():
                            return True
                        if k.split(".")[-1].replace("LLM", "").lower() == llm.lower().replace("_", ""):
                            return True
                return False

            found_llm = search_package(
                llama_index.llms,
                _match_func,
                path_attrs=True,
                return_first=True,
            )
            if found_llm is None:
                raise ValueError(f"LLM {llm!r} not found")
            llm = found_llm
        if isinstance(llm, type):
            checks.assert_subclass_of(llm, LLM, arg_name="llm")
            llm_name = llm.__name__.replace("LLM", "").lower()
            module_name = llm.__module__
        else:
            checks.assert_instance_of(llm, LLM, arg_name="llm")
            llm_name = type(llm).__name__.replace("LLM", "").lower()
            module_name = type(llm).__module__
        llm_configs = llama_index_config.pop("llm_configs", {})
        if llm_name in llm_configs:
            llama_index_config = merge_dicts(llama_index_config, llm_configs[llm_name])
        elif module_name in llm_configs:
            llama_index_config = merge_dicts(llama_index_config, llm_configs[module_name])
        llm_kwargs = merge_dicts(llama_index_config, def_llm_kwargs, llm_kwargs)
        def_model = llm_kwargs.pop("model", None)
        quick_model = llm_kwargs.pop("quick_model", None)
        model = quick_model if self.quick_mode else def_model
        if model is None:
            func_kwargs = get_func_kwargs(type(llm).__init__)
            model = func_kwargs.get("model", None)
        else:
            llm_kwargs["model"] = model
        if isinstance(llm, type):
            llm = llm(**llm_kwargs)
        elif len(kwargs) > 0:
            raise ValueError("Cannot apply config to already initialized LLM")

        chat_kwargs = merge_dicts(def_chat_kwargs, chat_kwargs)

        self._model = model
        self._llm = llm
        self._chat_kwargs = chat_kwargs

    @property
    def model(self) -> tp.Optional[str]:
        return self._model

    @property
    def llm(self) -> LLMT:
        """Initialized LLM instance used for generating completions.

        Returns:
            LLM: Initialized LLM instance.
        """
        return self._llm

    @property
    def chat_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for LLM chat methods.

        Returns:
            Kwargs: Keyword arguments for chat methods.
        """
        return self._chat_kwargs

    def format_messages(self, messages: tp.ChatMessages, enable_tools: bool = True) -> tp.List[ChatMessageT]:
        """Format messages.

        Args:
            messages (ChatMessages): List representing the conversation history.
            enable_tools (bool): Whether to enable tool usage.

        Returns:
            List[ChatMessage]: List of chat messages.
        """
        from llama_index.core.llms import ChatMessage

        new_messages = []
        for message in messages:
            if isinstance(message, dict):
                new_messages.append(ChatMessage(**message))
            elif isinstance(message, str):
                new_messages.append(ChatMessage(role="user", content=message))
            else:
                new_messages.append(message)
        return new_messages

    def get_chat_response(self, messages: tp.ChatMessages, enable_tools: bool = True) -> ChatResponseT:
        from llama_index.core.tools import FunctionTool

        if not enable_tools:
            messages = list(messages)
            messages += [{"role": "user", "content": "Write the final answer without calling tools."}]
        messages = self.format_messages(messages, enable_tools=enable_tools)
        tools = self.tools
        if enable_tools and tools:
            fns = [self.tool_registry[tool] for tool in tools]
            tools = [FunctionTool.from_defaults(fn=fn) for fn in fns]
            return self.llm.chat_with_tools(tools, chat_history=messages, **self.chat_kwargs)
        else:
            return self.llm.chat(messages, **self.chat_kwargs)

    def get_message_content(self, response: ChatResponseT) -> tp.Optional[str]:
        return self.process_thought(content=response.message.content, flush=True)

    def get_stream_response(self, messages: tp.ChatMessages, enable_tools: bool = True) -> tp.Iterator[ChatResponseT]:
        from llama_index.core.tools import FunctionTool

        if not enable_tools:
            messages = list(messages)
            messages += [{"role": "user", "content": "Write the final answer without calling tools."}]
        messages = self.format_messages(messages, enable_tools=enable_tools)
        tools = self.tools
        if enable_tools and tools:
            fns = [self.tool_registry[tool] for tool in tools]
            tools = [FunctionTool.from_defaults(fn=fn) for fn in fns]
            return self.llm.stream_chat_with_tools(tools, chat_history=messages, **self.chat_kwargs)
        else:
            return self.llm.stream_chat(messages, **self.chat_kwargs)

    def get_delta_content(self, response_chunk: ChatResponseT) -> tp.Optional[str]:
        return self.process_thought(content=response_chunk.delta)

    @property
    def supports_tool_calling(self) -> bool:
        return True

    def get_chat_tool_calls(self, response: ChatResponseT) -> tp.List[tp.Kwargs]:
        tool_call_mapping = {}
        for i, tc in enumerate(self.llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)):
            tool_call_mapping[i] = {
                "id": tc.tool_id,
                "name": tc.tool_name,
                "arguments": tc.tool_kwargs,
            }
        additional_kwargs = get(response.message, "additional_kwargs", {})
        tool_calls = get(additional_kwargs, "tool_calls", [])
        for i, tc in enumerate(tool_calls):
            tool_call_mapping[i]["tool_call"] = tc
        return [tool_call_mapping[i] for i in sorted(tool_call_mapping)]

    def get_stream_tool_calls(self, response_chunks: tp.Iterator[ChatResponseT]) -> tp.List[tp.Kwargs]:
        tool_call_mapping = {}
        for chunk in response_chunks:
            for i, tc in enumerate(self.get_chat_tool_calls(chunk)):
                tool_call_mapping[i] = tc
        return [tool_call_mapping[i] for i in sorted(tool_call_mapping)]

    def get_tool_call_messages(self, tool_calls: tp.List[tp.Kwargs]) -> tp.List[tp.Kwargs]:
        from llama_index.core.llms import ChatMessage

        return [
            ChatMessage(
                role="assistant",
                content=None,
                additional_kwargs={
                    "tool_calls": list(map(lambda x: x["tool_call"], tool_calls)),
                },
            )
        ]

    def get_tool_result_messages(self, tool_results: tp.List[tp.Kwargs]) -> tp.List[tp.Kwargs]:
        from llama_index.core.llms import ChatMessage

        messages = []
        for tr in tool_results:
            messages.append(
                ChatMessage(
                    role="tool",
                    content=tr["output"],
                    additional_kwargs={
                        "tool_call_id": tr["id"],
                    },
                )
            )
        return messages


class OllamaCompletions(OpenAICompatibleCompletions):
    """Completions class for Ollama.

    !!! info
        For default settings, see `chat.completions_configs.ollama` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): Ollama model identifier.

            Pulls the model if not already available locally.
        client_kwargs (KwargsLike): Keyword arguments for `ollama.Client`.
        chat_kwargs (KwargsLike): Keyword arguments for `ollama.Client.chat`.
        **kwargs: Keyword arguments for `Completions` or used as `client_kwargs` or `chat_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "ollama"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.ollama"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        chat_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        OpenAICompatibleCompletions.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            chat_kwargs=chat_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ollama")
        from ollama import Client

        ollama_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = ollama_config.pop("model", None)
        def_quick_model = ollama_config.pop("quick_model", None)
        def_client_kwargs = ollama_config.pop("client_kwargs", None)
        def_chat_kwargs = ollama_config.pop("chat_kwargs", None)

        if model is None:
            model = def_quick_model if self.quick_mode else def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = self.get_init_arg_names()
        for k in list(ollama_config.keys()):
            if k in init_arg_names:
                ollama_config.pop(k)

        _client_kwargs = {}
        _chat_kwargs = {}
        for k, v in ollama_config.items():
            _chat_kwargs[k] = v

        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        chat_kwargs = merge_dicts(_chat_kwargs, def_chat_kwargs, chat_kwargs)

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
        self._chat_kwargs = chat_kwargs

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
    def chat_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `ollama.Client.chat`.

        Returns:
            Kwargs: Keyword arguments for chat completion.
        """
        return self._chat_kwargs

    def format_kwargs(self, messages: tp.ChatMessages, enable_tools: bool = True) -> tp.Kwargs:
        """Format keyword arguments for the API call.

        Args:
            messages (ChatMessages): List representing the conversation history.

        Returns:
            Kwargs: Keyword arguments for the API call.
        """
        if not enable_tools:
            messages = list(messages)
            messages += [{"role": "user", "content": "Write the final answer without calling tools."}]
        chat_kwargs = dict(self.chat_kwargs)
        chat_kwargs["tools"] = chat_kwargs.get("tools", []) + self.tools
        for i, tool in enumerate(chat_kwargs["tools"]):
            if isinstance(tool, str) and tool in self._tool_registry:
                chat_kwargs["tools"][i] = self.function_to_tool_spec(self._tool_registry[tool], name=tool)
        if not enable_tools:
            chat_kwargs.pop("tools", None)
            chat_kwargs.pop("tool_choice", None)
        return dict(
            messages=messages,
            model=self.model,
            stream=self.stream,
            **chat_kwargs,
        )

    def get_chat_response(self, messages: tp.ChatMessages, enable_tools: bool = True) -> OllamaChatResponseT:
        return self.client.chat(**self.format_kwargs(messages, enable_tools=enable_tools))

    def get_message_content(self, response: OllamaChatResponseT) -> tp.Optional[str]:
        message = response["message"]
        if hasattr(message, "thinking"):
            thought = message.thinking
        elif hasattr(message, "reasoning"):
            thought = message.reasoning
        elif hasattr(message, "reasoning_content"):
            thought = message.reasoning_content
        else:
            thought = None
        content = message.content
        return self.process_thought(thought=thought, content=content, flush=True)

    def get_stream_response(
        self,
        messages: tp.ChatMessages,
        enable_tools: bool = True,
    ) -> tp.Iterator[OllamaChatResponseT]:
        return self.client.chat(**self.format_kwargs(messages, enable_tools=enable_tools))

    def get_delta_content(self, response_chunk: OllamaChatResponseT) -> tp.Optional[str]:
        message = response_chunk["message"]
        if hasattr(message, "thinking"):
            thought = message.thinking
        elif hasattr(message, "reasoning"):
            thought = message.reasoning
        elif hasattr(message, "reasoning_content"):
            thought = message.reasoning_content
        else:
            thought = None
        content = message.content
        return self.process_thought(thought=thought, content=content)

    def get_stream_tool_calls(self, response_chunks: tp.Iterator[OllamaChatResponseT]) -> tp.List[tp.Kwargs]:
        for response_chunk in response_chunks:
            tool_calls = self.get_chat_tool_calls(response_chunk)
            if tool_calls:
                return tool_calls
        return []

    @property
    def should_dump_arguments(self) -> bool:
        return False


def resolve_completions(completions: tp.CompletionsLike = None) -> tp.MaybeType[Completions]:
    """Resolve and return a `Completions` subclass or instance.

    !!! info
        For default settings, see `chat` in `vectorbtpro._settings.knowledge`.

    Args:
        completions (CompletionsLike): Identifier, subclass, or instance of `Completions`.

            Supported identifiers:

            * "openai" for `OpenAICompletions`
            * "anthropic" for `AnthropicCompletions`
            * "gemini" for `GeminiCompletions`
            * "hf_inference" for `HFInferenceCompletions`
            * "litellm" for `LiteLLMCompletions`
            * "llama_index" for `LlamaIndexCompletions`
            * "ollama" for `OllamaCompletions`
            * "auto" to select the first available option

    Returns:
        Completions: Resolved completions class or instance.
    """
    if completions is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["chat"]
        completions = chat_cfg["completions"]
    if isinstance(completions, str):
        if completions.lower() == "auto":
            import os
            from vectorbtpro.utils.module_ import check_installed

            if check_installed("openai") and os.getenv("OPENAI_API_KEY"):
                completions = "openai"
            elif check_installed("anthropic") and os.getenv("ANTHROPIC_API_KEY"):
                completions = "anthropic"
            elif check_installed("google.genai") and os.getenv("GEMINI_API_KEY"):
                completions = "gemini"
            elif check_installed("huggingface_hub") and os.getenv("HF_TOKEN"):
                completions = "hf_inference"
            elif check_installed("openai"):
                completions = "openai"
            elif check_installed("anthropic"):
                completions = "anthropic"
            elif check_installed("google.genai"):
                completions = "gemini"
            elif check_installed("huggingface_hub"):
                completions = "hf_inference"
            elif check_installed("litellm"):
                completions = "litellm"
            elif check_installed("llama_index"):
                completions = "llama_index"
            elif check_installed("ollama"):
                completions = "ollama"
            else:
                raise ValueError(
                    "No completions available. "
                    "Please install one of the supported packages: "
                    "openai, "
                    "litellm, "
                    "llama-index, "
                    "huggingface-hub, "
                    "google-genai, "
                    "anthropic, "
                    "ollama."
                )
        curr_module = sys.modules[__name__]
        found_completions = None
        for name, cls in inspect.getmembers(curr_module, inspect.isclass):
            if name.endswith("Completions"):
                _short_name = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == completions.lower():
                    found_completions = cls
                    break
        if found_completions is None:
            raise ValueError(f"Invalid completions: {completions!r}")
        completions = found_completions
    if isinstance(completions, type):
        checks.assert_subclass_of(completions, Completions, arg_name="completions")
    else:
        checks.assert_instance_of(completions, Completions, arg_name="completions")
    return completions


def complete(message: str, completions: tp.CompletionsLike = None, **kwargs) -> tp.ChatOutput:
    """Get and return the chat completion for a provided message.

    Args:
        message (str): Input message for which to generate a completion.
        completions (CompletionsLike): Identifier, subclass, or instance of `Completions`.

            Resolved using `resolve_completions`.
        **kwargs: Keyword arguments to initialize or update `completions`.

    Returns:
        ChatOutput: Completion output generated by the resolved completions.
    """
    completions = resolve_completions(completions=completions)
    if isinstance(completions, type):
        completions = completions(**kwargs)
    elif kwargs:
        completions = completions.replace(**kwargs)
    return completions.get_completion(message)


def completed(message: str, completions: tp.CompletionsLike = None, **kwargs) -> str:
    """Return completion content for a given message using the provided completions configuration.

    Args:
        message (str): Input message.
        completions (CompletionsLike): Identifier, subclass, or instance of `Completions`.

            Resolved using `resolve_completions`.
        **kwargs: Keyword arguments to initialize or update `completions`.

    Returns:
        str: Completion content based on the input message.
    """
    completions = resolve_completions(completions=completions)
    if isinstance(completions, type):
        completions = completions(**kwargs)
    elif kwargs:
        completions = completions.replace(**kwargs)
    return completions.get_completion_content(message)
