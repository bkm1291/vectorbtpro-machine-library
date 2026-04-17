# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing configuration for optional dependencies for internal use."""

from vectorbtpro.utils.config import HybridConfig

__all__ = []

__pdoc__ = {}

opt_dep_config = HybridConfig(
    {
        "yfinance": {"version": ">=0.2.56"},
        "binance": {"dist_name": "python-binance", "version": ">=1.0.16"},
        "ccxt": {"version": ">=1.89.14"},
        "ta": {},
        "pandas_ta": {"dist_name": "pandas-ta"},
        "talib": {"dist_name": "TA-Lib"},
        "bottleneck": {},
        "numexpr": {},
        "ray": {"version": ">=1.4.1"},
        "dask": {},
        "matplotlib": {"version": ">=3.2.0"},
        "plotly": {"version": ">=5.0.0"},
        "dash": {},
        "ipywidgets": {"version": ">=7.0.0"},
        "kaleido": {},
        "telegram": {"dist_name": "python-telegram-bot", "version": ">=13.4"},
        "quantstats": {"version": ">=0.0.37"},
        "dill": {},
        "alpaca": {"dist_name": "alpaca-py", "version": ">=0.40.0"},
        "polygon": {"dist_name": "polygon-api-client", "version": ">=1.0.0"},
        "bs4": {"dist_name": "beautifulsoup4"},
        "nasdaqdatalink": {"dist_name": "Nasdaq-Data-Link"},
        "pypfopt": {"dist_name": "PyPortfolioOpt", "version": ">=1.5.1"},
        "universal": {"dist_name": "universal-portfolios"},
        "plotly_resampler": {"dist_name": "plotly-resampler"},
        "technical": {},
        "riskfolio": {"dist_name": "Riskfolio-Lib", "version": ">=3.3.0"},
        "pathos": {},
        "lz4": {},
        "blosc": {},
        "blosc2": {},
        "tables": {},
        "optuna": {},
        "sqlalchemy": {"dist_name": "SQLAlchemy", "version": ">=2.0.0"},
        "mpire": {},
        "duckdb": {},
        "duckdb_engine": {"dist_name": "duckdb-engine"},
        "pyarrow": {},
        "fastparquet": {},
        "tabulate": {},
        "alpha_vantage": {"version": ">=3.0.0"},
        "databento": {},
        "smartmoneyconcepts": {},
        "findatapy": {},
        "github": {"dist_name": "PyGithub", "version": ">=1.59.0"},
        "jmespath": {},
        "jsonpath_ng": {"dist_name": "jsonpath-ng"},
        "fuzzysearch": {},
        "rapidfuzz": {},
        "nestedtext": {},
        "yaml": {"dist_name": "PyYAML"},
        "ruamel": {"dist_name": "ruamel.yaml"},
        "tomlkit": {},
        "markdown": {},
        "pygments": {},
        "IPython": {"dist_name": "ipython"},
        "pymdownx": {"dist_name": "pymdown-extensions"},
        "openai": {},
        "litellm": {},
        "llama_index": {"dist_name": "llama-index"},
        "tiktoken": {},
        "lmdbm": {},
        "bm25s": {},
        "PyStemmer": {},
        "pyperclip": {},
        "platformdirs": {},
        "mcp": {},
        "ipykernel": {},
        "huggingface_hub": {"dist_name": "huggingface-hub"},
        "google.genai": {"dist_name": "google-genai"},
        "anthropic": {},
        "ollama": {},
        "docstring_parser": {},
        "networkx": {},
        "arcticdb": {},
    }
)
"""_"""

__pdoc__[
    "opt_dep_config"
] = f"""Configuration for optional dependencies used internally by vectorbtpro.

Contains package metadata including download links, version requirements, and distribution names where applicable.

```python
{opt_dep_config.prettify_doc()}
```
"""
