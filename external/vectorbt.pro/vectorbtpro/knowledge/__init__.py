# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Package providing utility functions and classes for constructing and managing knowledge assets.

Run for the examples:

```pycon
>>> dataset = [
...     {"s": "ABC", "b": True, "d2": {"c": "red", "l": [1, 2]}},
...     {"s": "BCD", "b": True, "d2": {"c": "blue", "l": [3, 4]}},
...     {"s": "CDE", "b": False, "d2": {"c": "green", "l": [5, 6]}},
...     {"s": "DEF", "b": False, "d2": {"c": "yellow", "l": [7, 8]}},
...     {"s": "EFG", "b": False, "d2": {"c": "black", "l": [9, 10]}, "xyz": 123}
... ]
>>> asset = vbt.KnowledgeAsset(dataset)
```

!!! info
    For default settings, see `vectorbtpro._settings.knowledge`.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.knowledge.asset_pipelines import *
    from vectorbtpro.knowledge.base_asset_funcs import *
    from vectorbtpro.knowledge.base_assets import *
    from vectorbtpro.knowledge.completions import *
    from vectorbtpro.knowledge.custom_asset_funcs import *
    from vectorbtpro.knowledge.custom_assets import *
    from vectorbtpro.knowledge.doc_ranking import *
    from vectorbtpro.knowledge.doc_storing import *
    from vectorbtpro.knowledge.embeddings import *
    from vectorbtpro.knowledge.formatting import *
    from vectorbtpro.knowledge.text_splitting import *
    from vectorbtpro.knowledge.tokenization import *
