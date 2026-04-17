# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for color manipulation, conversion, and adjustment."""

import numpy as np

from vectorbtpro import _typing as tp

__all__ = []


def map_value_to_cmap(
    value: tp.MaybeSequence[float],
    cmap: tp.Any,
    vmin: tp.Optional[float] = None,
    vcenter: tp.Optional[float] = None,
    vmax: tp.Optional[float] = None,
    as_hex: bool = False,
) -> tp.MaybeSequence[str]:
    """Return the RGB color(s) corresponding to the input value(s) according to the given colormap.

    Args:
        value (MaybeSequence[float]): Numeric value or sequence of values to map to colors.
        cmap (Any): Colormap identifier provided as a string name or a collection (list/tuple) of colors.
        vmin (Optional[float]): Minimum data value for colormap normalization.
        vcenter (Optional[float]): Midpoint for two-slope colormap normalization.
        vmax (Optional[float]): Maximum data value for colormap normalization.
        as_hex (bool): Whether to return colors in hexadecimal format.

    Returns:
        MaybeSequence[str]: Color string in `rgb(r,g,b)` format for each input value.
    """
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("matplotlib")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    value_is_scalar = np.isscalar(value)
    if value_is_scalar:
        value = np.array([value])
    else:
        value = np.asarray(value)

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    elif isinstance(cmap, (tuple, list)):
        cmap = mcolors.LinearSegmentedColormap.from_list("", cmap)

    if vmin is not None and vcenter is not None and vmin > vcenter:
        vmin = vcenter
    if vmin is not None and vcenter is not None and vmin == vcenter:
        vcenter = None
    if vmax is not None and vcenter is not None and vmax < vcenter:
        vmax = vcenter
    if vmax is not None and vcenter is not None and vmax == vcenter:
        vcenter = None

    if vcenter is not None:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        value = norm(value)
    elif vmin is not None or vmax is not None:
        if vmin == vmax:
            value = value * 0 + 0.5
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            value = norm(value)

    colors = cmap(value)

    def to_int_channel(x):
        return int(np.clip(x, 0.0, 1.0) * 255)

    out = []
    for c in colors:
        r = to_int_channel(c[0])
        g = to_int_channel(c[1])
        b = to_int_channel(c[2])

        if as_hex:
            out.append(f"#{r:02x}{g:02x}{b:02x}")
        else:
            out.append(f"rgb({r},{g},{b})")

    if value_is_scalar:
        return out[0]
    return out


def parse_rgba_tuple(color: str) -> tp.Tuple[float, float, float, float]:
    """Return a tuple of normalized RGBA components parsed from the provided color string.

    Args:
        color (str): RGBA color string in the format "rgba(r,g,b,a)".

    Returns:
        Tuple[float, float, float, float]: Tuple containing the red, green, and blue components
            normalized to [0, 1] and the alpha value.
    """
    rgba = color.replace("rgba", "").replace("(", "").replace(")", "").split(",")
    return int(rgba[0]) / 255, int(rgba[1]) / 255, int(rgba[2]) / 255, float(rgba[3])


def parse_rgb_tuple(color: str) -> tp.Tuple[float, float, float]:
    """Return a tuple of normalized RGB components parsed from the provided color string.

    Args:
        color (str): RGB color string in the format "rgb(r,g,b)".

    Returns:
        Tuple[float, float, float]: Tuple containing the red, green, and blue components
            normalized to [0, 1].
    """
    rgb = color.replace("rgb", "").replace("(", "").replace(")", "").split(",")
    return int(rgb[0]) / 255, int(rgb[1]) / 255, int(rgb[2]) / 255


def parse_hex_tuple(color: str) -> tp.Tuple[float, float, float]:
    """Return a tuple of normalized RGB components parsed from the provided hex color string.

    Args:
        color (str): Hex color string in the format "#RRGGBB".

    Returns:
        Tuple[float, float, float]: Tuple containing the red, green, and blue components
            normalized to [0, 1].
    """
    color = color.lstrip("#")
    lv = len(color)
    return tuple(int(color[i : i + lv // 3], 16) / 255 for i in range(0, lv, lv // 3))


def parse_hexa_tuple(color: str) -> tp.Tuple[float, float, float, float]:
    """Return a tuple of normalized RGBA components parsed from the provided hex color string.

    Args:
        color (str): Hex color string in the format "#RRGGBBAA".

    Returns:
        Tuple[float, float, float, float]: Tuple containing the red, green, blue components
            normalized to [0, 1] and the alpha value.
    """
    color = color.lstrip("#")
    lv = len(color)
    return tuple(int(color[i : i + lv // 4], 16) / 255 for i in range(0, lv, lv // 4))


def parse_color(color: tp.Any) -> tp.Tuple[float, float, float, float]:
    """Return a tuple of normalized RGBA components parsed from the provided color.

    Args:
        color (Any): Color represented as a Matplotlib color string,
            hex string (#RRGGBB or #RRGGBBAA), or RGB/RGBA tuple.

    Returns:
        Tuple[float, float, float, float]: Tuple containing the red, green, blue components
            normalized to [0, 1] and the alpha value.
    """
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("matplotlib")
    import matplotlib.colors as mc

    if isinstance(color, str) and color.startswith("rgba"):
        return parse_rgba_tuple(color)
    elif isinstance(color, str) and color.startswith("rgb"):
        r, g, b = parse_rgb_tuple(color)
        return r, g, b, 1.0
    elif isinstance(color, str) and color.startswith("#") and len(color) == 9:
        return parse_hexa_tuple(color)
    elif isinstance(color, str) and color.startswith("#") and len(color) == 7:
        r, g, b = parse_hex_tuple(color)
        return r, g, b, 1.0
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    rgba = mc.to_rgba(c)
    return rgba


def adjust_opacity(color: tp.Any, opacity: float) -> str:
    """Return a color string with the specified opacity adjustment.

    Args:
        color (Any): Color represented as a Matplotlib color string,
            hex string (#RRGGBB or #RRGGBBAA), or RGB/RGBA tuple.
        opacity (float): Desired opacity value.

    Returns:
        str: Color in `rgba(r,g,b,a)` format with the updated opacity.
    """
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("matplotlib")
    import matplotlib.colors as mc

    if isinstance(color, str) and color.startswith("rgba"):
        color = parse_rgba_tuple(color)
    elif isinstance(color, str) and color.startswith("rgb"):
        color = parse_rgb_tuple(color)
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    rgb = mc.to_rgb(c)
    return "rgba(%d,%d,%d,%.4f)" % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), opacity)


def adjust_lightness(color: tp.Any, amount: float = 0.7) -> str:
    """Return an RGB color string with adjusted lightness.

    Args:
        color (Any): Color represented as a Matplotlib color string,
            hex string (#RRGGBB or #RRGGBBAA), or RGB/RGBA tuple.
        amount (float): Factor to adjust the lightness.

            Values less than 1 darken the color, while values greater than 1 lighten it.

    Returns:
        str: Adjusted color in `rgb(r,g,b)` format.
    """
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("matplotlib")
    import matplotlib.colors as mc
    import colorsys

    if isinstance(color, str) and color.startswith("rgba"):
        color = parse_rgba_tuple(color)
    elif isinstance(color, str) and color.startswith("rgb"):
        color = parse_rgb_tuple(color)
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    rgb = colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
    return "rgb(%d,%d,%d)" % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def flatten_opacity(color: tp.Any, background_color: tp.Any = None) -> str:
    """Return a color string with the opacity flattened against the specified background color.

    Args:
        color (Any): Color represented as a Matplotlib color string,
            hex string (#RRGGBB or #RRGGBBAA), or RGB/RGBA tuple.
        background_color (Any): Background color represented as a Matplotlib color string,
            hex string (#RRGGBB or #RRGGBBAA), or RGB/RGBA tuple.

            If None, defaults to white (`rgb(255,255,255)`).

    Returns:
        str: Flattened color in `rgb(r,g,b)` format.
    """
    if background_color is None:
        background_color = "rgb(255,255,255)"

    r_fg, g_fg, b_fg, a_fg = parse_color(color)
    r_bg, g_bg, b_bg, _ = parse_color(background_color)

    r_out = int((1 - a_fg) * r_bg * 255 + a_fg * r_fg * 255)
    g_out = int((1 - a_fg) * g_bg * 255 + a_fg * g_fg * 255)
    b_out = int((1 - a_fg) * b_bg * 255 + a_fg * b_fg * 255)

    return "rgb(%d,%d,%d)" % (r_out, g_out, b_out)


def get_contrast_color(
    color: tp.Any,
    light: tp.Any = "#000000",
    dark: tp.Any = "#ffffff",
    threshold: float = 0.5,
) -> tp.Any:
    """Get a contrasting color (light or dark) based on the luminance of the provided color.

    Args:
        color (Any): Color represented as a Matplotlib color string,
            hex string (#RRGGBB or #RRGGBBAA), or RGB/RGBA tuple.
        light (Any): Color to return for light backgrounds.
        dark (Any): Color to return for dark backgrounds.
        threshold (float): Luminance threshold to determine light vs dark background.

    Returns:
        Any: Contrasting color (either `light` or `dark`).
    """
    r, g, b, _ = parse_color(color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return light if luminance > threshold else dark
