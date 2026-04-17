# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing extensions for chunking of portfolio."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.merging import concat_arrays, column_stack_arrays
from vectorbtpro.portfolio.enums import SimulationOutput
from vectorbtpro.records.chunking import merge_records
from vectorbtpro.utils.chunking import ChunkMeta, ArraySlicer
from vectorbtpro.utils.config import ReadonlyConfig
from vectorbtpro.utils.template import Rep

__all__ = []


def get_flex_array_1d_cs_slicer(ann_args: tp.AnnArgs) -> ArraySlicer:
    """Return the slicer for flexible 1D arrays based on the cash sharing configuration.

    Args:
        ann_args (AnnArgs): Annotated arguments.

            See `vectorbtpro.utils.parsing.annotate_args`.

    Returns:
        ArraySlicer: Slicer configured for slicing flexible 1D arrays.
    """
    cash_sharing = ann_args["cash_sharing"]["value"]
    if cash_sharing:
        return base_ch.FlexArraySlicer()
    return base_ch.flex_array_1d_gl_slicer


def get_flex_array_cs_slicer(ann_args: tp.AnnArgs) -> ArraySlicer:
    """Return the slicer for flexible 2D arrays based on the cash sharing configuration.

    Args:
        ann_args (AnnArgs): Annotated arguments.

            See `vectorbtpro.utils.parsing.annotate_args`.

    Returns:
        ArraySlicer: Slicer configured for slicing flexible 2D arrays.
    """
    cash_sharing = ann_args["cash_sharing"]["value"]
    if cash_sharing:
        return base_ch.FlexArraySlicer(axis=1)
    return base_ch.flex_array_gl_slicer


def namedtuple_merge_func(
    namedtuples: tp.List[tp.NamedTuple],
    chunk_meta: tp.Iterable[ChunkMeta],
    ann_args: tp.AnnArgs,
    mapper: base_ch.GroupLensMapper,
) -> tp.NamedTuple:
    """Merge chunks of named tuples.

    Concatenates 1-dimensional arrays, stacks columns of 2-dimensional arrays, and
    merges record arrays using `vectorbtpro.records.chunking.merge_records`.
    Other object types will raise an error.

    Args:
        namedtuples (List[tp.NamedTuple]): List of named tuple chunks.
        chunk_meta (Iterable[ChunkMeta]): Iterable containing metadata for each chunk.

            See `vectorbtpro.utils.chunking.iter_chunk_meta`.
        ann_args (AnnArgs): Annotated arguments.

            See `vectorbtpro.utils.parsing.annotate_args`.
        mapper (GroupLensMapper): Mapper for grouping and lens mapping.

    Returns:
        NamedTuple: Instance of the same type as the first named tuple with merged data.
    """
    nt_kwargs = dict()
    for k, v in namedtuples[0]._asdict().items():
        if v is None:
            nt_kwargs[k] = None
            continue
        if not isinstance(v, np.ndarray):
            raise TypeError(f"Cannot merge a named tuple field {k!r} of type {type(v)}")
        if v.ndim == 2:
            nt_kwargs[k] = column_stack_arrays([getattr(nt, k) for nt in namedtuples])
        elif v.ndim == 1:
            if v.dtype.fields is None:
                nt_kwargs[k] = np.concatenate([getattr(nt, k) for nt in namedtuples])
            else:
                records = [getattr(nt, k) for nt in namedtuples]
                nt_kwargs[k] = merge_records(records, chunk_meta, ann_args=ann_args, mapper=mapper)
        else:
            raise ValueError(f"Cannot merge a named tuple field {k!r} with number of dimensions {v.ndim}")
    return type(namedtuples[0])(**nt_kwargs)


def in_outputs_merge_func(results: tp.List[SimulationOutput], *args, **kwargs) -> tp.NamedTuple:
    """Merge chunks of in-place output objects.

    Args:
        results (List[SimulationOutput]): List of simulation output chunks.
        *args: Positional arguments for `namedtuple_merge_func`.
        **kwargs: Keyword arguments for `namedtuple_merge_func`.

    Returns:
        NamedTuple: Instance of the same type as `results[0].in_outputs` with merged data.
    """
    in_outputs = [r.in_outputs for r in results]
    return namedtuple_merge_func(in_outputs, *args, **kwargs)


def last_state_merge_func(results: tp.List[SimulationOutput], *args, **kwargs) -> tp.NamedTuple:
    """Merge chunks of last state objects.

    Args:
        results (List[SimulationOutput]): List of simulation output chunks.
        *args: Positional arguments for `namedtuple_merge_func`.
        **kwargs: Keyword arguments for `namedtuple_merge_func`.

    Returns:
        NamedTuple: Instance of the same type as `results[0].last_state` with merged data.
    """
    last_states = [r.last_state for r in results]
    return namedtuple_merge_func(last_states, *args, **kwargs)


def merge_sim_outs(
    results: tp.List[SimulationOutput],
    chunk_meta: tp.Iterable[ChunkMeta],
    ann_args: tp.AnnArgs,
    mapper: base_ch.GroupLensMapper,
    in_outputs_merge_func: tp.Callable = in_outputs_merge_func,
    last_state_merge_func: tp.Callable = last_state_merge_func,
    **kwargs,
) -> SimulationOutput:
    """Merge chunks of `vectorbtpro.portfolio.enums.SimulationOutput` instances.

    Merges various components including order and log records, cash deposits, cash earnings, call sequence,
    in-place outputs, and simulation timing arrays. If `vectorbtpro.portfolio.enums.SimulationOutput.in_outputs`
    is provided, a merge function such as `in_outputs_merge_func` must be used.

    Args:
        results (List[SimulationOutput]): List of simulation output chunks.
        chunk_meta (Iterable[ChunkMeta]): Iterable containing metadata for each chunk.

            See `vectorbtpro.utils.chunking.iter_chunk_meta`.
        ann_args (AnnArgs): Annotated arguments.

            See `vectorbtpro.utils.parsing.annotate_args`.
        mapper (GroupLensMapper): Mapper for grouping and lens mapping.
        in_outputs_merge_func (Callable): Function to merge in-place output objects.
        last_state_merge_func (Callable): Function to merge last state objects.
        **kwargs: Keyword arguments for `in_outputs_merge_func`.

    Returns:
        SimulationOutput: Merged simulation output instance.
    """
    order_records = [r.order_records for r in results]
    order_records = merge_records(order_records, chunk_meta, ann_args=ann_args, mapper=mapper)

    log_records = [r.log_records for r in results]
    log_records = merge_records(log_records, chunk_meta, ann_args=ann_args, mapper=mapper)

    target_shape = ann_args["target_shape"]["value"]
    if results[0].cash_deposits.shape == target_shape:
        cash_deposits = column_stack_arrays([r.cash_deposits for r in results])
    else:
        cash_deposits = results[0].cash_deposits
    if results[0].cash_earnings.shape == target_shape:
        cash_earnings = column_stack_arrays([r.cash_earnings for r in results])
    else:
        cash_earnings = results[0].cash_earnings
    if results[0].call_seq is not None:
        call_seq = column_stack_arrays([r.call_seq for r in results])
    else:
        call_seq = None
    if results[0].sim_start is not None:
        sim_start = concat_arrays([r.sim_start for r in results])
    else:
        sim_start = None
    if results[0].sim_end is not None:
        sim_end = concat_arrays([r.sim_end for r in results])
    else:
        sim_end = None
    if results[0].in_outputs is not None:
        in_outputs = in_outputs_merge_func(results, chunk_meta, ann_args, mapper, **kwargs)
    else:
        in_outputs = None
    if results[0].last_state is not None:
        last_state = last_state_merge_func(results, chunk_meta, ann_args, mapper, **kwargs)
    else:
        last_state = None
    return SimulationOutput(
        order_records=order_records,
        log_records=log_records,
        cash_deposits=cash_deposits,
        cash_earnings=cash_earnings,
        call_seq=call_seq,
        sim_start=sim_start,
        sim_end=sim_end,
        in_outputs=in_outputs,
        last_state=last_state,
    )


merge_sim_outs_config = ReadonlyConfig(
    dict(
        merge_func=merge_sim_outs,
        merge_kwargs=dict(
            chunk_meta=Rep("chunk_meta"),
            ann_args=Rep("ann_args"),
            mapper=base_ch.group_lens_mapper,
        ),
    )
)
"""Configuration for merging simulation outputs using `merge_sim_outs`."""
