#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The Google Research Authors and The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for constructing PyTrees of PartitionSpecs."""

# utils adapted from https://github.com/google-research/google-research/blob/master/flax_models/t5x/partitions.py

import re

from flax.core.frozen_dict import freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.experimental import PartitionSpec as P


# Sentinels
_unmatched = object()

# For specifying empty leaf dict `{}`
empty_dict = object()


def _match(qs, ks):
    """Return True if regexes in qs match any window of strings in tuple ks."""
    # compile regexes and force complete match
    qts = tuple(map(lambda x: re.compile(x + "$"), qs))
    for i in range(len(ks) - len(qs) + 1):
        matches = [x.match(y) for x, y in zip(qts, ks[i:])]
        if matches and all(matches):
            return True
    return False


def _replacement_rules(rules):
    def replace(key, val):
        for rule, replacement in rules:
            if _match(rule, key):
                return replacement
        return val

    return replace


def _get_partition_rules():
    return [
        # Embeddings
        (("SelfAttention", "relative_attention_bias", "embedding"), None),
        (("shared", "embedding"), P("mp", None)),
        # Attention
        ((r"SelfAttention", "(q|k|v)", "kernel"), P(None, "mp")),
        ((r"SelfAttention", "o", "kernel"), P("mp", None)),
        ((r"EncDecAttention", "(q|k|v)", "kernel"), P(None, "mp")),
        ((r"EncDecAttention", "o", "kernel"), P("mp", None)),
        # FFN
        ((r"DenseReluDense", "wi_0", "kernel"), P(None, "mp")),
        ((r"DenseReluDense", "wi_1", "kernel"), P(None, "mp")),
        ((r"DenseReluDense", "wo", "kernel"), P("mp", None)),
        # layer norms
        ((r"layer_norm", "weight"), None),
        ((r"final_layer_norm", "weight"), None),
        # projection
        (("lm_head", "kernel"), P(None, "mp")),
    ]


def set_partitions(in_dict):
    rules = _get_partition_rules()
    replace = _replacement_rules(rules)
    initd = {k: _unmatched for k in flatten_dict(in_dict)}
    result = {k: replace(k, v) for k, v in initd.items()}
    assert _unmatched not in result.values(), "Incomplete partition spec."
    return freeze(unflatten_dict(result))
