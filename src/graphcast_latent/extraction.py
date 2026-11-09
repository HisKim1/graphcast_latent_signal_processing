"""Observe an unchanged GraphCast forward pass through Haiku interception."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np


@contextmanager
def capture_latents(
    model: Any, sink: Callable[[int, np.ndarray], None]
) -> Iterator[None]:
    """Send encoder/state outputs to ``sink`` during a transformed forward pass.

    State zero is the encoder output; later states follow processor residual
    updates. Arrays have axes ``(node, batch, channel)``. Enter this context
    *inside* the Haiku forward function, before JIT tracing. The sink runs on
    the host at execution time; call ``jax.effects_barrier()`` before closing
    its writer. This inference-only path does not support autodifferentiation
    or an outer ``vmap``. Use GraphCast's native batch axis instead.

    The compiled function retains its sink. Keep that sink open for all calls;
    create a new transformed/JIT function when changing the sink.
    """
    import haiku as hk
    from jax.experimental import io_callback

    encoder = model._grid2mesh_gnn
    processor = model._mesh_gnn
    expected_steps = (
        processor._num_message_passing_steps * processor._num_processor_repetitions
    )
    if expected_steps < 1:
        raise ValueError("GraphCast must have at least one processor step")
    encoder_count = 0
    processor_count = 0

    def intercept(next_method, args, kwargs, context):
        nonlocal encoder_count, processor_count
        result = next_method(*args, **kwargs)
        if context.module is encoder and context.method_name == "__call__":
            encoder_count += 1
            state_index = 0
        elif context.module is processor and context.method_name == "_process_step":
            processor_count += 1
            state_index = processor_count
        else:
            return result
        values = result.nodes["mesh_nodes"].features
        if values.ndim != 3:
            raise ValueError("Latent arrays must have axes (node, batch, channel)")
        io_callback(sink, None, state_index, values, ordered=True)
        return result

    with hk.intercept_methods(intercept):
        yield
    if encoder_count != 1 or processor_count != expected_steps:
        raise RuntimeError(
            "Unsupported GraphCast execution: expected one encoder and "
            f"{expected_steps} processor steps; observed "
            f"{encoder_count} and {processor_count}"
        )
