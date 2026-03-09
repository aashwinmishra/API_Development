"""
models.py
=========
Factory functions for building Neural Operator models (FNO / TFNO) for
Ptychographic reconstruction.

Supported architectures:
    - FNO  : Fourier Neural Operator
    - TFNO : Tensorized Fourier Neural Operator (factorized weight tensors)

Both share the same hyperparameter interface so experiments can be run by
simply swapping the ``model_type`` argument.
"""

import torch
import torch.nn as nn
from neuralop.models import FNO, TFNO
from neuralop.utils import count_model_params


# Allowed factorization schemes for TFNO
TFNO_FACTORIZATIONS = {"tucker", "cp", "tt"}


def build_fno(
    n_modes: tuple[int, ...] = (24, 24),
    in_channels: int = 1,
    out_channels: int = 2,
    hidden_channels: int = 64,
    projection_channel_ratio: int = 2,
    n_layers: int = 4,
    positional_embedding: str = "grid",
) -> FNO:
    """Construct a Fourier Neural Operator (FNO) model.

    Args:
        n_modes: Number of Fourier modes to retain per spatial dimension.
            A 2-tuple for 2-D data, e.g. ``(24, 24)``.
        in_channels: Number of input channels (e.g. 1 for a single diffraction
            pattern channel; the positional grid channels are added internally
            when ``positional_embedding="grid"``).
        out_channels: Number of output channels (e.g. 2 for amplitude + phase).
        hidden_channels: Width of the FNO layers (latent channel dimension).
        projection_channel_ratio: Ratio of the projection MLP's hidden size to
            ``hidden_channels``. A value of 2 doubles the hidden channels in
            the projection head.
        n_layers: Number of FNO blocks stacked in sequence.
        positional_embedding: Type of positional embedding. Use ``"grid"`` to
            append spatial grid coordinates to the input (standard FNO setup)
            or ``None`` to disable.

    Returns:
        An ``FNO`` model instance (not yet moved to a device).
    """
    model = FNO(
        n_modes=n_modes,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        projection_channel_ratio=projection_channel_ratio,
        n_layers=n_layers,
        positional_embedding=positional_embedding,
    )
    return model


def build_tfno(
    n_modes: tuple[int, ...] = (24, 24),
    in_channels: int = 1,
    out_channels: int = 2,
    hidden_channels: int = 64,
    projection_channel_ratio: int = 2,
    n_layers: int = 4,
    factorization: str = "tucker",
    rank: float = 0.1,
    positional_embedding: str = "grid",
) -> TFNO:
    """Construct a Tensorized Fourier Neural Operator (TFNO) model.

    TFNO factorizes the spectral weight tensors (Tucker / CP / TT), which
    dramatically reduces parameter count while retaining expressive power.

    Args:
        n_modes: Number of Fourier modes to retain per spatial dimension.
        in_channels: Number of input channels.
        out_channels: Number of output channels (e.g. 2 for amplitude + phase).
        hidden_channels: Width of the FNO layers.
        projection_channel_ratio: Ratio of projection MLP hidden size to
            ``hidden_channels``.
        n_layers: Number of FNO blocks.
        factorization: Tensor factorization scheme for the spectral weights.
            One of ``"tucker"``, ``"cp"``, or ``"tt"``.
        rank: Compression rank. A float in ``(0, 1]`` is treated as a fraction
            of the full tensor size; an int sets an exact rank.
        positional_embedding: Type of positional embedding (``"grid"`` or
            ``None``).

    Returns:
        A ``TFNO`` model instance (not yet moved to a device).

    Raises:
        ValueError: If ``factorization`` is not one of the supported schemes.
    """
    if factorization not in TFNO_FACTORIZATIONS:
        raise ValueError(
            f"Unsupported factorization '{factorization}'. "
            f"Choose from {sorted(TFNO_FACTORIZATIONS)}."
        )

    model = TFNO(
        n_modes=n_modes,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        projection_channel_ratio=projection_channel_ratio,
        n_layers=n_layers,
        factorization=factorization,
        rank=rank,
        positional_embedding=positional_embedding,
    )
    return model


def build_model(
    model_type: str = "tfno",
    **kwargs,
) -> nn.Module:
    """Build an FNO or TFNO model by name.

    This is the primary entry point for model construction. All keyword
    arguments are forwarded to the underlying factory function, so only
    the parameters relevant to the chosen ``model_type`` need to be
    provided.

    Args:
        model_type: Architecture to use. One of ``"fno"`` or ``"tfno"``
            (case-insensitive).
        **kwargs: Hyperparameters forwarded to :func:`build_fno` or
            :func:`build_tfno`. TFNO-specific kwargs (``factorization``,
            ``rank``) are silently ignored when ``model_type="fno"``.

    Returns:
        An ``nn.Module`` instance (not yet moved to a device).

    Raises:
        ValueError: If ``model_type`` is not ``"fno"`` or ``"tfno"``.

    Example::

        model = build_model(
            model_type="tfno",
            n_modes=(16, 16),
            hidden_channels=32,
            n_layers=3,
            factorization="tucker",
            rank=0.2,
        )
    """
    model_type = model_type.lower()

    if model_type == "fno":
        # Remove TFNO-only kwargs so build_fno does not receive unknown args
        for key in ("factorization", "rank"):
            kwargs.pop(key, None)
        model = build_fno(**kwargs)

    elif model_type == "tfno":
        model = build_tfno(**kwargs)

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose 'fno' or 'tfno'."
        )

    n_params = count_model_params(model)
    print(f"Model : {model_type.upper()}  |  Parameters : {n_params:,}")
    return model
