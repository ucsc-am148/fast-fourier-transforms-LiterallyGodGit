"""STUDENT FILE: implement the canonical twiddle helpers.

Three patterns (so seven inconsistently-named lecture helpers collapse to ~3):
  1. radix-2 length-N/2 twiddles   make_radix2_twiddles
  2. per-stage radix-16 twiddles   make_radix16_twiddles
  3. Bailey cross-term twiddles    make_bailey_cross_twiddles

Plus two scaffolding tables (full DFT, padded-R DFT) and the bit-reversal
permutation. Use the forward-FFT sign convention exp(-2*pi*i * ...) and
return (re, im) tuples of separate real-valued tensors everywhere.

When you implement each function, the signature should match the docstring
exactly -- the harness expects (re, im) tuples with specific shapes/dtypes,
and sanity_check.py will FAIL if you return something else.
"""

import math

import torch


# =============================================================================
# Pattern 1: radix-2 length-N/2 twiddles  (F2, F3)
# =============================================================================

def make_radix2_twiddles(
    N: int,
    dtype: torch.dtype = torch.float32,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.arange(N // 2, dtype=torch.float64, device=device)
    angles = -2 * torch.pi * k / N
    tw_re = torch.cos(angles).to(dtype)
    tw_im = torch.sin(angles).to(dtype)
    return tw_re, tw_im


# =============================================================================
# Pattern 2: per-stage radix-16 twiddles  (F4; reused by F5/F6/F7 via F4)
# =============================================================================
# The index bookkeeping for this helper is given -- the per-stage permute
# schedule means the column-axis labels at stage s are a mix of already-
# transformed output digits and not-yet-transformed input digits, in an
# order set by the cumulative permutation history. 

def _column_axis_labeling(L: int) -> list[tuple]:
    """Track axis labels through the per-stage permute schedule.

    Convention: input n decomposes as n = sum_i d_i * 16^(L-1-i) with d_0 the
    high digit; output k similarly with e_i. Initial tile has axis i labeled
    ('d', i). At each stage s the kernel applies perm = (s,) + (others in
    original order), bringing axis s to position 0; the four-tl.dot then
    transforms position 0 from ('d', s) to ('e', L-1-s).

    Returns a list of length L; entry s is the tuple of L-1 labels at axis
    positions 1..L-1 of the (16,)*L tile *after* the stage-s permute.
    """
    A = [('d', i) for i in range(L)]
    out = []
    for s in range(L):
        P = [A[s]] + [A[i] for i in range(L) if i != s]
        out.append(tuple(P[1:]))
        A = [('e', L - 1 - s)] + P[1:]
    return out


def make_radix16_twiddles(
    N: int,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    L = round(math.log(N, 16))
    assert 16 ** L == N, f"N must be a power of 16, got {N}"

    labels = _column_axis_labeling(L)  # list of length L; entry s = tuple of L-1 labels

    # Build flat index [0, N//16) for the column axis
    col = torch.arange(N // 16, dtype=torch.int64, device=device)  # (N//16,)

    tw_re_list = []
    tw_im_list = []

    for s in range(L):
        m = torch.arange(16, dtype=torch.float64, device=device)  # (16,)

        if s == 0:
            # Stage 0: twiddle is all ones
            tw_re_list.append(torch.ones(16, N // 16, dtype=torch.float16, device=device))
            tw_im_list.append(torch.zeros(16, N // 16, dtype=torch.float16, device=device))
            continue

        # Reconstruct t for each column index c in [0, N//16)
        # t = sum_{j=0}^{s-1} digit-value-of-e_{L-1-j} at position j * 16^j
        # The axis labels at stage s are labels[s]: a tuple of L-1 labels
        # for positions 1..L-1 of the tile after the stage-s permute.
        # We need the labels that are output digits e_{L-1-j} for j=0..s-1.
        # Those appear at positions 0..s-2 in labels[s] (0-indexed into the tuple).

        # The column index c indexes the flattened (L-1)-dimensional remaining axes.
        # Each axis has size 16. We extract each digit by repeated mod/div.
        stage_labels = labels[s]  # tuple of L-1 labels for positions 1..L-1

        t = torch.zeros(N // 16, dtype=torch.float64, device=device)

        for j in range(s):
            # We want digit e_{L-1-j}, which appears at position j in stage_labels
            # (positions 0..s-2 of stage_labels are the s output digits already done)
            # The column index is laid out with position 0 of stage_labels as slowest axis
            # and position L-2 as fastest axis (size-16 each).
            # So digit at position p in stage_labels = (col // 16^(L-2-p)) % 16
            axis_pos = j  # position in stage_labels tuple
            stride = 16 ** (L - 2 - axis_pos)
            digit = (col // stride) % 16
            t = t + digit.double() * (16 ** j)

        # tw[m, c] = exp(-2*pi*i * m * t / 16^(s+1))
        denom = 16 ** (s + 1)
        # angles shape: (16, N//16)
        angles = -2 * torch.pi * m[:, None] * t[None, :] / denom

        tw_re_list.append(torch.cos(angles).to(torch.float16))
        tw_im_list.append(torch.sin(angles).to(torch.float16))

    tw_re = torch.stack(tw_re_list, dim=0)  # (L, 16, N//16)
    tw_im = torch.stack(tw_im_list, dim=0)

    return tw_re, tw_im


# =============================================================================
# Pattern 3: Bailey cross-term twiddles  (F3, F5, F6, F7)
# =============================================================================

def make_bailey_cross_twiddles(
    m0: int,
    M: int,
    N: int,
    dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    n1 = torch.arange(m0, dtype=torch.float64, device=device)  # (m0,)
    kM = torch.arange(M,  dtype=torch.float64, device=device)  # (M,)
    angles = -2 * torch.pi * n1[:, None] * kM[None, :] / N    # (m0, M)
    re = torch.cos(angles).to(dtype)
    im = torch.sin(angles).to(dtype)
    return re, im

# =============================================================================
# Scaffolding tables
# =============================================================================

def make_dft_matrix(
    N: int,
    dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    j = torch.arange(N, dtype=torch.float64, device=device)
    k = torch.arange(N, dtype=torch.float64, device=device)
    angles = -2 * torch.pi * j[:, None] * k[None, :] / N  # (N, N)
    W_re = torch.cos(angles).to(dtype)
    W_im = torch.sin(angles).to(dtype)
    return W_re, W_im

def make_dft_R_padded(
    R: int,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    j = torch.arange(16, dtype=torch.float64, device=device)
    k = torch.arange(16, dtype=torch.float64, device=device)
    # rows wrap mod R, columns wrap mod R; outside [0,R) cols are zero (padding)
    angles = -2 * torch.pi * (j[:, None] % R) * (k[None, :] % R) / R  # (16, 16)
    M_re = torch.cos(angles).to(torch.float16)
    M_im = torch.sin(angles).to(torch.float16)
    # zero out columns >= R (the padded region)
    M_re[:, R:] = 0.0
    M_im[:, R:] = 0.0
    return M_re, M_im

def bit_reversal_perm(N: int, device: str = 'cuda') -> torch.Tensor:
    n_bits = round(math.log2(N))
    idx = torch.arange(N, dtype=torch.int32, device=device)
    rev = torch.zeros(N, dtype=torch.int32, device=device)
    for _ in range(n_bits):
        rev = (rev << 1) | (idx & 1)
        idx = idx >> 1
    return rev
