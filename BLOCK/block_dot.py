import numpy as np

MAX_ELEMENTS = int(2**22)

np_dot = np.dot

#
# based on:
#
# http://stackoverflow.com/a/21096605/1444073
#


def _block_slices(dim_size, block_size):
    """Generator that yields slice objects for indexing into 
    sequential blocks of an array along a particular axis
    """
    count = 0
    while True:
        yield slice(count, count + block_size, 1)
        count += block_size
        if count > dim_size:
            raise StopIteration


def _block_buffer(buf, block):
    """Returns `block` and uses it to buffer succeeding blocks
    if `buf` is `None`, and `buf` otherwise.
    """
    if buf is None:
        buf = block.copy()
        return buf, buf
    else:
        view = buf[0:block.shape[0], 0:block.shape[1]]
        np.copyto(view, block)
        return buf, view


def block_dot(A, B, max_elements=None, out=None):
    """Computes the dot product of two matrices in a block-wise fashion. 
    Only blocks of `A` with a maximum size of `max_elements` will be 
    processed simultaneously.
    """

    if isinstance(B, np.memmap) and not isinstance(A, np.memmap):
        # compute (B' A')' instead, because this function is optimized
        # for the *first* argument being the off-core array
        return block_dot(B.T, A.T, max_elements, out.T if out else None).T

    max_elements = max_elements or MAX_ELEMENTS

    m,  n = A.shape
    n1, o = B.shape

    if n1 != n:
        raise ValueError('matrices are not aligned')

    if A.flags.f_contiguous:
        # prioritize processing as many columns of A as possible
        max_cols = max(1, max_elements / m)
        max_rows = max_elements / max_cols

    else:
        # prioritize processing as many rows of A as possible
        max_rows = max(1, max_elements / n)
        max_cols = max_elements / max_rows

    if out is None:
        out = np.empty((m, o), dtype=np.result_type(A, B))
    elif out.shape != (m, o):
        raise ValueError('output array has incorrect dimensions')

    A_block_buffer = None
    for mm in _block_slices(m, max_rows):
        out[mm, :] = 0
        for nn in _block_slices(n, max_cols):
            A_block = A[mm, nn]
            A_block_buffer, A_block_view = _block_buffer(
                A_block_buffer, A_block)
            out[mm, :] += np_dot(A_block_view, B[nn, :])
    del A_block_buffer

    return out
