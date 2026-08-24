"""Shared constants for the DeltaNet NKI kernels.

P_MAX is the SBUF partition width on Trainium (128). Every NKI operation in this
package operates on tiles that fit into 128 partitions.

_BROADCAST_MASK is passed to `nc_stream_shuffle` to broadcast a single-row PSUM
result across all 32 partitions of a lane group. Used in Step 4 of the DeltaNet
recurrence to broadcast `delta` across the 128 rows of the state matrix before
computing the outer product.
"""

P_MAX = 128
_BROADCAST_MASK = [0] * 32
