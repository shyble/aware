// Grouped GEMM for cap-keyed projections.
//
// Computes, for every cap k:
//
//     out[offsets[k] : offsets[k+1], :] =
//         xs[offsets[k] : offsets[k+1], :] @ weights[k]
//
// with a *different* row count per cap and no padding. This is the
// primitive the padded dispatch paths approximate: they widen every
// bucket to a common capacity so one dense batched matmul covers all
// caps, then multiply through the padding. With 330 caps over 4096
// tokens the mean bucket holds ~12 rows, so most of that arithmetic is
// against zeros.
//
// Parallelisation: the host builds a flat work list, one entry per
// output tile, so a single launch covers all caps regardless of how
// unevenly tokens are distributed. A cap with 3 rows costs one tile; a
// cap with 300 costs ten. Empty caps cost nothing because they generate
// no work items at all.
//
// Layouts (all row-major, contiguous):
//   xs      [total,  d_in]
//   weights [n_caps, d_in, d_out]
//   out     [total,  d_out]

#define TILE 16

extern "C" __global__ void grouped_gemm_f32(
    const float* __restrict__ xs,
    const float* __restrict__ weights,
    float* __restrict__ out,
    const int* __restrict__ work_cap,   // [n_work] cap owning this tile
    const int* __restrict__ work_row,   // [n_work] tile row start, within cap
    const int* __restrict__ work_col,   // [n_work] tile col start, in d_out
    const int* __restrict__ offsets,    // [n_caps + 1]
    const int d_in,
    const int d_out)
{
    const int w = blockIdx.x;
    const int cap = work_cap[w];

    const int row_base = offsets[cap];                 // first row of this cap
    const int rows_in_cap = offsets[cap + 1] - row_base;

    const int ty = threadIdx.y;   // row within tile
    const int tx = threadIdx.x;   // col within tile

    const int local_row = work_row[w] + ty;            // row within the cap
    const int global_col = work_col[w] + tx;           // column in d_out

    // Tiles at the ragged edge of a cap or of d_out have threads with no
    // work; they still participate in the shared-memory loads below (with
    // zeros) so the whole block can synchronise together.
    const bool row_active = local_row < rows_in_cap;
    const bool col_active = global_col < d_out;

    const int global_row = row_base + local_row;

    // Weight slab for this cap: weights + cap * d_in * d_out
    const float* w_cap = weights + (size_t)cap * d_in * d_out;

    __shared__ float a_tile[TILE][TILE];
    __shared__ float b_tile[TILE][TILE];

    float acc = 0.0f;

    for (int k0 = 0; k0 < d_in; k0 += TILE) {
        // Load one TILE x TILE block of xs and of the weight slab.
        const int a_k = k0 + tx;
        a_tile[ty][tx] = (row_active && a_k < d_in)
            ? xs[(size_t)global_row * d_in + a_k]
            : 0.0f;

        const int b_k = k0 + ty;
        b_tile[ty][tx] = (col_active && b_k < d_in)
            ? w_cap[(size_t)b_k * d_out + global_col]
            : 0.0f;

        __syncthreads();

        // Accumulate in the same order the reference path does: a plain
        // sequential reduction over the k tile. Reduction order is what
        // decides whether results match the portable implementation
        // bit-for-bit, so keep it simple and deterministic.
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += a_tile[ty][k] * b_tile[k][tx];
        }

        __syncthreads();
    }

    if (row_active && col_active) {
        out[(size_t)global_row * d_out + global_col] = acc;
    }
}
