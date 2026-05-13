#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

constexpr int TILE_N = 16;
constexpr int MAX_WARPS = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    // Full-warp reduction. Caller ensures the participating threads are
    // within the same warp and that inactive lanes contribute 0 when needed.
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void online_warp_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B, int N, int D
) {
    int n = blockIdx.x % N;
    int b = blockIdx.x / N;
    int d = threadIdx.x;

    if (d >= D) return;

    const int warp_id = d >> 5;
    const int lane = d & 31;
    const int num_warps = (D + 31) >> 5;

    extern __shared__ float shared_mem[];
    float* q_s = shared_mem;                         // [D]
    float* k_s = q_s + D;                            // [TILE_N * D]
    float* v_s = k_s + TILE_N * D;                   // [TILE_N * D]
    float* row_scores = v_s + TILE_N * D;            // [TILE_N]
    float* warp_sums = row_scores + TILE_N;          // [MAX_WARPS]

    q_s[d] = q[(b * N + n) * D + d];
    __syncthreads();

    float* out_ptr = out + (b * N + n) * D;
    const float scale = rsqrtf((float)D);

    float max_score = -1e20f;
    float denom = 0.0f;
    float numer = 0.0f;

    for (int tile_start = 0; tile_start < N; tile_start += TILE_N) {
        // Load K/V tile into shared memory.
        for (int local_j = 0; local_j < TILE_N; ++local_j) {
            int global_j = tile_start + local_j;
            if (global_j < N) {
                k_s[local_j * D + d] = k[(b * N + global_j) * D + d];
                v_s[local_j * D + d] = v[(b * N + global_j) * D + d];
            }
        }
        __syncthreads();

        // Compute one score per tile row using warp-level reduction.
        for (int local_j = 0; local_j < TILE_N; ++local_j) {
            int global_j = tile_start + local_j;
            if (global_j < N) {
                float partial = q_s[d] * k_s[local_j * D + d];
                partial = warp_reduce_sum(partial);

                if (lane == 0) {
                    warp_sums[warp_id] = partial;
                }
            }
            __syncthreads();

            // Combine warp partials into one row score.
            if (warp_id == 0) {
                float block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
                block_sum = warp_reduce_sum(block_sum);
                if (lane == 0) {
                    row_scores[local_j] = block_sum * scale;
                }
            }
            __syncthreads();
        }

        // Online softmax update for this tile.
        float tile_max = -1e20f;
        for (int local_j = 0; local_j < TILE_N; ++local_j) {
            int global_j = tile_start + local_j;
            if (global_j < N) {
                tile_max = fmaxf(tile_max, row_scores[local_j]);
            }
        }

        float max_score_new = fmaxf(max_score, tile_max);
        float alpha = expf(max_score - max_score_new);
        denom *= alpha;
        numer *= alpha;

        for (int local_j = 0; local_j < TILE_N; ++local_j) {
            int global_j = tile_start + local_j;
            if (global_j < N) {
                float w = expf(row_scores[local_j] - max_score_new);
                denom += w;
                numer += w * v_s[local_j * D + d];
            }
        }

        max_score = max_score_new;
        __syncthreads();
    }

    out_ptr[d] = numer / denom;
}


torch::Tensor online_warp_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();

    int B = q.size(0);
    int N = q.size(1);
    int D = q.size(2);

    auto out = torch::empty_like(q);

    int blocks = B * N;  // one block per output row
    int threads = D;     // one thread per output dimension

    // q_s[D] + k_tile[TILE_N * D] + v_tile[TILE_N * D] + row_scores[TILE_N] + warp_sums[MAX_WARPS]
    size_t smem_bytes = (D + 2 * TILE_N * D + TILE_N + MAX_WARPS) * sizeof(float);

    online_warp_kernel<<<blocks, threads, smem_bytes>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        B, N, D
    );

    return out;
}
