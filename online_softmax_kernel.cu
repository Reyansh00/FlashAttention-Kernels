#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

constexpr int TILE_N = 16;

__global__ void online_softmax_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B, int N, int D
) {
    int n = blockIdx.x % N;
    int b = blockIdx.x / N;
    int d = threadIdx.x;

    extern __shared__ float shared_mem[];
    float* q_s = shared_mem; // [D]
    float* k_s = shared_mem + D; // [D]*TILE_N
    float* v_s = k_s + TILE_N*D ; // [D]*TILE_N
    float* scores_s = v_s + TILE_N*D ; // [D]*TILE_N

    q_s[d] = q[(b * N + n) * D + d];
    __syncthreads();
    
    float* out_ptr = out + (b * N + n) * D;

    float scale = rsqrtf((float)D);

    // 1) max for numerical stability
    float max_score = -1e20f;
    float max_score_new;
    float tile_max;
    float denom = 0.0f;
    float numer = 0.0f;
    float alpha;

    for(int tile_start = 0; tile_start < N; tile_start += TILE_N){
        // load tile
        tile_max = -1e20f;
        for(int local_j = 0; local_j < TILE_N; local_j++){
            int global_j = tile_start + local_j;
            if(global_j < N){
                k_s[local_j * D + d] =
                    k[(b * N + global_j) * D + d];
                v_s[local_j * D + d] =
                    v[(b * N + global_j) * D + d];
            }
        }
        __syncthreads();
        // compute partial dot products
        for(int local_j = 0; local_j < TILE_N; local_j++){
            int global_j = tile_start + local_j;
            if(global_j < N){
                float tmp = q_s[d] * k_s[local_j * D + d];
                scores_s[local_j * D + d] = tmp;
            }
        }
        __syncthreads();
        // reduction per tile row
        for(int stride = D/2; stride > 0; stride /= 2){
            if(d < stride){
                for(int local_j = 0; local_j < TILE_N; local_j++){
                    int global_j = tile_start + local_j;
                    if(global_j < N){
                        scores_s[local_j * D + d] +=
                            scores_s[local_j * D + d + stride];
                    }
                }
            }
            __syncthreads();
        }

        // update max
        for(int local_j = 0; local_j < TILE_N; local_j++){
            int global_j = tile_start + local_j;
            if(global_j < N){
                float score = scores_s[local_j * D] * scale;
                if(score > tile_max)
                    tile_max = score;
            }
        }
        max_score_new = fmax(tile_max, max_score);
        alpha = expf(max_score - max_score_new);
        denom = denom * alpha;
        numer = numer * alpha;
        __syncthreads();

        for(int local_j = 0; local_j < TILE_N; local_j++){
            int global_j = tile_start + local_j;
            if(global_j < N){
                float score = scores_s[local_j * D] * scale;
                float exp_score = expf(score - max_score_new);
                denom += exp_score;
                numer += exp_score * v_s[local_j * D + d];
            }
        }

        max_score = max_score_new;
        __syncthreads();
    }   

    // 2) softmax denominator + weighted sum for one output dimension d

    out_ptr[d] = numer / denom;
}

torch::Tensor online_softmax_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();

    int B = q.size(0);
    int N = q.size(1);
    int D = q.size(2);

    auto out = torch::empty_like(q);

    int blocks = B*N; // one block per output row
    int threads = D; // one thread per output dimension
    size_t smem_bytes = (D + 3 * TILE_N * D) * sizeof(float); // shared memory for q_row, k_tile, v_tile, scores_tile

    online_softmax_kernel<<<blocks, threads, smem_bytes>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        B, N, D
    );

    return out;
}