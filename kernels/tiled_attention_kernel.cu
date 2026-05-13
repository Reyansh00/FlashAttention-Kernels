#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void tiled_attention_kernel(
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
    float* k_s = shared_mem + D; // [D]
    float* v_s = shared_mem + 2*D; // [D]
    float* scores_s = shared_mem + 3*D; // 

    q_s[d] = q[(b * N + n) * D + d];
    __syncthreads();
    
    float* out_ptr = out + (b * N + n) * D;

    float scale = rsqrtf((float)D);

    // 1) max for numerical stability
    float max_score = -1e20f;

    for(int j=0; j<N; j++){
        k_s[d] = k[(b * N + j) * D + d];
        v_s[d] = v[(b * N + j) * D + d];
        __syncthreads();
        float tmp = q_s[d] * k_s[d];
        scores_s[d] = tmp;
        __syncthreads();
        for(int stride = D/2; stride > 0; stride /= 2){
            if(d < stride)
                scores_s[d] += scores_s[d + stride];

            __syncthreads();
        }
        float score = scores_s[0] * scale;
        if(score > max_score) max_score = score;
        
    }
    __syncthreads();
    // 2) softmax denominator + weighted sum for one output dimension d
    float denom = 0.0f;
    float numer = 0.0f;

    for (int j = 0; j < N; j++) {
        k_s[d] = k[(b * N + j) * D + d];
        v_s[d] = v[(b * N + j) * D + d];
        __syncthreads();
        float tmp = q_s[d] * k_s[d];
        scores_s[d] = tmp;
        __syncthreads();
        for(int stride = D/2; stride > 0; stride /= 2){
            if(d < stride)
                scores_s[d] += scores_s[d + stride];

            __syncthreads();
        }
        float score = scores_s[0] * scale;
        float w = expf(score - max_score);
        denom += w;
        numer += w * v_s[d];
    }

    out_ptr[d] = numer / denom;
}

torch::Tensor tiled_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();

    int B = q.size(0);
    int N = q.size(1);
    int D = q.size(2);

    auto out = torch::empty_like(q);

    int blocks = B*N; // one block per output row
    int threads = D; // one thread per output dimension
    size_t smem_bytes = 4*D*sizeof(float); // shared memory for q_row, k_row, v_row, scores_row

    tiled_attention_kernel<<<blocks, threads, smem_bytes>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        B, N, D
    );

    return out;
}