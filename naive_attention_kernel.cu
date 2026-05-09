#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void naive_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B, int N, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * D;
    if (idx >= total) return;

    int d = idx % D;
    int tmp = idx / D;
    int n = tmp % N;
    int b = tmp / N;

    const float* q_row = q + (b * N + n) * D;
    const float* k_base = k + b * N * D;
    const float* v_base = v + b * N * D;
    float* out_ptr = out + (b * N + n) * D;

    float scale = rsqrtf((float)D);

    // 1) max for numerical stability
    float max_score = -1e20f;
    for (int j = 0; j < N; j++) {
        const float* k_row = k_base + j * D;
        float score = 0.0f;
        for (int t = 0; t < D; t++) {
            score += q_row[t] * k_row[t];
        }
        score *= scale;
        if (score > max_score) max_score = score;
    }

    // 2) softmax denominator + weighted sum for one output dimension d
    float denom = 0.0f;
    float numer = 0.0f;

    for (int j = 0; j < N; j++) {
        const float* k_row = k_base + j * D;
        const float* v_row = v_base + j * D;

        float score = 0.0f;
        for (int t = 0; t < D; t++) {
            score += q_row[t] * k_row[t];
        }
        score *= scale;

        float w = expf(score - max_score);
        denom += w;
        numer += w * v_row[d];
    }

    out_ptr[d] = numer / denom;
}

torch::Tensor naive_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();

    int B = q.size(0);
    int N = q.size(1);
    int D = q.size(2);

    auto out = torch::empty_like(q);

    int total = B * N * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    naive_attention_kernel<<<blocks, threads>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        B, N, D
    );

    return out;
}