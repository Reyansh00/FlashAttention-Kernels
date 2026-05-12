#include <torch/extension.h>

torch::Tensor online_warp_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v);

torch::Tensor online_warp(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(v.is_cuda(), "v must be CUDA");
    TORCH_CHECK(q.dtype() == torch::kFloat32, "use float32 first");
    TORCH_CHECK(k.dtype() == torch::kFloat32, "use float32 first");
    TORCH_CHECK(v.dtype() == torch::kFloat32, "use float32 first");
    return online_warp_cuda(q, k, v);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &online_warp, "Online warp forward (CUDA)");
}