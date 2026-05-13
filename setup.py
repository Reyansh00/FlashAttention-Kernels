from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="attention_extensions",
    ext_modules=[
        CUDAExtension(
            name="naive_attention",
            sources=[
                "cpp_extensions/naive_attention.cpp",
                "kernels/naive_attention_kernel.cu",
            ],
        ),
        CUDAExtension(
            name="tiled_attention",
            sources=[
                "cpp_extensions/tiled_attention.cpp",
                "kernels/tiled_attention_kernel.cu",
            ],
        ),
        CUDAExtension(
            name="tiled_attention_2",
            sources=[
                "cpp_extensions/tiled_attention_2.cpp",
                "kernels/tiled_attention_kernel_2.cu",
            ],
        ),
        CUDAExtension(
            name="online_softmax",
            sources=[
                "cpp_extensions/online_softmax.cpp",
                "kernels/online_softmax_kernel.cu",
            ],
        ),
        CUDAExtension(
            name="online_warp",
            sources=[
                "cpp_extensions/online_warp.cpp",
                "kernels/online_warp_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    options={"build_ext": {"build_lib": "bin"}},
)