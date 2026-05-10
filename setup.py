from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="attention_extensions",
    ext_modules=[
        CUDAExtension(
            name="naive_attention",
            sources=[
                "naive_attention.cpp",
                "naive_attention_kernel.cu",
            ],
        ),
        CUDAExtension(
            name="tiled_attention",
            sources=[
                "tiled_attention.cpp",
                "tiled_attention_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)