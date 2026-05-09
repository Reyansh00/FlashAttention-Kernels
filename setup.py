from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="naive_attention",
    ext_modules=[
        CUDAExtension(
            name="naive_attention",
            sources=["naive_attention.cpp", "naive_attention_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)