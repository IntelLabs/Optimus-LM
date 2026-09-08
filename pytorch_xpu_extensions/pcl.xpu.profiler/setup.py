import os
import torch
import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import SyclExtension, BuildExtension

# os.environ["TORCH_XPU_ARCH_LIST"] = ""

def _wrap_sycl_host_flags(cflags):
    #print("cflags: ", cflags)
    return ""
    # host_cxx = get_cxx_compiler()
    # host_cflags = [
    #     f'-fsycl-host-compiler={host_cxx}',
    #     shlex.quote(f'-fsycl-host-compiler-options={cflags}'),
    # ]
    # return host_cflags

torch.utils.cpp_extension._wrap_sycl_host_flags = _wrap_sycl_host_flags


library_name = "pcl_profiler"
extra_compile_args = {
    "cxx": ["-O3",
            "-fdiagnostics-color=always"],
    "sycl": ["-O3", "-std=c++20"]
}

# assert(torch.xpu.is_available()), "XPU is not available, please check your environment"
# Source files collection
sources = ['csrc/profiler/profile.dp.cpp']
# Construct extension
ext_modules = [
    SyclExtension(
        library_name,
        sources,
        extra_compile_args=extra_compile_args
    )
]
setup(
    name=library_name,
    ext_modules=ext_modules,
    install_requires=["torch"],
    description="Simple Example of PyTorch Sycl extensions",
    cmdclass={"build_ext": BuildExtension}
)



"""
from setuptools import setup
from intel_extension_for_pytorch.xpu.cpp_extension import DPCPPExtension, DpcppBuildExtension

def profiler_sources():
    return [
        'csrc/profiler/profile.dp.cpp',
    ]

setup(
    name='pcl_profiler',
    ext_modules=[
        DPCPPExtension(
            name='pcl_profiler',
            sources=profiler_sources(),
            extra_compile_args=["-w"],
        ),
    ],
    cmdclass={
        'build_ext': DpcppBuildExtension
})
"""