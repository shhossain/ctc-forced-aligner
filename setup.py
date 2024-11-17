from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

ext_modules = [
    Pybind11Extension(
        "ctc_forced_aligner._ctc_forced_align",  # Note the underscore
        ["ctc_forced_aligner/forced_align_impl.cpp"],
        extra_compile_args=["-O3"],
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
