import os
from os.path import join
from waflib.extras.test_base import summary
from waflib.extras.symwaf2ic import get_toplevel_path


EXPERIMENT_NAME: str = "template"
"""
Name of this experiment.

To be changed if this template is used.
The name given to this variable needs to correspond to the library folders in
 * src/cc
 * src/ppu
 * src/py
 * tests/**/cc
 * tests/**/ppu
 * tests/**/py
"""


def depends(ctx):
    ctx("haldls")
    ctx("libnux")
    ctx("code-format")


def options(opt):
    opt.load("test_base")
    opt.load("pytest")


def configure(conf):
    conf.load("test_base")
    conf.load("pytest")

    conf.load("compiler_cxx")
    conf.load("python")


def build(bld):
    bld.env.DLSvx_HARDWARE_AVAILABLE = "cube" == os.environ.get("SLURM_JOB_PARTITION")

    build_host_cpp(bld)
    build_host_python(bld)
    build_ppu_cpp(bld)

    bld.add_post_fun(summary)


def build_host_cpp(bld):
    """
    Waf build targets for C++ code running on the host system.
    """
    bld(target=f"{EXPERIMENT_NAME}-host_includes",
        export_includes=["src/cc"])

    bld.program(name=f"{EXPERIMENT_NAME}-host_helloworld",
                features="cxx cxxprogram",
                target="hello_world",
                source=[f"src/cc/{EXPERIMENT_NAME}/hello_world.cpp"],
                use=[f"{EXPERIMENT_NAME}-host_includes"])


def build_host_python(bld):
    """
    Waf build targets for python code running on the host system.
    """
    bld(name=f"{EXPERIMENT_NAME}-python_libraries",
        features="py use pylint pycodestyle",
        source=bld.path.ant_glob("src/py/**/*.py"),
        relative_trick=True,
        install_path="${PREFIX}/lib",
        install_from="src/py",
        pylint_config=join(get_toplevel_path(), "code-format", "pylintrc"),
        pycodestyle_config=join(get_toplevel_path(), "code-format", "pycodestyle"),
        use=["dlens_vx"])

    bld(name=f"{EXPERIMENT_NAME}-python_hwtests",
        tests=bld.path.ant_glob("tests/hw/py/**/*.py"),
        features="use pytest pylint pycodestyle",
        use=f"{EXPERIMENT_NAME}-python_libraries",
        install_path="${PREFIX}/bin/tests/hw",
        pylint_config=join(get_toplevel_path(), "code-format", "pylintrc"),
        pycodestyle_config=join(get_toplevel_path(), "code-format", "pycodestyle"),
        skip_run=not bld.env.DLSvx_HARDWARE_AVAILABLE)

    bld(name=f"{EXPERIMENT_NAME}-python_swtests",
        tests=bld.path.ant_glob("tests/sw/py/**/*.py"),
        features="use pytest pylint pycodestyle",
        use=f"{EXPERIMENT_NAME}-python_libraries",
        install_path="${PREFIX}/bin/tests/sw",
        pylint_config=join(get_toplevel_path(), "code-format", "pylintrc"),
        pycodestyle_config=join(get_toplevel_path(), "code-format", "pycodestyle"))


def build_ppu_cpp(bld):
    """
    Waf build targets for C++ code to be compiled for the embedded plasticity
    processor.
    """
    bld(target=f"{EXPERIMENT_NAME}-ppu_includes",
        export_includes=["src/ppu"],
        env=bld.all_envs["nux_vx"])

    bld.program(name=f"{EXPERIMENT_NAME}-ppu_helloworld",
                features="cxx",
                target="hello_world.bin",
                source=[f"src/ppu/{EXPERIMENT_NAME}/hello_world.cpp"],
                use=["nux_vx", "nux_runtime_vx",
                     f"{EXPERIMENT_NAME}-ppu_includes"],
                env=bld.all_envs["nux_vx"])
