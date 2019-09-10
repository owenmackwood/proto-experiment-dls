from waflib.extras.test_base import summary

def depends(ctx):
    ctx('haldls')
    ctx('libnux')
    ctx('dls2calib')
    ctx('code-format')


def options(opt):
    opt.load("doxygen")
    opt.load('test_base')
    opt.load('pytest')


def configure(conf):
    conf.load("doxygen")
    conf.load('test_base')
    conf.load('pytest')


def build(bld):
    bld(
        target = 'nux_includes',
        export_includes = ['src/ppu'],
        env = bld.all_envs['nux'],
    )

    bld(
        target = 'host_includes',
        export_includes = ['src/cc'],
    )

    bld(
        name = 'template_python_libraries',
        features = 'py use',
        source = bld.path.ant_glob('src/py/**/*.py'),
        relative_trick = True,
        install_from = bld.path.find_node('src/py'),
        install_path = '${PREFIX}/lib',
        use = ['dlens_v2'],
    )

    bld(
        name='template_pyhwtests',
        tests=bld.path.ant_glob('test/hw/py/**/*.py'),
        features='use pytest',
        use='template_python_libraries',
        install_path='${PREFIX}/bin/tests',
        skip_run=False,
    )

    bld.program(
        features = 'c objcopy',
        objcopy_bfdname = 'binary',
        target = 'template_test.bin',
        source = 'src/ppu/test_project/test.c',
        use = ['nux', 'nux_runtime', 'nux_includes'],
        env = bld.all_envs['nux'],
    )

    bld.program(
        features = 'cxx objcopy',
        objcopy_bfdname = 'binary',
        target = 'template_HelloWorld.bin',
        source = ['src/ppu/test_project/HelloWorld.cpp'],
        use = ['nux', 'nux_runtime_cpp', 'nux_includes'],
        env = bld.all_envs['nux'],
    )

    bld.program(
        features = 'cxx cxxprogram',
        target = 'template_host',
        source = ['src/cc/test_project/test-host-code.cpp'],
        use = ['haldls_v2', 'host_includes'],
    )

    bld(
        features = 'doxygen',
        doxyfile = bld.root.make_node('%s/code-format/doxyfile' % bld.env.PREFIX),
        install_path = 'doc/template-experiment-dls',
        pars = {
            "PROJECT_NAME": "\"Template-Experiment DLS\"",
            "INPUT": "%s/template-experiment-dls/src/cc" % bld.env.PREFIX
        },
    )

    bld.add_post_fun(summary)
