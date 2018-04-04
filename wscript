def depends(ctx):
    ctx('frickel-dls')
    ctx('libnux')


def options(opt):
    pass


def configure(conf):
    pass


def build(bld):
    # manual dependency on pydlsnew...
    bld.get_tgen_by_name('_pydlsnew').post()
    if bld.cmd == 'install':
        bld.get_tgen_by_name('pydlsnew').post()

    bld.program(
        features = 'c objcopy',
        objcopy_bfdname = 'binary',
        target = 'template_test.bin',
        source = 'src/ppu/test_project/test.c',
        use = ['nux', 'nux_runtime'],
        env = bld.all_envs['nux'],
    )

    bld.program(
        features = 'cxx objcopy',
        objcopy_bfdname = 'binary',
        target = 'template_HelloWorld.bin',
        source = ['src/ppu/test_project/HelloWorld.cpp'],
        use = ['nux', 'nux_runtime_cpp'],
        env = bld.all_envs['nux'],
    )

    bld.program(
        features = 'cxx',
        target = 'template_host.bin',
        source = ['src/cc/test_project/test-host-code.cpp'],
        use = ['frickel_dls'],
    )
