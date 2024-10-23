from bodo.submit.spawner import submit_jit

# TODO(aneesh): this print stmt helps us track how many times mymodule was
# imported. In the future, when we can pass environment variables across the
# spawner/executor, we should have this module write to a file controlled by an
# envvar to count how many times this module was imported across all processes.
# Then we can read that file from the test and assert that it was only imported
# once per executor plus once on the spawner.
print("imported mymodule")


def f():
    # TODO(aneesh): replace this print with returning an actual value
    print("called mymodule.f")


@submit_jit
def jit_fn():
    print("called mymodule.jit_fn")
