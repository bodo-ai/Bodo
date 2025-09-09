import bodo

if bodo.test_compiler:
    import bodo.libs.uuid


def test_uuid4(memory_leak_check):
    @bodo.jit
    def impl():
        return bodo.libs.uuid.uuidV4()

    a = impl()
    b = impl()
    assert a != b

    # Check that version is 4
    assert a[14] == "4"
    assert b[14] == "4"


def test_uuid5(memory_leak_check):
    @bodo.jit
    def impl(namespace, name):
        return bodo.libs.uuid.uuidV5(namespace, name)

    # Assert that output is determinisitic
    a = impl("fe971b24-9572-4005-b22f-351e9c09274d", "foo")
    assert a == "dc0b6f65-fca6-5b4b-9d37-ccc3fde1f3e2"

    b = impl("fe971b24-9572-4005-b22f-351e9c09274d", "b")
    c = impl("fe971b24-9572-4005-b22f-351e9c09274d", "c")
    # Check that version is 5
    assert b[14] == "5"
    assert c[14] == "5"
    assert b != c


def test_uuid5_invalid_namespace(memory_leak_check):
    @bodo.jit
    def impl(namespace, name):
        return bodo.libs.uuid.uuidV5(namespace, name)

    assert impl("1234", "foo") == ""
