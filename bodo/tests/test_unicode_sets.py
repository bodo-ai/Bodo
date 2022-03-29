# Copyright (C) 2022 Bodo Inc.
import pytest

from bodo.tests.utils import check_func


_non_repeating_data = [f"A{i}" for i in range(0, 100)]

_non_repeating_data_subset = _non_repeating_data[::2]

_repeating_data = [f"A{i}" for i in range(0, 100)] * 3

_repeating_data_subset = _repeating_data[::2]

@pytest.mark.parametrize("arg", [_non_repeating_data, _repeating_data])
def test_unicode_set_constructor(arg, memory_leak_check):
    """
    Test constructing a unicode set.
    """

    def impl(arg):
        s = set(arg)
        return len(s)

    check_func(impl, (arg,))


@pytest.mark.parametrize("arg", [_non_repeating_data, _repeating_data])
def test_unicode_set_return(arg, memory_leak_check):
    """
    Test returning a unicode set.
    """

    def impl(arg):
        s = set(arg)
        return s

    check_func(impl, (arg,))


@pytest.mark.parametrize("arg", [_non_repeating_data, _repeating_data])
def test_unicode_set_iterator(arg, memory_leak_check):
    """
    Test iterating over a unicode set.
    """

    def impl(arg):
        s = set(arg)
        l = []
        for v in s:
            l.append(v)
        return len(l)

    check_func(impl, (arg,))


@pytest.mark.parametrize("arg", [_non_repeating_data, _repeating_data])
def test_unicode_set_update(arg, memory_leak_check):
    """
    Test updating a unicode set.
    """

    def impl(arg):
        s = set()
        s.update(arg)
        return len(s)

    check_func(impl, (arg,))


def _set_remove_impl(to_add, to_remove):
    s = set(to_add)
    for v in to_remove:
        s.remove(v)
    return len(s)

@pytest.mark.parametrize("to_add,to_remove", [
    (_non_repeating_data, _non_repeating_data),
    (_non_repeating_data, _non_repeating_data_subset),
])
def test_unicode_set_remove(to_add, to_remove, memory_leak_check):
    """
    Test removing values from a unicode set.
    """

    check_func(_set_remove_impl, (to_add, to_remove))


@pytest.mark.parametrize("to_add,to_remove", [
    (_non_repeating_data_subset, _non_repeating_data),
    (_repeating_data_subset, _repeating_data),
])
def test_unicode_set_remove_error(to_add, to_remove, memory_leak_check):
    """
    Test removing a value from a unicode set that doesn't exist.
    """

    with pytest.raises(KeyError):
        _set_remove_impl(to_add, to_remove)


@pytest.mark.parametrize("to_add,to_discard", [
    (_non_repeating_data, _non_repeating_data),
    (_non_repeating_data, _non_repeating_data_subset),
    (_non_repeating_data_subset, _non_repeating_data),
    (_non_repeating_data_subset, _non_repeating_data_subset),
    (_repeating_data, _repeating_data),
    (_repeating_data, _repeating_data_subset),
    (_repeating_data_subset, _repeating_data),
    (_repeating_data_subset, _repeating_data_subset),
])
def test_unicode_set_discard(to_add, to_discard, memory_leak_check):
    """
    Test discarding values from a unicode set.
    """

    def impl(to_add, to_discard):
        s = set(to_add)
        for v in to_discard:
            s.discard(v)
        return len(s)

    check_func(impl, (to_add, to_discard))


@pytest.mark.parametrize("arg", [_non_repeating_data, _repeating_data])
def test_unicode_set_pop(arg, memory_leak_check):
    """
    Test popping a value from a unicode set.
    """

    def impl(arg):
        s = set(arg)
        l = []
        while len(s) > 0:
            l.append(s.pop())
        return len(l)

    check_func(impl, (arg,))


@pytest.mark.parametrize("to_add,to_check", [
    (_non_repeating_data, _non_repeating_data),
    (_non_repeating_data, _non_repeating_data_subset),
    (_non_repeating_data_subset, _non_repeating_data),
    (_non_repeating_data_subset, _non_repeating_data_subset),
    (_repeating_data, _repeating_data),
    (_repeating_data, _repeating_data_subset),
    (_repeating_data_subset, _repeating_data),
    (_repeating_data_subset, _repeating_data_subset),
])
def test_unicode_set_contains(to_add, to_check, memory_leak_check):
    """
    Test checking if a value is in a unicode set.
    """

    def impl(to_add, to_check):
        s = set(to_add)
        l = []
        for v in to_check:
            l.append(v in s)
        return l

    check_func(impl, (to_add, to_check))


@pytest.mark.parametrize("arg", [_non_repeating_data, _repeating_data])
def test_unicode_set_clear(arg, memory_leak_check):
    """
    Test clearing a unicode set.
    """

    def impl(arg):
        s = set(arg)
        s.clear()
        return len(s)

    check_func(impl, (arg,))


@pytest.mark.parametrize("arg", [_non_repeating_data, _repeating_data])
def test_unicode_set_copy(arg, memory_leak_check):
    """
    Test copying a unicode set.
    """

    def impl(arg):
        s = set(arg)
        s2 = s.copy()
        s.pop()
        return len(s2)

    check_func(impl, (arg,))
