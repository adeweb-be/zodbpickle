##############################################################################
#
# Copyright (c) Zope Foundation and Contributors.
# All Rights Reserved.
#
# This software is subject to the provisions of the Zope Public License,
# Version 2.1 (ZPL).  A copy of the ZPL should accompany this distribution.
# THIS SOFTWARE IS PROVIDED "AS IS" AND ANY AND ALL EXPRESS OR IMPLIED
# WARRANTIES ARE DISCLAIMED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF TITLE, MERCHANTABILITY, AGAINST INFRINGEMENT, AND FITNESS
# FOR A PARTICULAR PURPOSE.
#
##############################################################################
"""Pickle and unpickle with the C extension from many threads at once.

Each thread has its own pickler and unpickler. All threads share the
copyreg tables, the extension cache, sys.modules, one dict, one list and one
unpickler. One more thread changes these shared objects all the time.

On the free-threaded build the test runs without the GIL. If the extension
shares state in an unsafe way, the interpreter crashes, or a thread gets an
error or bytes that are not the same as the bytes of a single thread.
"""
import copyreg
import functools
import io
import sys
import threading
import types
import unittest

from . import _is_pure
from . import _is_pypy


try:
    from zodbpickle import _pickle
except ImportError:  # pragma: no cover
    _pickle = None


READERS = 8
ROUNDS = 150
EXT_CODE = 0x5A0E0451
OTHER_CODE = 0x5A0E0452
HIGHEST_PROTOCOL = 3


class _Ext:
    """A class that the extension registry knows."""


class _Other:
    """A class that the changer thread adds to the registry and removes."""


class _Reduced:
    """A class that copyreg.dispatch_table reduces."""

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return type(other) is _Reduced and other.value == self.value


def _reduce_reduced(obj):
    return _Reduced, (obj.value,)


class _Plain:
    """A class that the default __reduce_ex__ pickles."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __eq__(self, other):
        return type(other) is _Plain and other.__dict__ == self.__dict__


class _PersRef:
    """A reference that persistent_id writes and persistent_load reads."""

    def __init__(self, oid):
        self.oid = oid

    def __eq__(self, other):
        return type(other) is _PersRef and other.oid == self.oid


def _orphan():
    """A function with no __module__. The pickler finds it in sys.modules."""


_orphan.__module__ = None


def _payload():
    shared = [1, 2, 3]
    return {
        'ints': [0, 1, -1, 255, 256, 65535, 65536, 2**31 - 1, -2**31,
                 2**63, -2**100],
        'floats': [0.0, -1.5, 1e300],
        'text': ['', 'abc', '\xe9t\xe9', '€'],
        'bytes': [b'', b'abc', bytes(range(256))],
        'consts': [True, False, None],
        'nested': {'t': (1, (2, (3,))), 'l': [[1, 2], {'k': None}]},
        'shared': (shared, shared),
        'ext': _Ext,
        'reduced': _Reduced(7),
        'plain': _Plain(a=1, b='two'),
        'orphan': _orphan,
        'mapped': functools.reduce,
        'refs': [_PersRef('oid-1'), _PersRef('oid-2')],
    }


@unittest.skipIf(_pickle is None or _is_pure or _is_pypy,
                 "needs the C extension")
class TestManyThreads(unittest.TestCase):

    def setUp(self):
        class Pickler(_pickle.Pickler):
            def persistent_id(self, obj):
                if type(obj) is _PersRef:
                    return obj.oid
                return None

        class Unpickler(_pickle.Unpickler):
            def persistent_load(self, oid):
                # Protocol 0 gives the id as bytes.
                if isinstance(oid, bytes):
                    oid = oid.decode('ascii')
                return _PersRef(oid)

        self.Pickler = Pickler
        self.Unpickler = Unpickler
        copyreg.add_extension(__name__, '_Ext', EXT_CODE)
        copyreg.pickle(_Reduced, _reduce_reduced)
        self.addCleanup(copyreg.remove_extension, __name__, '_Ext', EXT_CODE)
        self.addCleanup(copyreg.dispatch_table.pop, _Reduced, None)
        self.addCleanup(sys.setswitchinterval, sys.getswitchinterval())
        sys.setswitchinterval(1e-6)

    def _dumps(self, obj, proto):
        f = io.BytesIO()
        self.Pickler(f, proto).dump(obj)
        return f.getvalue()

    def _loads(self, data):
        return self.Unpickler(io.BytesIO(data)).load()

    def test_shared_tables_and_containers(self):
        payload = _payload()
        expected = {proto: self._dumps(payload, proto)
                    for proto in range(HIGHEST_PROTOCOL + 1)}
        for proto, data in expected.items():
            self.assertEqual(self._loads(data), payload, proto)

        shared_dict = {i: [i] for i in range(64)}
        shared_list = [[i] for i in range(64)]
        shared_unpickler = _pickle.Unpickler(io.BytesIO(b''))
        barrier = threading.Barrier(READERS + 1)
        stop = threading.Event()
        errors = []

        def change():
            barrier.wait()
            i = 0
            while not stop.is_set():
                i += 1
                # A new reducer object replaces the old one, which is freed.
                copyreg.pickle(_Reduced, functools.partial(_reduce_reduced))
                copyreg.add_extension(__name__, '_Other', OTHER_CODE)
                copyreg.remove_extension(__name__, '_Other', OTHER_CODE)
                copyreg.clear_extension_cache()
                name = '_zodbpickle_ft_dummy_%d' % (i % 8)
                sys.modules[name] = types.ModuleType(name)
                del sys.modules[name]
                shared_dict[i % 64] = [i]
                shared_dict.pop((i + 32) % 64, None)
                shared_list.append([i])
                if len(shared_list) > 96:
                    del shared_list[:32]

        def read():
            barrier.wait()
            for _ in range(ROUNDS):
                for proto in range(HIGHEST_PROTOCOL + 1):
                    data = self._dumps(payload, proto)
                    if data != expected[proto]:
                        errors.append('bytes differ, protocol %d' % proto)
                    if self._loads(data) != payload:
                        errors.append('round trip differs, protocol %d'
                                      % proto)
                    for shared in (shared_dict, shared_list):
                        # The changer can change the size while the pickler
                        # reads. The pickler can then raise, but it must
                        # not crash.
                        try:
                            data = self._dumps(shared, proto)
                        except (RuntimeError, IndexError):
                            continue
                        if type(self._loads(data)) is not type(shared):
                            errors.append('wrong type, protocol %d' % proto)
                # A fresh unpickler has protocol 0, so find_class() reads
                # the name mapping tables of _compat_pickle.
                if shared_unpickler.find_class(
                        '__builtin__', 'reduce') is not functools.reduce:
                    errors.append('find_class: wrong reduce')
                if shared_unpickler.find_class(
                        'copy_reg', '_reconstructor') is not \
                        copyreg._reconstructor:
                    errors.append('find_class: wrong _reconstructor')

        def run(target):
            try:
                target()
            except BaseException as e:  # pragma: no cover
                errors.append(repr(e))

        changer = threading.Thread(target=run, args=(change,))
        readers = [threading.Thread(target=run, args=(read,))
                   for _ in range(READERS)]
        changer.start()
        for t in readers:
            t.start()
        for t in readers:
            t.join()
        stop.set()
        changer.join()
        self.assertEqual(errors[:5], [])


def test_suite():
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
