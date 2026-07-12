Tensors
=======

The `msgspec.data` module adds first-class support for encoding and decoding
packed n-dimensional arrays ("tensors", e.g. `numpy`_ arrays) - efficiently as
a MessagePack extension, or as a self-describing JSON object.

``msgspec`` itself never imports ``numpy``; the tensor machinery works through
an opaque handle type and optional hooks, so there is no hard dependency on any
array library.


Dtypes and shapes
-----------------

``msgspec.data`` exports a set of dtype aliases and a ``Tensor`` type
constructor for annotating fields:

.. code-block:: python

    from msgspec import Struct
    from msgspec.data import Tensor, Float32, Int64, Scalar

    class Layer(Struct):
        weights: Tensor[3, Float32]   # a rank-3 float32 tensor
        bias: Scalar                  # a single int/float/bool
        count: Int64

The available dtype aliases are ``UInt8``, ``UInt16``, ``UInt32``, ``UInt64``,
``Int8``, ``Int16``, ``Int32``, ``Int64``, ``Float32``, ``Float64``, and
``Bool``, plus the generic markers ``Int`` (an integer, dtype ``int64``) and
``Float`` (dtype ``float64``). ``Tensor[shape, dtype]`` takes a shape (an
``int`` number of axes, a tuple of sizes, or ``None``) and a dtype (one of the
aliases, or ``None`` for any).

``Int`` and ``Float`` are also useful as *ordinary* field annotations: unlike
plain ``int``/``float`` (both ``number`` in TypeScript), ``Int`` maps to
TypeScript ``bigint`` - preserving the integer/float distinction into JS. The
:doc:`codec <typescript>` decodes these into real bigints (``BigInt(...)``). On
encode, since `@msgpack/msgpack` can't serialize ``bigint`` directly, the codec
narrows to a ``number`` but *throws* if the value exceeds the JS safe-integer
range (``2**53``). Pass ``force_int64=True`` to ``codec`` to instead enable the
msgpack library's ``useBigInt64`` mode and encode full 64-bit ints (larger,
non-compact output).

These are ordinary :doc:`inspectable <inspect>` types. ``msgspec.inspect``
reports them as ``ScalarType`` / ``TensorType`` nodes:

.. code-block:: python

    >>> import msgspec
    >>> msgspec.inspect.type_info(Float32)
    ScalarType(dtype='float32')
    >>> msgspec.inspect.type_info(Tensor[3, Float32])
    TensorType(ndims=3, sizes=None, dtype='float32')


TensorHandle
------------

`msgspec.msgpack.TensorHandle` is the wrapper used to move raw tensor data
through ``msgspec``. It holds ``native`` (any buffer-protocol object on encode;
a ``memoryview`` on decode), a ``dtype`` string, and a ``shape`` tuple.

On **encode**, ``msgspec`` takes a ``memoryview`` of ``native`` to obtain the
raw bytes and writes them - along with the dtype and shape - as a reserved
MessagePack extension (type code ``84``). On **decode**, it produces a
``TensorHandle`` whose ``native`` is a ``memoryview`` onto a fresh ``bytes``
object; it's up to you to convert it.

.. code-block:: python

    import array
    import msgspec
    from msgspec.msgpack import TensorHandle, encode, decode

    buf = array.array("f", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    handle = TensorHandle(buf, dtype="float32", shape=(2, 3))

    data = encode(handle)
    out = decode(data)
    assert isinstance(out.native, memoryview)
    assert out.dtype == "float32"
    assert out.shape == (2, 3)


numpy
-----

If ``numpy`` is installed, ``msgspec`` recognizes ``numpy`` arrays
automatically on encode and wraps them as tensors - no ``TensorHandle`` needed:

.. code-block:: python

    import numpy as np
    import msgspec

    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    data = msgspec.msgpack.encode(arr)          # encoded as a tensor extension

This detection is dependency-free: if the value is a ``numpy`` array then
``numpy`` is necessarily already imported, so ``msgspec`` only ever looks it up
lazily.

To decode straight back to a ``numpy`` array (rather than a ``TensorHandle``),
pass a ``dec_tensor`` hook. It's called as ``dec_tensor(shape, dtype, data)``
where ``data`` is a ``bytes`` object, and its return value becomes the decoded
tensor:

.. code-block:: python

    def to_numpy(shape, dtype, data):
        arr = np.frombuffer(data, dtype=np.dtype(dtype))
        return arr.reshape(shape) if shape is not None else arr

    dec = msgspec.msgpack.Decoder(SomeStruct, dec_tensor=to_numpy)


JSON
----

Tensors also encode to JSON, as a self-describing object with the raw bytes
base64-encoded:

.. code-block:: python

    >>> msgspec.json.encode(TensorHandle(b"\x00\x01\x02\x03", dtype="uint8", shape=(2, 2)))
    b'{"type":"tensor","shape":[2,2],"dtype":"uint8","data":"AAECAw=="}'

Decoding a JSON tensor object works the same way as MessagePack when the target
type is a tensor (``Tensor[...]`` or ``TensorHandle``), including the
``dec_tensor`` hook:

.. code-block:: python

    msgspec.json.decode(msg, type=Tensor[2, Float32], dec_tensor=to_numpy)


.. note::

    When decoding as ``Any`` (rather than a tensor-typed target), a JSON tensor
    object decodes to a plain ``dict``. The ``"type": "tensor"`` tag is present
    so such objects can still be recognized. MessagePack tensors, whose
    extension code is self-identifying, always decode to a ``TensorHandle``.


.. _numpy: https://numpy.org/
