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
``Bool``, plus the generic markers ``Int``/``UInt`` (a signed / unsigned
integer, dtype ``int64``/``uint64``) and ``Float`` (dtype ``float64``).
``Tensor[shape, dtype]`` takes a shape (an ``int`` number of axes, a tuple of
sizes, or ``None``) and a dtype (one of the aliases, or ``None`` for any).

The integer/float markers double as *ordinary* field annotations that carry a
precise wire format - see :ref:`type-directed-encoding` below. In short:
``Int``/``UInt``/``Int64``/``UInt64`` become JavaScript ``bigint`` (round-tripped
losslessly, as hex strings in JSON when large), and ``Float32`` narrows to a
5-byte MessagePack float32. ``Scalar`` is the union ``Int | Float | Bool`` - a
cheap way to say "encode this scalar accurately" (int → ``bigint``, float →
``number``, bool → ``boolean`` on the JS side).

These are ordinary :doc:`inspectable <inspect>` types. As *scalar* field types
the integer/float markers carry their format inline on ``IntType``/``FloatType``
(via ``itype``/``ftype``); inside a ``Tensor`` they set the tensor ``dtype``:

.. code-block:: python

    >>> import msgspec
    >>> msgspec.inspect.type_info(Float32)
    FloatType(gt=None, ge=None, lt=None, le=None, multiple_of=None, ftype='float32')
    >>> msgspec.inspect.type_info(Int64)
    IntType(gt=None, ge=None, lt=None, le=None, multiple_of=None, itype='int64')
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


JavaScript / TypeScript
-----------------------

The :doc:`JavaScript <javascript>` and :doc:`TypeScript <typescript>` codecs
support tensors natively (in the default embedded runtime - no third-party
dependency). A ``Tensor[...]`` field decodes to a ``TensorHandle``: a small
class, re-exported by the generated module, holding the packed ``array`` (a
`typed array`_ picked by dtype - ``Float32Array``, ``BigInt64Array``, ...), the
``dtype`` string, and the ``shape``. It mirrors ``msgspec.data``'s TensorHandle,
so a decoded tensor looks nearly identical across Python and JS:

.. code-block:: javascript

    import { msgpack, TensorHandle } from "./codec.js";

    const doc = msgpack.decode(bytes);
    doc.field.array;   // Float32Array([...])
    doc.field.dtype;   // "float32"
    doc.field.shape;   // [2, 3]

    // construct one to encode
    const t = new TensorHandle(new Float32Array([1, 2, 3, 4, 5, 6]), "float32", [2, 3]);

Both the MessagePack ext form and the JSON object form are byte-identical to
``msgspec``'s. To decode into a third-party ndarray type instead, pass
``tensor_encoder`` / ``tensor_decoder`` (the names of JS adapter functions
converting your type to/from a ``TensorHandle``).


.. _numpy: https://numpy.org/
.. _typed array: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Typed_arrays
