"""Native TCP CPU collective communicator (MPI-free ``gloo`` backend).

Uses Jittor's existing Store (``TCPStore`` / ``FileStore``) for rendezvous and
persistent full-mesh TCP sockets for collectives and point-to-point transfers.
No ``mpicc`` or OpenMPI installation is required.
"""

import os
import socket
import struct
import threading

import numpy as np

import jittor as jt


_HEADER_STRUCT = struct.Struct("!QQ")
_RANK_STRUCT = struct.Struct("!I")
_COLLECTIVE_TAG = 0x4A54474C  # "JTGL"


def _resolve_advertise_host():
    explicit = os.environ.get("JT_GLOO_ADDR", "").strip()
    if explicit:
        return explicit, "0.0.0.0"
    master = os.environ.get("MASTER_ADDR", "").strip()
    if not master or master in ("127.0.0.1", "localhost", "::1"):
        return "127.0.0.1", "127.0.0.1"
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        port = int(os.environ.get("MASTER_PORT", "29500") or "29500")
        probe.connect((master, port))
        local_ip = probe.getsockname()[0]
        return local_ip, "0.0.0.0"
    except OSError:
        return "127.0.0.1", "127.0.0.1"
    finally:
        probe.close()


def _send_exact(sock, data):
    view = memoryview(data)
    if view.format != "B":
        view = view.cast("B")
    total = len(view)
    sent = 0
    while sent < total:
        n = sock.send(view[sent:])
        if n <= 0:
            raise ConnectionError("Gloo TCP peer closed connection during send")
        sent += n


def _recv_exact(sock, nbytes):
    buf = bytearray(nbytes)
    view = memoryview(buf)
    received = 0
    while received < nbytes:
        n = sock.recv_into(view[received:], nbytes - received)
        if n <= 0:
            raise ConnectionError(
                "Gloo TCP peer closed connection during recv "
                "(got %d of %d bytes)" % (received, nbytes)
            )
        received += n
    return buf


def _send_frame(sock, tag, payload):
    view = memoryview(payload)
    if view.format != "B":
        view = view.cast("B")
    header = _HEADER_STRUCT.pack(int(tag) & 0xFFFFFFFFFFFFFFFF, len(view))
    _send_exact(sock, header)
    if len(view):
        _send_exact(sock, view)


def _recv_frame(sock, expected_tag=None, expected_nbytes=None):
    raw_header = _recv_exact(sock, _HEADER_STRUCT.size)
    tag, nbytes = _HEADER_STRUCT.unpack(raw_header)
    if expected_tag is not None and tag != (int(expected_tag) & 0xFFFFFFFFFFFFFFFF):
        raise RuntimeError(
            "Gloo frame tag mismatch: expected %d, got %d" % (expected_tag, tag)
        )
    if expected_nbytes is not None and nbytes != expected_nbytes:
        raise RuntimeError(
            "Gloo frame size mismatch: expected %d bytes, got %d bytes"
            % (expected_nbytes, nbytes)
        )
    return _recv_exact(sock, nbytes) if nbytes else bytearray()


def _reduce_numpy(acc, other, operation):
    if operation in ("sum", "mean"):
        if acc.dtype == np.bool_:
            return np.logical_or(acc, other)
        np.add(acc, other, out=acc)
        return acc
    if operation == "max":
        np.maximum(acc, other, out=acc)
        return acc
    if operation == "min":
        np.minimum(acc, other, out=acc)
        return acc
    if operation == "product":
        if acc.dtype == np.bool_:
            return np.logical_and(acc, other)
        np.multiply(acc, other, out=acc)
        return acc
    raise ValueError("unsupported Gloo all_reduce operation {!r}".format(operation))


def _var_to_host_numpy(var, upcast_low_fp=False):
    dtype_str = str(var.dtype)
    shape = tuple(int(dim) for dim in var.shape)
    was_cuda = bool(getattr(var, "is_cuda", False))
    work = var
    if dtype_str == "bfloat16" or (upcast_low_fp and dtype_str == "float16"):
        work = var.cast("float32")
    arr = np.ascontiguousarray(work.numpy())
    return arr, shape, dtype_str, was_cuda


def _host_numpy_to_var(arr, dtype_str, was_cuda):
    out = jt.array(arr)
    if str(out.dtype) != dtype_str:
        out = out.cast(dtype_str)
    if was_cuda and getattr(jt.flags, "use_cuda", 0):
        out = out.cuda()
    return out


class GlooCommunicator:
    """Full-mesh TCP communicator for one WORLD or subgroup process group."""

    def __init__(self, store, rank, world_size, prefix="jittor/gloo/world",
                 timeout=300.0):
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.prefix = str(prefix)
        self.timeout = float(timeout)
        self._closed = False
        self._lock = threading.Lock()
        self._peers = {}
        self._listen_sock = None

        if self.world_size <= 1:
            return
        if store is None:
            raise RuntimeError("Gloo communicator requires a Store for rendezvous")

        adv_host, bind_host = _resolve_advertise_host()
        listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_sock.bind((bind_host, 0))
        listen_sock.listen(max(self.world_size * 2, 64))
        listen_sock.settimeout(self.timeout)
        listen_port = listen_sock.getsockname()[1]
        self._listen_sock = listen_sock

        try:
            store.set(
                "{}/addr/{}".format(self.prefix, self.rank),
                "{}:{}".format(adv_host, listen_port).encode("utf-8"),
            )
            keys = ["{}/addr/{}".format(self.prefix, r) for r in range(self.world_size)]
            store.wait(keys)
            peer_addrs = {}
            for r, key in enumerate(keys):
                raw = store.get(key).decode("utf-8")
                host, port_text = raw.rsplit(":", 1)
                peer_addrs[r] = (host, int(port_text))

            # Deadlock-free ordering: connect to every rank < self.rank, then
            # accept from every rank > self.rank.
            for peer in range(self.rank):
                host, port = peer_addrs[peer]
                sock = socket.create_connection((host, port), timeout=self.timeout)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(self.timeout)
                _send_exact(sock, _RANK_STRUCT.pack(self.rank))
                self._peers[peer] = sock

            expected_incoming = self.world_size - 1 - self.rank
            for _ in range(expected_incoming):
                conn, _addr = listen_sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.settimeout(self.timeout)
                (peer_rank,) = _RANK_STRUCT.unpack(
                    _recv_exact(conn, _RANK_STRUCT.size)
                )
                self._peers[int(peer_rank)] = conn
        except BaseException:
            self.close()
            raise
        finally:
            if self._listen_sock is not None:
                try:
                    self._listen_sock.close()
                except OSError:
                    pass
                self._listen_sock = None

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._listen_sock is not None:
            try:
                self._listen_sock.close()
            except OSError:
                pass
            self._listen_sock = None
        for sock in list(self._peers.values()):
            try:
                sock.close()
            except OSError:
                pass
        self._peers.clear()

    def _require_open(self):
        if self._closed:
            raise RuntimeError("Gloo communicator has been closed")

    def barrier(self):
        if self.world_size <= 1:
            return
        self._require_open()
        with self._lock:
            if self.rank == 0:
                for peer in range(1, self.world_size):
                    _recv_frame(self._peers[peer], expected_tag=_COLLECTIVE_TAG, expected_nbytes=1)
                for peer in range(1, self.world_size):
                    _send_frame(self._peers[peer], _COLLECTIVE_TAG, b"\x01")
            else:
                _send_frame(self._peers[0], _COLLECTIVE_TAG, b"\x01")
                _recv_frame(self._peers[0], expected_tag=_COLLECTIVE_TAG, expected_nbytes=1)

    def _broadcast_bytes_locked(self, buf, root=0):
        # Binomial tree broadcast rooted at `root` (relabeled so root == 0).
        n = self.world_size
        rel = (self.rank - root) % n
        nbytes = len(buf)
        step = 1
        while step < n:
            step <<= 1
        step >>= 1
        while step >= 1:
            if rel % (2 * step) == 0:
                dst_rel = rel + step
                if dst_rel < n:
                    dst = (dst_rel + root) % n
                    _send_frame(self._peers[dst], _COLLECTIVE_TAG, buf)
            elif rel % (2 * step) == step:
                src_rel = rel - step
                src = (src_rel + root) % n
                buf = _recv_frame(self._peers[src], expected_tag=_COLLECTIVE_TAG, expected_nbytes=nbytes)
            step >>= 1
        return buf

    def broadcast_array(self, arr, root=0):
        if self.world_size <= 1:
            return arr
        self._require_open()
        arr = np.ascontiguousarray(arr)
        raw = memoryview(arr).cast("B")
        with self._lock:
            out_buf = self._broadcast_bytes_locked(
                bytearray(raw) if self.rank == root else bytearray(len(raw)),
                root=int(root),
            )
        return np.frombuffer(out_buf, dtype=arr.dtype).reshape(arr.shape).copy()

    def all_reduce_array(self, arr, operation="sum"):
        if self.world_size <= 1:
            return arr
        self._require_open()
        buf = np.ascontiguousarray(arr).copy()
        nbytes = buf.nbytes
        n = self.world_size
        with self._lock:
            # Binomial tree reduce onto rank 0.
            step = 1
            while step < n:
                if self.rank % (2 * step) == 0:
                    src = self.rank + step
                    if src < n:
                        raw = _recv_frame(
                            self._peers[src],
                            expected_tag=_COLLECTIVE_TAG,
                            expected_nbytes=nbytes,
                        )
                        peer_arr = np.frombuffer(raw, dtype=buf.dtype).reshape(buf.shape)
                        buf = _reduce_numpy(buf, peer_arr, operation)
                elif self.rank % step == 0:
                    dst = self.rank - step
                    _send_frame(self._peers[dst], _COLLECTIVE_TAG, memoryview(buf).cast("B"))
                    break
                step <<= 1

            if self.rank == 0 and operation == "mean":
                if np.issubdtype(buf.dtype, np.floating):
                    buf /= n
                else:
                    buf //= n

            out_raw = self._broadcast_bytes_locked(
                bytearray(memoryview(buf).cast("B")) if self.rank == 0 else bytearray(nbytes),
                root=0,
            )
        return np.frombuffer(out_raw, dtype=buf.dtype).reshape(buf.shape).copy()

    def all_gather_array(self, arr):
        if self.world_size <= 1:
            return arr
        self._require_open()
        local = np.ascontiguousarray(arr)
        nbytes = local.nbytes
        full_shape = (local.shape[0] * self.world_size,) + local.shape[1:]
        total_bytes = nbytes * self.world_size
        with self._lock:
            if self.rank == 0:
                gathered = bytearray(total_bytes)
                gathered[0:nbytes] = memoryview(local).cast("B")
                for peer in range(1, self.world_size):
                    chunk = _recv_frame(
                        self._peers[peer],
                        expected_tag=_COLLECTIVE_TAG,
                        expected_nbytes=nbytes,
                    )
                    gathered[peer * nbytes:(peer + 1) * nbytes] = chunk
            else:
                _send_frame(self._peers[0], _COLLECTIVE_TAG, memoryview(local).cast("B"))
                gathered = bytearray(total_bytes)
            out_raw = self._broadcast_bytes_locked(gathered, root=0)
        return np.frombuffer(out_raw, dtype=local.dtype).reshape(full_shape).copy()

    def reduce_scatter_array(self, arr):
        if self.world_size <= 1:
            return arr
        reduced = self.all_reduce_array(arr, "sum")
        chunk = reduced.shape[0] // self.world_size
        start = self.rank * chunk
        return np.ascontiguousarray(reduced[start:start + chunk])

    def send_array(self, arr, peer_local_rank, tag=0):
        self._require_open()
        sock = self._peers[int(peer_local_rank)]
        local = np.ascontiguousarray(arr)
        _send_frame(sock, int(tag), memoryview(local).cast("B"))

    def recv_array(self, shape, np_dtype, peer_local_rank, tag=0):
        self._require_open()
        sock = self._peers[int(peer_local_rank)]
        dtype = np.dtype(np_dtype)
        expected = int(np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize
        raw = _recv_frame(sock, expected_tag=int(tag), expected_nbytes=expected)
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()

    # -- Jittor Var helpers (raw, without autograd wrapper) ----------------

    def all_reduce_var(self, tensor, operation="sum"):
        arr, _shape, dtype_str, was_cuda = _var_to_host_numpy(tensor, upcast_low_fp=True)
        reduced = self.all_reduce_array(arr, operation)
        return _host_numpy_to_var(reduced, dtype_str, was_cuda)

    def broadcast_var(self, tensor, root=0):
        arr, _shape, dtype_str, was_cuda = _var_to_host_numpy(tensor, upcast_low_fp=False)
        out = self.broadcast_array(arr, root=root)
        return _host_numpy_to_var(out, dtype_str, was_cuda)

    def all_gather_var(self, tensor):
        arr, _shape, dtype_str, was_cuda = _var_to_host_numpy(tensor, upcast_low_fp=False)
        out = self.all_gather_array(arr)
        return _host_numpy_to_var(out, dtype_str, was_cuda)

    def reduce_scatter_var(self, tensor):
        arr, _shape, dtype_str, was_cuda = _var_to_host_numpy(tensor, upcast_low_fp=True)
        out = self.reduce_scatter_array(arr)
        return _host_numpy_to_var(out, dtype_str, was_cuda)

    def send_var(self, tensor, peer_local_rank, tag=0):
        arr, _shape, _dtype_str, _was_cuda = _var_to_host_numpy(tensor, upcast_low_fp=False)
        self.send_array(arr, peer_local_rank, tag=tag)

    def recv_var(self, tensor, peer_local_rank, tag=0):
        arr, shape, dtype_str, was_cuda = _var_to_host_numpy(tensor, upcast_low_fp=False)
        received = self.recv_array(shape, arr.dtype, peer_local_rank, tag=tag)
        return _host_numpy_to_var(received, dtype_str, was_cuda)


class _GlooAllReduceFn(jt.Function):
    def execute(self, x, comm, operation):
        self.comm = comm
        self.operation = operation
        return comm.all_reduce_var(x, operation)

    def grad(self, grad_y):
        if self.operation in ("sum", "mean"):
            return self.comm.all_reduce_var(grad_y, self.operation), None, None
        return None, None, None


class _GlooBroadcastFn(jt.Function):
    def execute(self, x, comm, root):
        self.comm = comm
        self.root = int(root)
        return comm.broadcast_var(x, root=self.root)

    def grad(self, grad_y):
        reduced = self.comm.all_reduce_var(grad_y, "sum")
        if self.comm.rank == self.root:
            return reduced, None, None
        return jt.zeros_like(grad_y), None, None


class _GlooAllGatherFn(jt.Function):
    def execute(self, x, comm):
        self.comm = comm
        return comm.all_gather_var(x)

    def grad(self, grad_y):
        return self.comm.reduce_scatter_var(grad_y), None


class _GlooReduceScatterFn(jt.Function):
    def execute(self, x, comm):
        self.comm = comm
        return comm.reduce_scatter_var(x)

    def grad(self, grad_y):
        return self.comm.all_gather_var(grad_y), None


def _needs_grad(tensor):
    return (
        not getattr(jt.flags, "no_grad", 0)
        and bool(getattr(tensor, "requires_grad", False))
    )


def gloo_all_reduce(tensor, comm, operation="mean"):
    if comm is None or comm.world_size <= 1:
        return tensor
    if _needs_grad(tensor) and operation in ("sum", "mean"):
        return _GlooAllReduceFn.apply(tensor, comm, operation)
    return comm.all_reduce_var(tensor, operation)


def gloo_broadcast(tensor, comm, root=0):
    if comm is None or comm.world_size <= 1:
        return tensor
    if _needs_grad(tensor):
        return _GlooBroadcastFn.apply(tensor, comm, int(root))
    return comm.broadcast_var(tensor, root=int(root))


def gloo_all_gather(tensor, comm):
    if comm is None or comm.world_size <= 1:
        return tensor
    if _needs_grad(tensor):
        return _GlooAllGatherFn.apply(tensor, comm)
    return comm.all_gather_var(tensor)


def gloo_reduce_scatter(tensor, comm):
    if comm is None or comm.world_size <= 1:
        return tensor
    if _needs_grad(tensor):
        return _GlooReduceScatterFn.apply(tensor, comm)
    return comm.reduce_scatter_var(tensor)
