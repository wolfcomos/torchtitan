# Multi-node RL: trainer and generators on different hosts

`torchtitan.experiments.rl.train` runs the trainer and every generator on
`this_host()` by default. To put generators on other nodes, run a monarch
worker on each generator node and point the trainer at it. The controller
then spawns the vLLM procs on the remote host mesh and everything else --
rollout routing, weight sync, metrics -- works unchanged.

## Launch

On each generator node, inside the same container image and environment as
the trainer (same `PYTHONPATH`, same torchtitan/torchao checkouts):

```bash
python -m torchtitan.experiments.rl.scripts.generator_worker \
    --advertise tcp://<generator-node-fast-nic-ip>:26600
```

On the trainer node:

```bash
RL_GENERATOR_WORKER_ADDRS=tcp://<generator-node-fast-nic-ip>:26600 \
RL_GPUS_PER_NODE=4 \
python -m torchtitan.experiments.rl.train --module ... --config ... --num_generators 1
```

`RL_GENERATOR_WORKER_ADDRS` takes one comma-separated worker address per
generator (`--num_generators` must match). `RL_GPUS_PER_NODE` is how many
GPUs each node exposes to a role (default 8). The trainer always stays on the
launching host, so a 4+4 run on two 4-GPU nodes is trainer parallelism
`dp_shard * tp * ... = 4` plus generator `dp * tp = 4` with one worker.

The trainer switches monarch's client transport to TCP before its first
monarch call (`enable_transport("tcp")`): the default unix-domain reply
address is not dialable from another host.

## Put the traffic on the right NIC

Every byte between the two roles -- monarch RPC, rollout requests and
completions, and the TorchStore weight sync -- travels over whatever
network path the two hosts pick for each other. Three things decide it:

1. **The worker's `--advertise` address.** Use the IP of the fast NIC, not
   the hostname.
2. **What the node's own hostname resolves to inside the container.**
   TorchStore's gloo transport opens its `TCPStore` on
   `socket.getfqdn()`, and the trainer advertises its monarch reply
   address by hostname. Cluster DNS often maps a compute node's name to its
   1 GbE management interface, and docker's generated `/etc/hosts` maps it
   to `127.0.1.1`. Either one is wrong here: mount an `/etc/hosts` into
   both containers that maps every node's FQDN to its fast-NIC address.
3. **`GLOO_SOCKET_IFNAME`.** Pin gloo's sockets to the fast interface on
   both sides (e.g. `GLOO_SOCKET_IFNAME=vlan1202`). Without it gloo also
   falls back to the hostname's interface.

Check the result on the first weight sync. Each generator rank logs

```
[rank 0] weight pull: 15.31 GiB in 4.1s (3.73 GiB/s)
```

and emits `weight_sync/pull_gib`, `weight_sync/pull_seconds`,
`weight_sync/pull_gib_per_s` structured scalars. Compare the GiB/s against
your fabric: a number near 0.1 GiB/s means the pull is riding a 1 GbE link.

### Why this matters (measured)

Qwen3-30B-A3B, GRPO, trainer 4x GB200 on one node + vLLM generators 4x GB200
on a second node, bf16 weights (~61 GiB pulled per sync across the 4
generator ranks):

| path                                  | pull per step | share of 1346 s step |
|---------------------------------------|---------------|----------------------|
| hostname -> 1 GbE management NIC      | 547 s         | 41%                  |
| fast NIC (100 GbE, gloo over TCP)     | see PR        |                      |

`iperf` between the same two hosts: 0.94 Gb/s on the management NIC,
37 Gb/s per stream / 94 Gb/s with 4 streams on the 100 GbE NIC. The
weight pull was simply saturating the wrong link; nothing in the RL loop
was slow.

## Transport notes

- TorchStore picks the transport per storage volume: shared memory when the
  client is on the trainer node (the trainer's own push), otherwise
  torchcomms, monarch RDMA, gloo, or plain monarch RPC in that order of
  availability. Cross-node with RDMA unavailable or disabled it is gloo.
- `TORCHSTORE_RDMA_ENABLED=0` disables monarch ibverbs RDMA. Set it when
  the node exposes mlx5 devices that RDMA cannot actually use (e.g. no
  usable RoCE GIDs on the reachable Ethernet port); otherwise monarch tries
  ibverbs and fails at queue-pair setup instead of falling back.
- The trainer stages weights on CPU (`direct_rdma=False`), so the pull
  reads a snapshot and the trainer's GPU weights are free to move on.

## Container gotchas seen on GB200 nodes

- Run the container `--privileged` if GPU access disappears minutes after
  start (device cgroups revoked by slurm/systemd scope churn).
- Re-run the CUDA preflight inside the actual run container, not a
  throwaway one: the run container can come up without CUDA while a probe
  container passes.
