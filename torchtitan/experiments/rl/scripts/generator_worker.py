# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run a monarch worker loop on a generator node for cross-node RL.

The controller (``torchtitan.experiments.rl.train``) attaches to one such
worker per generator through ``RL_GENERATOR_WORKER_ADDRS`` and spawns the
vLLM generator procs on it. Start it inside the same container image and
environment as the trainer entrypoint::

    python -m torchtitan.experiments.rl.scripts.generator_worker \\
        --advertise tcp://10.0.1.24:26600

``--advertise`` is the address the trainer node dials; pick the IP of the
NIC you want trainer<->generator traffic (weight sync included) to use. See
docs/multi_node.md for why that choice matters.
"""

import argparse

from monarch._src.actor.bootstrap import run_worker_loop_forever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--advertise",
        required=True,
        help="address the trainer dials, e.g. tcp://<this-node-fast-nic-ip>:26600",
    )
    parser.add_argument(
        "--listen",
        default=None,
        help="local bind address (default: tcp://0.0.0.0:<advertised port>)",
    )
    parser.add_argument(
        "--ca",
        default="trust_all_connections",
        help="monarch certificate policy (default: trust_all_connections)",
    )
    args = parser.parse_args()
    listen = args.listen
    if listen is None:
        port = args.advertise.rsplit(":", 1)[-1]
        listen = f"tcp://0.0.0.0:{port}"
    # monarch's "<advertised>@<bind>" form: dial-back address first, bind second.
    run_worker_loop_forever(ca=args.ca, address=f"{args.advertise}@{listen}")


if __name__ == "__main__":
    main()
