#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse

import demo_runner as dr
import numpy as np

parser = argparse.ArgumentParser("Running AB tests on visibility culling")
parser.add_argument("--scene", type=str, default=dr.default_sim_settings["test_scene"])
parser.add_argument(
    "--max_frames",
    type=int,
    default=2000,
    help="Max number of frames simulated."
    "Default or larger value is suggested for accurate results.",
)
parser.add_argument(
    "--resolution",
    type=int,
    nargs="+",
    default=[128, 256, 512],
    help="Resolution r for frame (r x r).",
)
parser.add_argument(
    "--num_procs",
    type=int,
    nargs="+",
    default=[1, 3, 5],
    help="Number of concurrent processes.",
)

# TODO: --benchmark_semantic_sensor
# enable this option when we extend the culling on semantic sensor
parser.add_argument("--seed", type=int, default=1)

args = parser.parse_args()

default_settings = dr.default_sim_settings.copy()
default_settings["scene"] = args.scene
default_settings["silent"] = True
default_settings["seed"] = args.seed

default_settings["save_png"] = False
default_settings["print_semantic_scene"] = False
default_settings["print_semantic_mask_stats"] = False
default_settings["compute_shortest_path"] = False
default_settings["compute_action_shortest_path"] = False

default_settings["max_frames"] = args.max_frames
default_settings["frustum_culling"] = not args.disable_frustum_culling


benchmark_items = {
    "rgb": {},
    "rgbd": {"depth_sensor": True},
    "depth_only": {"color_sensor": False, "depth_sensor": True},
}

resolutions = args.resolution
nprocs_tests = args.num_procs

performance_all = {}
for nprocs in nprocs_tests:
    default_settings["num_processes"] = nprocs
    performance = []
    for resolution in resolutions:
        default_settings["width"] = default_settings["height"] = resolution
        perf = {}
        for key, value in benchmark_items.items():
            demo_runner = dr.DemoRunner(default_settings, dr.DemoRunnerType.BENCHMARK)
            print(" ---------------------- %s ------------------------ " % key)
            settings = default_settings.copy()
            settings.update(value)
            perf[key] = demo_runner.benchmark(settings)
            print(
                " ====== FPS (%d x %d, %s): %0.1f ======"
                % (settings["width"], settings["height"], key, perf[key].get("fps"))
            )
        performance.append(perf)

    performance_all[nprocs] = performance

for nproc, performance in performance_all.items():
    print(
        " ================ Performance (FPS) NPROC={} ===================================".format(
            nproc
        )
    )
    title = "Resolution "
    for key, value in perf.items():
        title += "\t%-10s" % key
    print(title)
    for idx in range(len(performance)):
        row = "%d x %d" % (resolutions[idx], resolutions[idx])
        for key, value in performance[idx].items():
            row += "\t%-8.1f" % value.get("fps")
        print(row)
    print(
        " =============================================================================="
    )

    # also print the average time per simulation step (including object perturbations)
    if args.enable_physics:
        print(
            " ================ Performance (step time: milliseconds) NPROC={} ===================================".format(
                nproc
            )
        )
        title = "Resolution "
        for key, value in perf.items():
            title += "\t%-10s" % key
        print(title)
        for idx in range(len(performance)):
            row = "%d x %d" % (resolutions[idx], resolutions[idx])
            for key, value in performance[idx].items():
                row += "\t%-8.2f" % (value.get("avg_sim_step_time") * 1000)
            print(row)
        print(
            " =============================================================================="
        )
