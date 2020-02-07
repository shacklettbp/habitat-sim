#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse

import demo_runner as dr
import numpy as np

parser = argparse.ArgumentParser("Running AB test on simulator")
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
parser.add_argument(
    "--semantic_sensor", action="store_true", help="Whether to enable semantic sensor."
)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--enable_physics",
    action="store_true",
    help="Whether to enable phyiscs (kinematic by default or dynamics if installed with bullet) during ab tests or not.",
)
parser.add_argument(
    "--num_objects",
    type=int,
    default=10,
    help="Number of objects to spawn if enable_physics is true.",
)
parser.add_argument(
    "--test_object_index",
    type=int,
    default=0,
    help="Index the objects to spawn if enable_physics is true. -1 indicates random.",
)
parser.add_argument(
    "--feature",
    type=str,
    required=True,
    help="the feature that is to be tested. (it must be defined as a boolean first in default_sim_settings",
)
args = parser.parse_args()

if not (args.feature in default_settings.keys()):
    raise RuntimeError("Feature to be tested is not defined in default_sim_settings.")

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

ab_tests_items = {
    "rgb": {},
    "rgbd": {"depth_sensor": True},
    "depth_only": {"color_sensor": False, "depth_sensor": True},
}
if args.semantic_sensor:
    ab_tests_items["semantic_only"] = {"color_sensor": False, "semantic_sensor": True}
    ab_tests_items["rgbd_semantic"] = {"depth_sensor": True, "semantic_sensor": True}

if args.enable_physics:
    # TODO: cannot test physics with no sensors as this won't create a renderer or load the scene.
    # ab_tests_items["enable_physics_no_obs"] = {"color_sensor": False, "enable_physics": True}
    ab_tests_items["phys_rgb"] = {"enable_physics": True}
    ab_tests_items["phys_rgbd"] = {"depth_sensor": True, "enable_physics": True}
    default_settings["num_objects"] = args.num_objects
    default_settings["test_object_index"] = args.test_object_index

resolutions = args.resolution
nprocs_tests = args.num_procs

performance_all = {}
for nprocs in nprocs_tests:
    default_settings["num_processes"] = nprocs
    performance = []
    for resolution in resolutions:
        default_settings["width"] = default_settings["height"] = resolution
        perf = {}
        for key, value in ab_tests_items.items():
            demo_runner = dr.DemoRunner(default_settings, dr.DemoRunnerType.AB_TESTS)
            print(" ---------------------- %s ------------------------ " % key)
            settings = default_settings.copy()
            settings.update(value)
            test_value = not settings[args.feature]
            perf[key] = demo_runner.ab_tests(settings, args.feature, test_value)
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
