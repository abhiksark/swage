# test/lit.cfg.py
# -*- Python -*-
"""Lit configuration for the Swage MLIR regression tests."""

import os

import lit.formats

from lit.llvm import llvm_config

config.name = "SWAGE"

# The internal shell keeps the suite independent of the host shell and of
# the execute_external option that lit 23 removed. Every RUN line here uses
# only pipes, `not`, and 2>&1, which the internal shell handles natively.
config.test_format = lit.formats.ShTest()

config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.swage_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%llvm_lib_dir", config.llvm_lib_dir))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

config.excludes = ["Inputs", "CMakeLists.txt", "README.md"]

config.swage_tools_dir = os.path.join(config.swage_obj_root, "bin")

llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.swage_tools_dir, config.llvm_tools_dir]
tools = ["swage-opt"]

llvm_config.add_tool_substitutions(tools, tool_dirs)
