load("@rules_cc//cc:defs.bzl", "cc_library")
cc_library (
    name = "gflags",
    hdrs = glob(["src/*.h"]),
    srcs = glob(["src/*.cc"], exclude=["src/windows_*"]),
    includes = ["src"],
    linkstatic = True,
    strip_include_prefix = "src",
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)
