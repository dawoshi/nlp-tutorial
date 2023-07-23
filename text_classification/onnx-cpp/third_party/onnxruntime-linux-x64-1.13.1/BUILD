load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "onnxruntime",
    srcs = glob([
        "lib/*.so*",
    ]),
    hdrs = glob([
        "include/**/*.h",
    ]),
    includes = ["include"],
    linkstatic = True,
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
