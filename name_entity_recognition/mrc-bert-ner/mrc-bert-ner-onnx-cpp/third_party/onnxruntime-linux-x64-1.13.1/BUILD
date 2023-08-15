# load("@rules_cc//cc:defs.bzl", "cc_library")
package(default_visibility = ["//visibility:public"])
load("//build:compiler.bzl", "CXXOPTS")
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
    copts = CXXOPTS,
    # visibility = ["//visibility:public"],
)
