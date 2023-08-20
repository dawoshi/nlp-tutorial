package(default_visibility = ["//visibility:public"])
load("//build:compiler.bzl", "CXXOPTS")
load("//build:buildflag_header.bzl", "buildflag_header")

cc_library(
    name = "span_ner",
    srcs = ["span_ner.cc",
           ],
    hdrs = [
            "span_ner.h",
          ],
    deps = [
            "//base:base",
            "//third_party/gflags:gflags",
            "//third_party/onnxruntime:onnxruntime",
           ],
    copts = CXXOPTS + ["-fexceptions"],
    linkstatic = True,
)
cc_test(
    name = "model_test",
    srcs = [
            "span_ner_test.cc",
           ],
    deps = [
            ":span_ner",
            "//base:base",
            "//third_party/gflags:gflags",
           ],
)
