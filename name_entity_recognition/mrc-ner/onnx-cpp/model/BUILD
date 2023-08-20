package(default_visibility = ["//visibility:public"])
load("//build:compiler.bzl", "CXXOPTS")
load("//build:buildflag_header.bzl", "buildflag_header")

cc_library(
    name = "mrc_ner",
    srcs = ["mrc_ner.cc",
           ],
    hdrs = [
            "mrc_ner.h",
          ],
    deps = [
            "//base:base",
	    "//third_party/gflags:gflags",
            "//third_party/onnxruntime:onnxruntime",
           ],
    copts = CXXOPTS + ["-fexceptions"],
    # linkstatic = True,
)
cc_test(
    name = "model_test",
    srcs = [
            "mrc_ner_test.cc",
	   ],
    deps = [
	    ":mrc_ner",
	    "//base:base",
	    "//third_party/gflags:gflags",
	   ],
)
