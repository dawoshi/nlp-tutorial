package(default_visibility = ["//visibility:public"])
load("//build:compiler.bzl", "CXXOPTS")
load("//build:buildflag_header.bzl", "buildflag_header")

cc_library(
    name = "bert_classification",
    srcs = ["bert_classification.cc",
           ],
    hdrs = [
            "bert_classification.h",
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
            "bert_classification_test.cc",
	   ],
    deps = [
	    ":bert_classification",
	    "//base:base",
	    "//third_party/gflags:gflags",
	   ],
)
cc_test(
    name = "ort_pred",
    srcs = ["ort_pred.cc"],
    deps = [
	    "//third_party/onnxruntime:onnxruntime",
	    "//base:base",
	    ],
)
