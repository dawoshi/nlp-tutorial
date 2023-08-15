# load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
package(default_visibility = ["//visibility:public"])
load("//build:compiler.bzl", "CXXOPTS")
load("//build:buildflag_header.bzl", "buildflag_header")

cc_library(
    name = "mrc_bert_ner",
    srcs = ["mrc_bert_ner.cc",
           ],
    hdrs = [
            "mrc_bert_ner.h",
          ],
    deps = [
            "//base:base",
            "//third_party/onnxruntime:onnxruntime",
           ],
    copts = CXXOPTS + ["-fexceptions"],
    linkstatic = True,
)
cc_test(
    name = "model_test",
    srcs = [
            "mrc_bert_ner_test.cc",
	   ],
    deps = [
	    ":mrc_bert_ner",
	    "//base:base",
	   ],
)
