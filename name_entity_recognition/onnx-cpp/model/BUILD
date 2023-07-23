load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
cc_library(
    name = "bert_classification",
    srcs = ["bert_classification.cc",
           ],
    hdrs = [
            "common.h",
            "bert_classification.h",
          ],
    deps = [
            "//base:base",
	    "//third_party/onnxruntime:onnxruntime",
           ],
)
cc_test(
    name = "model_test",
    srcs = [
            "bert_classification_test.cc",
	   ],
    deps = [
	    ":bert_classification",
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
