load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
cc_library(
    name = "bert_span_ner",
    srcs = ["bert_span_ner.cc",
           ],
    hdrs = [
            "bert_span_ner.h",
          ],
    deps = [
            "//base:base",
	    "//third_party/onnxruntime:onnxruntime",
           ],
    copts = [
        "-std=c++14"
    ]
)
cc_test(
    name = "model_test",
    srcs = [
            "bert_span_ner_test.cc",
	   ],
    deps = [
	    ":bert_span_ner",
	   ],
)
