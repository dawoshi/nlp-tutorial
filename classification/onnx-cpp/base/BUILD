package(default_visibility = ["//visibility:public"])

cc_library(
    name = "base",
    srcs = [
        "utf8proc.c",
        "tokenization.cc",
    ],
    hdrs = [
        "utf8proc_data.h",
        "utf8proc.h",
        "tokenization.h",
    ],
    deps = [
    ],
    linkstatic = True,
)
cc_test(
    name = "tokenization_test",
    srcs = ["tokenization_test.cc"],
    deps = [":base"],
)

