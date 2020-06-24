/*
    tests/autodiff_extraction.cpp -- tests the kernel extraction mechanism

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2019 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <vector>
#include <unordered_map>

#include "test.h"
#include <enoki/autodiff.h>
#include <enoki/color.h>
#include <enoki/cuda.h>
#include <enoki/dynamic.h>

using Float  = float;
using FloatC = CUDAArray<Float>;
using FloatD = DiffArray<FloatC>;
using UInt64C = replace_scalar_t<FloatC, uint64_t>;


ENOKI_TEST(test00_extract_forward_function) {
    FloatC a = 1.f, b = 2.f;
    set_label(a, "a");
    set_label(b, "b");

    cuda_start_recording_ptx_function("my_function");
    FloatC c = sqrt(sqr(a) + sqr(b));
    set_label(c, "c");

    cuda_set_inputs(a, b);
    cuda_set_outputs(c);
    cuda_stop_recording_ptx_function();

    const char *ptx_c = cuda_get_ptx_module();
    assert(ptx_c != nullptr);

    std::string ptx(ptx_c);
    std::string expected_start = ".version ";
    std::vector<std::string> expected_parts = {
        ".target sm_",
        ".func my_function",
        "// b",
        "  mul.rn.ftz.f32"
    };

    assert(ptx.rfind(expected_start, 0) == 0);
    for (const auto &e : expected_parts)
        assert(ptx.find(e) != std::string::npos);
}

ENOKI_TEST(test01_extract_autodiffed_function) {
    FloatD a = 1.f, b = 2.f;
    auto grad_of_output = zero<FloatC>(1);
    auto out_grad = zero<FloatC>(1);
    set_label(a, "a");
    set_label(b, "b");
    set_label(grad_of_output, "grad_of_output");
    set_label(out_grad, "out_grad");
    set_requires_gradient(a);

    cuda_start_recording_ptx_function("my_function_d");

    // Actual computation
    FloatD c = sqrt(sqr(a) + sqr(b));
    set_label(c, "c");
    // Propagate `grad_of_output` to the input through the computation.
    set_gradient(c, grad_of_output);
    backward<FloatD>(true);
    // Atomic accumulation to the output buffer
    auto grad_of_param = gradient(a);
    auto index = arange<UInt64C>(1);
    set_label(grad_of_param, "grad_of_param");
    set_label(index, "index");
    scatter_add(out_grad, grad_of_param, index);

    cuda_set_inputs(a, b, grad_of_output);
    // No function outputs that we care about,
    // gradients are accumulated to the "global" buffer
    // as a side effect.
    cuda_stop_recording_ptx_function();

    const char *ptx_c = cuda_get_ptx_module();
    assert(ptx_c != nullptr);

    std::string ptx(ptx_c);
    std::string expected_start = ".version ";
    std::vector<std::string> expected_parts = {
        ".target sm_",
        ".func my_function_d",
        "// b",
        "// grad_of_output",
        "  mul.rn.ftz.f32",
        "atom.global.add.f32"
    };

    assert(ptx.rfind(expected_start, 0) == 0);
    for (const auto &e : expected_parts)
        assert(ptx.find(e) != std::string::npos);
}


ENOKI_TEST(test02_extract_forward_function_with_globals) {
    cuda_create_ptx_module_context();

    FloatC a = 1.f, b = 2.f, g = 3.f;
    set_label(a, "my_a");
    set_label(b, "my_b");
    set_label(g, "my_g");

    cuda_start_recording_ptx_function("__direct_callable__callable");
    FloatC c = sqrt(sqr(a) + sqr(b)) - g;
    set_label(c, "c");

    // No function inputs, getting values from the globals
    cuda_set_outputs(c);
    cuda_stop_recording_ptx_function();

    const char *ptx_c = cuda_get_ptx_module();
    assert(ptx_c != nullptr);

    std::string ptx(ptx_c);
    // std::cout << ptx << std::endl;

    std::string expected_start = ".version ";
    std::vector<std::string> expected_parts = {
        ".target sm_",
        ".func __direct_callable__callable",
        "__globals_buf",
        "ld.global.f32",
        "mul.rn.ftz.f32"
    };

    assert(ptx.rfind(expected_start, 0) == 0);
    for (const auto &e : expected_parts)
        assert(ptx.find(e) != std::string::npos);

    // Check globals metadata
    char *names;
    size_t *name_sizes;
    size_t *offsets;
    size_t globals_count = cuda_get_ptx_globals(&names, &name_sizes, &offsets);

    // Convert back to std::unordered_map
    std::unordered_map<std::string, size_t> globals;
    size_t no = 0;
    for (size_t i = 0; i < globals_count; ++i) {
        std::string key = &names[no];
        globals[key] = offsets[i];
        no += name_sizes[i];
    }
    free(names);
    free(name_sizes);
    free(offsets);
    cuda_destroy_ptx_module_context();

    // printf("Extracted globals:\n");
    for (const auto &kv : globals) {
        // printf("- %s --> %lu\n", kv.first.c_str(), kv.second);
        assert(kv.first == "my_a" || kv.first == "my_b" || kv.first == "my_g");
        assert(kv.second == 0 || kv.second == 4 || kv.second == 8);
    }
}


ENOKI_TEST(test03_simple_function) {
    FloatC a = 1.f;
    set_label(a, "a");

    cuda_start_recording_ptx_function("my_jit_function");
    FloatC b = a + 2.f;
    set_label(b, "b");

    cuda_set_inputs(a);
    cuda_set_outputs(b);
    cuda_stop_recording_ptx_function();

    const char *ptx_c = cuda_get_ptx_module();
    assert(ptx_c != nullptr);

    std::string ptx(ptx_c);
    std::string expected_start = ".version ";
    std::vector<std::string> expected_parts = {
        ".target sm_",
        ".func my_jit_function",
        "__globals_buf",
        "  ld.global.f32",
        "  mul.rn.ftz.f32"
    };

    // std::cout << ptx << std::endl;

    assert(ptx.rfind(expected_start, 0) == 0);
    for (const auto &e : expected_parts)
        assert(ptx.find(e) != std::string::npos);
}
