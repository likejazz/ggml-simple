#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

// #include "_ggml.h"
// #include "_ggml-alloc.h"
// #include "_ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

static void ggml_log_callback_default(ggml_log_level level, const char* text, void* user_data) {
    (void)level;
    (void)user_data;
    fputs(text, stderr);
    fflush(stderr);
}

// This is a simple model with two tensors a and b
struct simple_model {
    struct ggml_tensor* a{};
    struct ggml_tensor* b{};

    // the backend to perform the computation (CPU, CUDA, METAL)
    ggml_backend_t backend = nullptr;
    // the backend buffer to storage the tensors data of a and b
    ggml_backend_buffer_t buffer{};
    // the context to define the tensor information (dimensions, size, memory address)
    struct ggml_context* ctx{};
};

int main() {
    ggml_time_init();

    // initialize data of matrices to perform matrix multiplication
    float matrix_A[4 * 2] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6
    };

    /* Transpose([
        10, 9, 5,
        5, 9, 4
    ]) 2 rows, 3 cols */
    float matrix_B[3 * 2] = {
        10, 5,
        9, 9,
        5, 4
    };

    simple_model model;
    float* a = matrix_A;
    float* b = matrix_B;

    // initialize the backend
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    model.backend = ggml_backend_cuda_init(0); // init device 0
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif
#ifdef GGML_USE_METAL
    fprintf(stderr, "%s: using Metal backend\n", __func__);
    ggml_backend_metal_log_set_callback(ggml_log_callback_default, nullptr);
    model.backend = ggml_backend_metal_init();
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
    }
#endif

    // if there aren't GPU Backends fallback to CPU backend
    if (!model.backend) {
        model.backend = ggml_backend_cpu_init();
    }

    int num_tensors = 2;

    ggml_init_params params{
        /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 2, 4);
    model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 2, 3);

    // create a backend buffer (backend memory) and alloc the tensors from the context
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);

    // load data from cpu memory to backend buffer
    ggml_backend_tensor_set(model.a, a, 0, ggml_nbytes(model.a));
    ggml_backend_tensor_set(model.b, b, 0, ggml_nbytes(model.b));

    // calculate the temporaly memory required to compute
    const ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    // create the worst case graph for memory usage estimation
    static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };
    // create a temporally context to build the graph
    ggml_context* ctx0 = ggml_init(params0);
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    // result = a*b^T
    ggml_tensor* result = ggml_mul_mat(ctx0, model.a, model.b);
    // build operations nodes
    ggml_build_forward_expand(gf, result);
    // delete the temporally context used to build the graph
    ggml_free(ctx0);

    ggml_gallocr_reserve(allocr, gf);
    size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);
    fprintf(stderr, "%s: compute buffer size: %.4f KB\n", __func__, mem_size / 1024.0);

    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, 1);
    }
    ggml_backend_graph_compute(model.backend, gf);

    // create a array to print result
    std::vector<float> out_data(ggml_nelements(result));

    // bring the data from the backend memory
    ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));

    // expected result:
    // [ 60.00 110.00 54.00 29.00
    //  55.00 90.00 126.00 28.00
    //  50.00 54.00 42.00 64.00 ]
    printf("mul mat (%d x %d) (transposed result):\n[", static_cast<int>(result->ne[0]), static_cast<int>(result->ne[1]));
    for (int j = 0; j < result->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result->ne[0] /* cols */; i++) {
            printf(" %.2f", out_data[i * result->ne[1] + j]);
        }
    }
    printf(" ]\n");

    // release backend memory used for computation
    ggml_gallocr_free(allocr);
    // free memory
    ggml_free(model.ctx);
    // release backend memory and free backend
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);

    return 0;
}
