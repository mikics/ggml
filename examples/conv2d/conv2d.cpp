#include "ggml-backend.h"
#include "ggml-cuda.h"  // or ggml-metal.h
#include "ggml.h"

#include <cstdlib>
#include <cstring>
#include <vector>

int main() {
    // --- 1. Init backend (GPU) ---
    ggml_backend_t backend = ggml_backend_cuda_init(0);

    // --- 2. Create context (no_alloc = true — backend handles memory) ---
    struct ggml_init_params params = {
        .mem_size   = ggml_tensor_overhead() * 9 + ggml_graph_overhead(),
        .mem_buffer = NULL,
        .no_alloc   = true,  // critical for GPU path
    };
    struct ggml_context * ctx = ggml_init(params);

    // --- 3. Define tensors (shape only, no data yet) ---
    struct ggml_tensor * input  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 224, 224, 3, 1);
    struct ggml_tensor * kernel = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 3, 64);

    // --- 4. Build compute graph ---
    struct ggml_tensor * out = ggml_conv_2d(ctx, kernel, input, 1, 1, 1, 1, 1, 1);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    // --- 5. Allocate all tensors on GPU via backend allocator ---
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // --- 6. Upload data to GPU ---
    std::vector<float> input_data(224 * 224 * 3, 0.0f);
    std::vector<float> kernel_data(3 * 3 * 3 * 64, 0.0f);

    ggml_backend_tensor_set(input, input_data.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(kernel, kernel_data.data(), 0, ggml_nbytes(kernel));

    // --- 7. Compute on GPU ---
    ggml_backend_graph_compute(backend, graph);

    // --- 8. Read result back to CPU ---
    float * result = (float *) malloc(ggml_nbytes(out));

    ggml_backend_tensor_get(out, result, 0, ggml_nbytes(out));

    // --- 9. Cleanup ---
    free(result);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
}
