#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void normalize_kernel(float* input, float* output, float* mean, float* std, int batch_size, int channels, int height, int width) {
    int b = blockIdx.z;
    int c = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h = idx / width;
    int w = idx % width;
    if (b < batch_size && c < channels && h < height && w < width) {
        int offset = b * channels * height * width + c * height * width + h * width + w;
        output[offset] = (input[offset] - mean[c]) / std[c];
    }
}

torch::Tensor normalize_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor std) {
    TORCH_CHECK(input.is_cuda(), "input must be cuda tensor");
    TORCH_CHECK(mean.is_cuda(), "mean must be cuda tensor");
    TORCH_CHECK(std.is_cuda(), "std must be cuda tensor");
    auto output = torch::empty_like(input);
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    dim3 threads(256);
    dim3 blocks((height * width + threads.x - 1) / threads.x, channels, batch_size);
    normalize_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), mean.data_ptr<float>(), std.data_ptr<float>(), batch_size, channels, height, width);
    return output;
}

__global__ void clip_mask_kernel(long long* input, long long* output, long long min_val, long long max_val, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        long long val = input[idx];
        output[idx] = (val < min_val) ? min_val : (val > max_val) ? max_val : val;
    }
}

torch::Tensor clip_mask_cuda(torch::Tensor input, long long min_val, long long max_val) {
    TORCH_CHECK(input.is_cuda(), "input must be cuda tensor");
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    dim3 threads(256);
    dim3 blocks((num_elements + threads.x - 1) / threads.x);
    clip_mask_kernel<<<blocks, threads>>>(input.data_ptr<long long>(), output.data_ptr<long long>(), min_val, max_val, num_elements);
    return output;
}