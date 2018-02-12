#include "caffe/util/math_functions.hpp"
namespace caffe {

template <typename Dtype>
__global__ void ClampData(int N, Dtype* data, Dtype low, Dtype high) {
  CUDA_KERNEL_LOOP(i, N) {
    data[i] = (data[i] > high) ? (high) : (data[i]);
    data[i] = (data[i] < low) ? (low) : (data[i]);
  }
}

template <typename Dtype>
void Clamp_data_gpu(int N, Dtype* data, Dtype low, Dtype high) {
  ClampData<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, data, low, high);
  CUDA_POST_KERNEL_CHECK;
}
template void Clamp_data_gpu<float>(int, float*, float, float);
template void Clamp_data_gpu<double>(int, double*, double, double);

}

