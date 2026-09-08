#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

#include "tools/onnxruntime_cuda.hpp"

int main(int argc, char * argv[])
{
  assert(argc == 3 || argc == 4);
  const int iterations = argc == 4 ? std::stoi(argv[3]) : 1;
  assert(iterations > 0);
  const cv::Mat input(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));

  const auto verify = [&input, iterations](
                        const char * model_path, const std::vector<std::int64_t> & expected_shape) {
    std::cerr << "creating " << model_path << '\n';
    inference::OnnxRuntimeCudaSession session(model_path);
    assert(session.uses_cuda());
    const auto warmup = session.run_bgr_nchw(input);
    assert(warmup.shape == expected_shape);
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
      const auto output = session.run_bgr_nchw(input);
      assert(output.shape == expected_shape);
      assert(output.values.size() == static_cast<std::size_t>(
        expected_shape.at(0) * expected_shape.at(1) * expected_shape.at(2)));
    }
    const double total_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start)
                              .count();
    std::cout << model_path << " mean_inference_ms=" << total_ms / iterations << '\n';
  };

  verify(argv[1], {1, 25200, 22});
  verify(argv[2], {1, 18, 8400});
  return 0;
}
