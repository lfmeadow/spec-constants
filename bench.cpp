#include <sycl/sycl.hpp>
typedef sycl::queue gpuStream_t;
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <array>
#include <iostream>
#include <chrono>

#define CONST_N 64
struct spec_const_struct { double v[CONST_N]; };

// specializatioh constant handle
constexpr sycl::specialization_id<spec_const_struct> spec_const;

struct indata {
  double v[CONST_N];
};

void
kernel(
  sycl::id<1> idx,
  sycl::kernel_handler kh,
  const indata *input,
  double *output
  )
{
  double s = 0;
  for (int i = 0; i < CONST_N; ++i)
    s += input[idx].v[i] * kh.get_specialization_constant<spec_const>().v[i];
  output[idx] = s;
}

int
main()
{
  sycl::queue queue;
  spec_const_struct initval;

  for (int i = 0; i < CONST_N; ++i)
    initval.v[i] = i;

  // initialize input
  size_t N = 1024*1024;
  std::vector<indata> h_in(N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < CONST_N; ++j)
      h_in[i].v[j] = 1;
  indata *d_in = sycl::malloc_device<indata>(N, queue);
  queue.memcpy(d_in, h_in.data(), N*sizeof(indata));
  queue.wait();


  std::vector<double> h_out(N);
  double *d_out = sycl::malloc_device<double>(N, queue);

#if 0
  const auto start_clock = std::chrono::high_resolution_clock::now();
  auto input_bundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
    queue.get_context());
  
  input_bundle.set_specialization_constant<spec_const>(initval);
  auto exec_bundle = sycl::build(input_bundle);
#endif

  const auto kernel_clock = std::chrono::high_resolution_clock::now();

  for (int ntimes = 0; ntimes < 100; ++ntimes) {
    queue.submit([&](sycl::handler &cgh) {
      //cgh.use_kernel_bundle(exec_bundle);
      cgh.template set_specialization_constant<spec_const>(initval);
      cgh.parallel_for(
	N, [=](sycl::id<1> idx, sycl::kernel_handler kh) {
	  kernel(idx, kh, d_in, d_out);
	});
    });

    queue.wait();
  }
  const auto end_clock = std::chrono::high_resolution_clock::now();
  queue.memcpy(h_out.data(), d_out, N*sizeof(double));
  queue.wait();
  const std::chrono::duration<double> kernel_time = end_clock - kernel_clock;
  std::cout << " kernel execution: " << kernel_time.count() << "\n";
  std::cout << N << " elements " << CONST_N << " spec constants " <<
    "output[0] " << h_out[0] << " output[N/2] " << h_out[N/2] <<
    " output[N-1] " << h_out[N-1] << "\n";
}
