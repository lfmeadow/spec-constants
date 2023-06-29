#include <sycl/sycl.hpp>
typedef sycl::queue gpuStream_t;
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <array>

#define CONST_N 64
struct spec_const_struct { double v[CONST_N]; };

// specializatioh constant handle
constexpr static sycl::specialization_id<spec_const_struct> spec_const;

// specialization constant value
#define spec_const_val kh.get_specialization_constant<spec_const>().v


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
    s += input[idx].v[i] * spec_const_val[i];
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
  std::vector<indata> in_val(N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < CONST_N; ++j)
      in_val[i].v[j] = i*j;

  std::vector<double> out_val(N);

  auto input_bundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
    queue.get_context());
  
  input_bundle.set_specialization_constant<spec_const>(initval);
  auto exec_bundle = sycl::build(input_bundle);

  queue.submit([&](sycl::handler &cgh) {
    cgh.use_kernel_bundle(exec_bundle);
    indata *in = in_val.data();
    double *out = out_val.data();
    cgh.parallel_for(
      N, [=](auto idx, sycl::kernel_handler kh) {
	kernel(idx, kh, in, out);
      });
  });

  queue.wait();
}
