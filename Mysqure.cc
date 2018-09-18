#include <iostream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include <stdio.h>

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

//compute mul with cuda
void eltwise_mul(const float * A,const float * B, float * C, int n);


//squre op
//y = x*x
REGISTER_OP("Mysqure")
      .Input("input_data: float32")
      .Output("output_data: float32")
      .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
      });


template <typename Device, typename T>
class Mysqure : public OpKernel {
 public:
  explicit Mysqure(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //LOG(INFO) <<"==============DEVICE_CPU=============="<<std::endl;
    // 获取输入 tensor
    FILE* f =fopen("./1.txt","w");
    fprintf(f,"==============DEVICE_CPU===================");
    fclose(f);
    
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();

    // 创建输出 tensor, context->allocate_output 用来分配输出内存？
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<T>();

    // 执行计算操作。
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = input(i) * input(i);
    }
  }
};     


REGISTER_KERNEL_BUILDER(Name("Mysqure").Device(DEVICE_CPU), Mysqure<CPUDevice,float>);


template <typename T>
class Mysqure<GPUDevice, T> : public OpKernel {
 public:
  explicit Mysqure(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //LOG(INFO) << "==============DEVICE_GPU==================="<<std::endl;
    //log
    FILE* f =fopen("./2.txt","w");
    fprintf(f,"==============DEVICE_GPU===================");
    fclose(f);


    // 获取输入 tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();

    // 创建输出 tensor, context->allocate_output 用来分配输出内存？
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<T>();

    // 执行计算操作。
    const int N = input.size();
    eltwise_mul(input.data(),input.data(),output_flat.data(),N);
  }
};     


REGISTER_KERNEL_BUILDER(Name("Mysqure").Device(DEVICE_GPU), Mysqure<GPUDevice,float>);





//grad squre op
REGISTER_OP("MysqureGrad")
      .Input("grad: float32")
      .Input("input_data: float32")
      .Output("output_data: float32")
      .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
      });



template <typename Device, typename T>
class MysqureGrad : public OpKernel {
 public:
  explicit MysqureGrad(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // 获取输入 tensor
    const Tensor& input_tensor1 = context->input(0);
    const Tensor& input_tensor2 = context->input(1);

    auto input1 = input_tensor1.flat<T>();
    auto input2 = input_tensor2.flat<T>();

    // 创建输出 tensor, context->allocate_output 用来分配输出内存？
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor1.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<T>();

    // 执行计算操作。
    const int N = input1.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = 2 * input1(i) * input2(i);
    }
  }
};     


REGISTER_KERNEL_BUILDER(Name("MysqureGrad").Device(DEVICE_CPU), MysqureGrad<CPUDevice,float>);