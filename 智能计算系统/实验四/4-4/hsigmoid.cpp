// //Pytorch扩展头文件的引用
// #include <torch/extension.h> 
// using namespace std; 

// //hsigmoid_cpu函数的具体实现
// torch::Tensor hsigmoid_cpu(const torch::Tensor & dets) {
//   //TODO: 将输入的tensor转化为浮点类型的vector
//   ______________________________________
//   int input_size = input_data.size(); 
//   //TODO: 创建一个浮点类型的output_data，output_data为大小与输入相同的vector
//   ______________________________________
//   //TODO: 对于输入向量的每个元素计算hsigmoid
//   ______________________________________
//   //TODO: Create tensor options with dtype float32
//   auto opts = torch::TensorOptions().dtype(torch::kFloat32);
//   //TODO: Create a tensor from the output vector
//   auto foo= torch::from_blob(output_data.data(), {int64_t(output_data.size())}, opts).clone();
//   //TODO: 将得到的tensor reshape为所需的大小
//   ______________________________________
//   return output;
// } 
// //TODO: 算子绑定为Pytorch的模块
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {	
// ______________________________________
// }       

//Pytorch扩展头文件的引用
#include <torch/extension.h> 
#include <vector>
#include <cmath>
using namespace std; 

//hsigmoid_cpu函数的具体实现
torch::Tensor hsigmoid_cpu(const torch::Tensor & dets) {
  //TODO: 将输入的tensor转化为浮点类型的vector
  auto input_data = dets.accessor<float, 3>();
  int batch_size = dets.size(0);
  int height = dets.size(1);
  int width = dets.size(2);
  //TODO: 创建一个浮点类型的output_data，output_data为大小与输入相同的vector
  std::vector<float>  output_data(batch_size * height * width);
  //TODO: 对于输入向量的每个元素计算hsigmoid
  for(int i=0;i<batch_size;++i){
    for(int j=0;j<height;++j){
      for(int k=0;k<width;++k){
        output_data[i * height * width + j * width + k] = 1.0 / (1 + std::exp(-input_data[i][j][k]));
      }
    }
  }
  //TODO: Create tensor options with dtype float32
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  //TODO: Create a tensor from the output vector
  auto foo= torch::from_blob(output_data.data(), {int64_t(output_data.size())}, opts).clone();
  //TODO: 将得到的tensor reshape为所需的大小
  foo = foo.view_as(dets);
  return foo;
} 
//TODO: 算子绑定为Pytorch的模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {	
  m.def("hsigmoid_cpu", &hsigmoid_cpu, "HSigmoid function (CPU)");
}       
