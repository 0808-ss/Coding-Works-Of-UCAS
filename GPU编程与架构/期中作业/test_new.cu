// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc test.cu -o test -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp
// nvcc ./test.cu -o main -lhdf5 -lhdf5_cpp
// ./main ./weight

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <fstream>
#include <map>
#include <dirent.h>
#include <cstring>
#include <hdf5/serial/H5Cpp.h>
#include <cfloat>


float *d_feat_stn_conv1_weight;
float *d_feat_stn_conv1_bias;
float *d_feat_stn_bn1_weight;
float *d_feat_stn_bn1_bias;
float *d_feat_stn_bn1_running_mean;
float *d_feat_stn_bn1_running_var;
float *d_feat_stn_conv2_weight;
float *d_feat_stn_conv2_bias;
float *d_feat_stn_bn2_running_mean;
float *d_feat_stn_bn2_running_var;
float *d_feat_stn_bn2_weight;
float *d_feat_stn_bn2_bias;
float *d_feat_stn_conv3_weight;
float *d_feat_stn_conv3_bias;
float *d_feat_stn_bn3_running_mean;
float *d_feat_stn_bn3_running_var;
float *d_feat_stn_bn3_bias;
float *d_feat_stn_bn3_weight;
float *d_feat_stn_fc1_weight;
float *d_feat_stn_fc1_bias;
float *d_feat_stn_bn4_running_mean;
float *d_feat_stn_bn4_running_var;
float *d_feat_stn_bn4_bias;
float *d_feat_stn_bn4_weight;
float *d_feat_stn_fc2_weight;
float *d_feat_stn_fc2_bias;
float *d_feat_stn_bn5_running_mean;
float *d_feat_stn_bn5_running_var;
float *d_feat_stn_bn5_bias;
float *d_feat_stn_bn5_weight;
float *d_feat_stn_fc3_weight;
float *d_feat_stn_fc3_bias;
float *d_feat_conv1_weight;
float *d_feat_conv1_bias;
float *d_feat_bn1_running_mean;
float *d_feat_bn1_running_var;
float *d_feat_bn1_weight;
float *d_feat_bn1_bias;
float *d_fstn_conv1_weight;
float *d_fstn_conv1_bias;
float *d_feat_fstn_bn1_running_mean; 
float *d_feat_fstn_bn1_running_var;
float *d_feat_fstn_bn1_weight; 
float *d_feat_fstn_bn1_bias;
float *d_feat_fstn_conv2_weight;
float *d_feat_fstn_conv2_bias;
float *d_feat_fstn_bn2_running_mean;
float *d_feat_fstn_bn2_running_var;
float *d_feat_fstn_bn2_weight;
float *d_feat_fstn_bn2_bias;
float *d_feat_fstn_conv3_weight;
float *d_feat_fstn_conv3_bias;
float *d_feat_fstn_bn3_running_mean;
float *d_feat_fstn_bn3_running_var;
float *d_feat_fstn_bn3_weight;
float *d_feat_fstn_bn3_bias;
float *d_feat_fstn_fc1_weight;
float *d_feat_fstn_fc1_bias;
float *d_feat_fstn_bn4_weight;
float *d_feat_fstn_bn4_bias;
float *d_feat_fstn_bn4_mean;
float *d_feat_fstn_bn4_var;
float *d_feat_fstn_fc2_weight;
float *d_feat_fstn_fc2_bias;
float *d_feat_fstn_bn5_weight;
float *d_feat_fstn_bn5_bias;
float *d_feat_fstn_bn5_mean;
float *d_feat_fstn_bn5_var;
float *d_feat_fstn_fc3_weight;
float *d_feat_fstn_fc3_bias;
float *d_feat_conv2_weight;
float *d_feat_conv2_bias;
float *d_feat_bn2_running_mean;
float *d_feat_bn2_running_var;
float *d_feat_bn2_weight;
float *d_feat_bn2_bias;
float *d_feat_conv3_weight;
float *d_feat_conv3_bias;
float *d_feat_bn3_running_mean;
float *d_feat_bn3_running_var;
float *d_feat_bn3_weight;
float *d_feat_bn3_bias;
float *d_fc1_weight;
float *d_fc1_bias;
float *d_bn1_weight;
float *d_bn1_bias;
float *d_bn1_running_mean;
float *d_bn1_running_var;
float *d_fc2_weight;
float *d_fc2_bias;
float *d_bn2_weight;
float *d_bn2_bias;
float *d_bn2_running_mean;
float *d_bn2_running_var;
float *d_fc3_weight;
float *d_fc3_bias;

class GPU_pointer {
    public:
    GPU_pointer(int num) {
        cudaMalloc((void**)&data, (sizeof(float) * num));
        this->size = num;

    }
    ~GPU_pointer() {
        cudaFree(data);
    }   
    float* data;
    int size;

};

/****************************************************************************************
 * 读取模型参数
 ****************************************************************************************/
// 获取目录中的所有 .txt 文件
std::vector<std::string> get_files_in_directory(const std::string& dir) {
    std::vector<std::string> files;
    DIR* dp;
    struct dirent* entry;
    if ((dp = opendir(dir.c_str())) != NULL) {
        while ((entry = readdir(dp)) != NULL) {
            std::string filename = entry->d_name;
            if (filename.find(".txt") != std::string::npos) {
                files.push_back(filename);
            }
        }
        closedir(dp);
    } else {
        perror("opendir");
    }
    return files;
}

// 读取 .txt 文件并转换为 std::vector<float>
std::vector<float> read_param(const std::string& filepath) {
    std::vector<float> data;
    std::ifstream file(filepath);
    if (file.is_open()) {
        float value;
        while (file >> value) {
            data.push_back(value);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filepath << std::endl;
    }
    return data;
}

std::map<std::string, std::vector<float>> read_params(std::string dir) {
    // std::string dir = "."; // 当前目录
    std::map<std::string, std::vector<float>> params;

    // 获取目录中的所有 .txt 文件
    std::vector<std::string> param_files = get_files_in_directory(dir);
    for (const auto& file : param_files) {
        std::string filename = file.substr(0, file.find_last_of(".")); // 获取不带扩展名的文件名
        params[filename] = read_param(dir + "/" + file);
    }

    // // 访问参数时可以使用 params["conv1_weight"]
    // for (const auto& kv : params) {
    //     std::cout << "Key: " << kv.first << ", Values: ";
    //     // for (const auto& value : kv.second) {
    //     //     std::cout << value << " ";
    //     // }
    //     std::cout << std::endl;
    // }

    return params;
}

/****************************************************************************************
 * 读取训练集数据
 ****************************************************************************************/

using namespace H5;
void read_h5_file(const std::string& file_path, std::vector<std::vector<float>>& list_of_points, std::vector<int>& list_of_labels) {
    try {
        // 打开文件
        H5File file(file_path, H5F_ACC_RDONLY);

        // 获取文件中的所有数据集名称
        std::vector<std::string> dataset_names;
        hsize_t num_objs = file.getNumObjs();
        for (hsize_t i = 0; i < num_objs; i++) {
            dataset_names.push_back(file.getObjnameByIdx(i));
        }

        // 读取每个数据集
        for (const auto& name : dataset_names) {
            DataSet dataset = file.openDataSet(name + "/points");
            DataSpace dataspace = dataset.getSpace();

            // 获取数据集的维度
            hsize_t dims[2];
            dataspace.getSimpleExtentDims(dims, NULL);

            // 读取数据
            std::vector<float> points(dims[0] * dims[1]);
            dataset.read(points.data(), PredType::NATIVE_FLOAT);

            // 存储点云数据
            list_of_points.push_back(points);

            // 读取标签
            Attribute label_attr = file.openGroup(name).openAttribute("label");
            int label;
            label_attr.read(PredType::NATIVE_INT, &label);

            // 存储标签
            list_of_labels.push_back(label);
        }
    } catch (FileIException& error) {
        error.printErrorStack();
    } catch (DataSetIException& error) {
        error.printErrorStack();
    } catch (DataSpaceIException& error) {
        error.printErrorStack();
    } catch (DataTypeIException& error) {
        error.printErrorStack();
    }
}


// 函数参数（输出内存，矩阵a[m, k], 矩阵b[k, n], m , n ,k)
__global__ void matMulNaiveGpu(float* c, float* a, float* b, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

// 进行了一个乘法：A（m*k） B（k*n）
void matMulNaive(float* d_c, float * d_a, float * d_b, int m, int n, int k) {

    dim3 blockSize(16, 16);
    dim3 gridSize(((n + blockSize.x - 1) / blockSize.x), ((m + blockSize.y - 1) / blockSize.y));
    // 帮我补全gridsize
    matMulNaiveGpu<<<gridSize, blockSize>>>(d_c, d_a, d_b, m, n, k);

    cudaDeviceSynchronize();
}
// 这个问题是内存得自己管理？其实最好是用面向对象的思路

__global__ void transposeKernel(float *data, float *transposed, int row, int col) {
    // 计算当前线程的行和列索引
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;  // 列索引
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;  // 行索引

    if (xIndex < col && yIndex < row) {
        // 将数据从 (yIndex, xIndex) 位置拷贝到 (xIndex, yIndex) 位置
        transposed[xIndex * row + yIndex] = data[yIndex * col + xIndex];
    }
}

// 完成转置 参数（原矩阵， 目标空间， 行数， 列数）
void trans(float* input, float* output, int row, int col) {
    dim3 blockSize(16, 16);
    dim3 gridSize((col + blockSize.x - 1) / blockSize.x, (row + blockSize.y - 1) / blockSize.y);
    transposeKernel<<<gridSize, blockSize>>>(input, output, row, col);
    cudaDeviceSynchronize();
}
// 完成矩阵的转置

void Vector_Show(float* data, int Max) {

    float *c = (float*)malloc(Max * sizeof(float));
    
    cudaMemcpy((void*)c, (void*)data, Max * sizeof(float), cudaMemcpyDeviceToHost);

    // 注意得把数据迁移回来
    for(int i = 0; i < Max; i++) {
        std::cout<< c[i] << ' ';
    }

    std::cout<<std::endl;
    free(c);
}
void Vector_Show_col(float* data, int Max, int col, int max) {
    float *c = (float*)malloc(Max * sizeof(float));

    cudaMemcpy((void*)c, (void*)data, Max * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < max; i++) {
        std::cout<< c[i * col] << ' ';
    }
    std::cout<<std::endl;
    free(c);
}

void Vector_Show_row(float* data, int Max, int row) {

    float *c = (float*)malloc(Max * sizeof(float));
    
    cudaMemcpy((void*)c, (void*)data, Max * sizeof(float), cudaMemcpyDeviceToHost);

    // 注意得把数据迁移回来
    for(int i = 0; i < 100; i++) {
        std::cout<< c[row * 22500 + i] << ' ';

    }

    std::cout<<std::endl;
    free(c);
}

__global__ void Add_bias_kernel(float* data, int repeat, const float* bias, int dimen) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index % repeat;  // 对应原来第一个for循环
    int j = index / repeat;  // 对应原来第二个for循环

    if (j < dimen && i < repeat) {
        data[index] += bias[j];
    }
}

void Add_bias(float* data, int repeat, float* bias, int dimen) {


    // 启动kernel，每个线程处理一个index
    int totalThreads = repeat * dimen;
    int blockSize = 256;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;

    Add_bias_kernel<<<numBlocks, blockSize>>>(data, repeat, bias, dimen);
    cudaDeviceSynchronize();
}

// CUDA kernel for batch normalization
__global__ void BatchNormKernel(float* data, const float* mean, const float* variance, 
                                const float* weight, const float* bias, int num_features, 
                                int num_samples, float epsilon) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;  // Feature index
    int s = blockIdx.y * blockDim.y + threadIdx.y;  // Sample index
    
    if (c < num_features && s < num_samples) {
        // Calculate the index in the data array
        int index = c * num_samples + s;
        
        // Apply normalization
        float normalized_value = ((data[index] - mean[c]) / (sqrt(variance[c] + epsilon)));
        data[index] = weight[c] * normalized_value + bias[c];
    }
}

void BatchNorm(float* data, int data_size,
               float* mean, float* variance, 
               float* weight, float* bias, 
               int num_features, float epsilon = 1e-6) {
    
    int num_samples =  (data_size / num_features);

    // Define CUDA block and grid dimensions
    dim3 blockDim(16, 16);  // 16x16 threads per block
    dim3 gridDim((num_features + blockDim.x - 1) / blockDim.x, (num_samples + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    BatchNormKernel<<<gridDim, blockDim>>>(data, mean, variance, weight, bias, num_features, num_samples, epsilon);
    cudaDeviceSynchronize();
}

// CUDA kernel for ReLU operation
__global__ void ReluKernel(float* matrix, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (matrix[idx] < 0.0f) {
            matrix[idx] = 0.0f;
        }
    }
}

void Vector_show_cpu(std::vector<float> data) {
    std::cout << "num of item : " << data.size() << "  这是行优先元素  " <<std::endl;
    for(int i = 0; i < 100 && i < data.size(); i++) {
        std::cout << data[i] << ' ';
    }
    std::cout << std::endl;
}

// Host function for ReLU operation
void Relu(float* matrix, int data_size) {
    int size = data_size;

    // Define CUDA block and grid dimensions
    int blockSize = 256;  // Number of threads per block
    int gridSize = (size + blockSize - 1) / blockSize;  // Number of blocks

    // Launch kernel
    ReluKernel<<<gridSize, blockSize>>>(matrix, size);
    cudaDeviceSynchronize();
}


__global__ void Stn_Max_CUDA(const float* data, float* max_values, int row, int col) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < row) {
        float max_val = -999; 
        // 初始化为最小浮点数
        for (int i = 0; i < col; ++i) {
            if (data[row_idx * col + i] > max_val) {
                max_val = data[row_idx * col + i];
            }
        }
        max_values[row_idx] = max_val;
    }
}

// 宿主函数，用于在CPU上调用核函数
void Stn_Max_CUDA_Wrapper(float* data, float* recv, int row, int col) {

    // 定义线程块大小和网格大小
    int threadsPerBlock = 256;
    int blocksPerGrid = (row + threadsPerBlock - 1) / threadsPerBlock;

    // 调用核函数
    Stn_Max_CUDA<<<blocksPerGrid, threadsPerBlock>>>(data, recv, row, col);
    cudaDeviceSynchronize();
}

__global__ void Add_E_N_Kernel(float* d_data, int N) {
    int idx = threadIdx.x;

    // 只处理对角线上的元素
    if (idx < N) {
        d_data[idx * N + idx] += 1.0f;
    }
}

void Add_E_N(float* d_data, int N) {
    // 启动CUDA核函数，N <= 64 时可以直接使用一个 block，最多 64 个线程
    Add_E_N_Kernel<<<1, N>>>(d_data, N);
    cudaDeviceSynchronize();
}

__global__ void CopyKernel(float* input, float* output, int size) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 检查线程索引是否在数据范围内
    if (idx < size) {
        output[idx] = input[idx];
    }
}

// 宿主函数，用于在 CPU 上调用核函数
void Copy(float* input, float* output, int size) {
    // 定义线程块和线程网格的大小
    int blockSize = 256; // 每个线程块中的线程数
    int numBlocks = (size + blockSize - 1) / blockSize; // 网格中的块数

    // 启动核函数，直接在 GPU 内存中进行操作
    CopyKernel<<<numBlocks, blockSize>>>(input, output, size);

    // 等待 GPU 执行完毕
    cudaDeviceSynchronize();
}
__global__ void find_max_index(float* input, int *max_index) {

    float max_val = input[0];
    int max_idx = 0;

    // 循环遍历整个数组
    for (int i = 1; i < 10; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_idx = i;
        }
    }

    // 记录最大值的下标
    *max_index = max_idx;
}
int Select_Max(float* input) {
    // input是一個gpu數組，大小為10 ，你需要選出其中最大元素的下標
    int* d_max_index;
    int* h_max_index = (int*)malloc(sizeof(int));

    // 分配 GPU 内存
    cudaMalloc((void**)&d_max_index, sizeof(int));

    // 启动 CUDA 内核，使用 1 个 block 和 1 个线程
    find_max_index<<<1, 1>>>(input, d_max_index);

    // 将最大下标从设备复制回主机
    cudaMemcpy(h_max_index, d_max_index, sizeof(int), cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(d_max_index);

    return *(h_max_index);

}
int transpose(float* input) {
    // float * d_feat_stn_conv1_bia;
    // cudaMalloc((void**)&d_feat_stn_conv1_bias, sizeof(float) * 64);
    // cudaMemcpy(d_feat_stn_conv1_weight, params["feat.stn.conv1.weight"].data(), sizeof(float) * 64 * 3, cudaMemcpyHostToDevice);
    GPU_pointer input_T(67500);

    // input_T[3, 22500]
    trans(input, input_T.data, 22500, 3);
    //Vector_Show(input_T.data, 100);
    
    //res[64, 22500]
    GPU_pointer res(1440000);
    matMulNaive(res.data, d_feat_stn_conv1_weight, input_T.data, 64, 22500, 3);
    // Vector_Show(res.data, 100);
    // 和老版本相比没问题
    Add_bias(res.data, 22500, d_feat_stn_conv1_bias, 64);
    // Vector_Show(res.data, 100);
    BatchNorm(res.data, 1440000, d_feat_stn_bn1_running_mean, d_feat_stn_bn1_running_var, 
        d_feat_stn_bn1_weight, d_feat_stn_bn1_bias, 64);
    // Vector_Show(res.data, 100);
    // 和老版本相比没问题
    Relu(res.data, 1440000);

    // Vector_Show(res.data, 100);
    // 这里没有对照，如果出问题来书写num_zero函数

    // res_1[128, 22500]
    GPU_pointer res_1(2880000);
    matMulNaive(res_1.data, d_feat_stn_conv2_weight, res.data, 128, 22500, 64);
    Add_bias(res_1.data, 22500, d_feat_stn_conv2_bias, 128);
    BatchNorm(res_1.data, res_1.size, d_feat_stn_bn2_running_mean, d_feat_stn_bn2_running_var, 
        d_feat_stn_bn2_weight, d_feat_stn_bn2_bias, 128);
    // Vector_Show(res_1.data, 100);
    Relu(res_1.data, res_1.size);

    GPU_pointer res_2(1024 * 22500);
    matMulNaive(res_2.data, d_feat_stn_conv3_weight, res_1.data, 1024, 22500, 128);
    Add_bias(res_2.data, 22500, d_feat_stn_conv3_bias, 1024);
    BatchNorm(res_2.data, res_2.size, d_feat_stn_bn3_running_mean, d_feat_stn_bn3_running_var, 
        d_feat_stn_bn3_weight, d_feat_stn_bn3_bias, 1024);
    // Vector_Show(res_2.data, 100);
    // 与原结果一样
    Relu(res_2.data, res_2.size);

    GPU_pointer res_3(1024);
    Stn_Max_CUDA_Wrapper(res_2.data, res_3.data, 1024, 22500);
    // Vector_Show(res_3.data, 1024);
    // 与原结果相似

    // res_4[512, 1]
    GPU_pointer res_4(512);
    matMulNaive(res_4.data, d_feat_stn_fc1_weight, res_3.data, 512, 1, 1024);
    Add_bias(res_4.data, 1, d_feat_stn_fc1_bias, 512);
    BatchNorm(res_4.data, res_4.size, d_feat_stn_bn4_running_mean, d_feat_stn_bn4_running_var, 
        d_feat_stn_bn4_weight, d_feat_stn_bn4_bias, 512);
    Relu(res_4.data, res_4.size);

    GPU_pointer res_5(256);
    matMulNaive(res_5.data, d_feat_stn_fc2_weight, res_4.data, 256, 1, 512);
    Add_bias(res_5.data, 1, d_feat_stn_fc2_bias, 256);
    BatchNorm(res_5.data, res_5.size, d_feat_stn_bn5_running_mean, d_feat_stn_bn5_running_var, 
        d_feat_stn_bn5_weight, d_feat_stn_bn5_bias, 256);
    Relu(res_5.data, res_5.size);
    
    GPU_pointer res_6(9);
    matMulNaive(res_6.data, d_feat_stn_fc3_weight, res_5.data, 9, 1, 256);
    Add_bias(res_6.data, 1, d_feat_stn_fc3_bias, 9);
    // Vector_Show(res_6.data, 9);
    // 与原来的输出一样
    Add_E_N(res_6.data, 3);

    // Vector_Show(res_6.data, 9);


    // res_stn_T[22500, 3]
    GPU_pointer res_stn_T(67500);
    matMulNaive(res_stn_T.data, input, res_6.data, 22500, 3, 3);
    // Vector_Show(res_stn_T.data, 30000);
    
    // Vector_Show_col(res_stn_T.data, 67500, 22500, 3);
    // 数值相同并且沒有nan

    // res_stn[3, 22500]
    GPU_pointer res_stn(67500);
    trans(res_stn_T.data, res_stn.data, 22500, 3);
    // Vector_Show(res_stn.data, 30000);
    // 沒有nan

    GPU_pointer res_7(1440000);
    matMulNaive(res_7.data, d_feat_conv1_weight, res_stn.data, 64, 22500, 3);   
    // Vector_Show(res_7.data, 30000);    
    // 沒有nan
    // Vector_Show_col(res_7.data, 1440000, 22500, 60);
    
    Add_bias(res_7.data, 22500, d_feat_conv1_bias, 64);

    BatchNorm(res_7.data, res_7.size, d_feat_bn1_running_mean, d_feat_bn1_running_var, 
        d_feat_bn1_weight, d_feat_bn1_bias, 64);
    // Vector_Show(res_7.data, 30000);
    // 相差不大
    // Vector_Show_col(res_7.data, 64 * 22500, 64, 22500);
    // 沒問題
    // 沒有nan

    Relu(res_7.data, res_7.size);
    // Vector_Show_col(res_7.data, 64 * 22500, 64, 22500);
    // 沒有nan

    GPU_pointer res_stnkd(1440000);
    Copy(res_7.data, res_stnkd.data, res_7.size);
    // Vector_Show_col(res_stnkd.data, 64 * 22500, 64, 22500);
    
    GPU_pointer res_stnkd_1(1440000);
    matMulNaive(res_stnkd_1.data, d_fstn_conv1_weight, res_stnkd.data, 64, 22500, 64); 
    // Vector_Show_col(res_stnkd_1.data, 1440000, 22500, 60);
    // 沒問題

    Add_bias(res_stnkd_1.data, 22500, d_fstn_conv1_bias, 64);
    BatchNorm(res_stnkd_1.data, res_stnkd_1.size, d_feat_fstn_bn1_running_mean, d_feat_fstn_bn1_running_var, 
        d_feat_fstn_bn1_weight, d_feat_fstn_bn1_bias, 64);
    // Vector_Show(res_stnkd_1.data, 100);
    // Vector_Show_col(res_stnkd_1.data, 1440000, 22500, 60);
    // 相差不大
    Relu(res_stnkd_1.data, res_stnkd_1.size);
    // Vector_Show_col(res_stnkd_1.data, 1440000, 22500, 60);
    // 沒有nan, 沒問題


    GPU_pointer res_stnkd_2(128 * 22500);

    matMulNaive(res_stnkd_2.data, d_feat_fstn_conv2_weight, res_stnkd_1.data, 128, 22500, 64); 
    // Vector_Show(res_stnkd_2.data, 200);
    // Vector_Show_col(res_stnkd_2.data, 128 * 22500, 22500, 60);
    // 沒問題
    Add_bias(res_stnkd_2.data, 22500, d_feat_fstn_conv2_bias, 128);
    // Vector_Show_col(res_stnkd_2.data, 28880000, 22500, 60);
    // 沒問題
    // BatchNorm函數應該是沒有問題的，但是爲什麽結果不太對呢？

    // 輸入也沒問題
    BatchNorm(res_stnkd_2.data, res_stnkd_2.size, d_feat_fstn_bn2_running_mean, d_feat_fstn_bn2_running_var, 
        d_feat_fstn_bn2_weight, d_feat_fstn_bn2_bias, 128);

    // Vector_Show_col(res_stnkd_2.data, 128 * 22500, 22500, 60);
    
    // 結果似乎存在問題

    Relu(res_stnkd_2.data, res_stnkd_2.size);

    GPU_pointer res_stnkd_3(23040000);

    matMulNaive(res_stnkd_3.data, d_feat_fstn_conv3_weight, res_stnkd_2.data, 1024, 22500, 128);
    // Vector_Show(res_stnkd_3.data, 100);
    // Vector_Show(d_feat_fstn_conv3_weight, 100);

    Add_bias(res_stnkd_3.data, 22500, d_feat_fstn_conv3_bias, 1024);


    BatchNorm(res_stnkd_3.data, res_stnkd_3.size, d_feat_fstn_bn3_running_mean, d_feat_fstn_bn3_running_var, 
        d_feat_fstn_bn3_weight, d_feat_fstn_bn3_bias, 1024);
    // Vector_Show(d_feat_fstn_bn3_weight, 128);
    // Vector_Show_col(res_stnkd_3.data, 1024 * 22500, 1024, 22500);
    // Vector_Show(res_stnkd_3.data, 2024);
    // Vector_Show_row(res_stnkd_3.data, 1024 * 22500, 261);
    // 没有区别



    Relu(res_stnkd_3.data, res_stnkd_3.size);
    // Vector_Show(res_stnkd_3.data, 1024);


    GPU_pointer res_stnkd_4(1024);

    Stn_Max_CUDA_Wrapper(res_stnkd_3.data, res_stnkd_4.data, 1024, 22500);
    
    // Vector_Show(res_stnkd_4.data, 200);

    // 這個是絕對欸問題的 --- 20/14/12

    GPU_pointer res_stnkd_5(512);
    // Vector_Show(d_feat_fstn_fc1_weight, 100);
    matMulNaive(res_stnkd_5.data, d_feat_fstn_fc1_weight, res_stnkd_4.data, 512, 1, 1024); 
    Add_bias(res_stnkd_5.data, 1, d_feat_fstn_fc1_bias, 512);
    BatchNorm(res_stnkd_5.data, res_stnkd_5.size, d_feat_fstn_bn4_mean, d_feat_fstn_bn4_var, 
        d_feat_fstn_bn4_weight, d_feat_fstn_bn4_bias, 512);
    

    // 絕對沒問題
    
    Relu(res_stnkd_5.data, res_stnkd_5.size);

    GPU_pointer res_stnkd_6(256);
    matMulNaive(res_stnkd_6.data, d_feat_fstn_fc2_weight, res_stnkd_5.data, 256, 1, 512); 
    Add_bias(res_stnkd_6.data, 1, d_feat_fstn_fc2_bias, 256);
    BatchNorm(res_stnkd_6.data, res_stnkd_6.size, d_feat_fstn_bn5_mean, d_feat_fstn_bn5_var, 
        d_feat_fstn_bn5_weight, d_feat_fstn_bn5_bias, 256);

    // Vector_Show(res_stnkd_6.data, 200);
    // 絕對沒問題 ------ 10/14

    Relu(res_stnkd_6.data, res_stnkd_6.size);

    GPU_pointer res_stnkd_7(4096);
    matMulNaive(res_stnkd_7.data, d_feat_fstn_fc3_weight, res_stnkd_6.data, 4096, 1, 256); 
    
    // 絕對沒問題 --
    Add_bias(res_stnkd_7.data, 1, d_feat_fstn_fc3_bias, 4096);
    // Vector_Show_col(res_stnkd_7.data, res_stnkd_7.size, 64, 64);  
    // 絕對沒問題
    Add_E_N(res_stnkd_7.data, 64);
    // Vector_Show_col(res_stnkd_7.data, res_stnkd_7.size, 64, 64);
    // Vector_Show(res_stnkd_7.data, 64);
    // 這裏存在問題


    GPU_pointer stnkd_pre(1440000);
    trans(res_7.data, stnkd_pre.data, 64, 22500);
    // Vector_Show_col(stnkd_pre.data, 1440000, 64, 200);

    // 沒有問題,按列查看

    GPU_pointer res_8_pre(22500 * 64);
    matMulNaive(res_8_pre.data, stnkd_pre.data, res_stnkd_7.data, 22500, 64, 64); 
    // Vector_Show(res_8_pre.data, 100);
    // Vector_Show_col(res_8_pre.data, res_8_pre.size, 64, 64);
    // res_stnkd_7似乎不太一樣
    // 沒有問題了--10/14

    GPU_pointer res_8_pre_T(64 * 22500);
    trans(res_8_pre.data, res_8_pre_T.data, 22500, 64);

    GPU_pointer res_8(128 * 22500);
    matMulNaive(res_8.data, d_feat_conv2_weight, res_8_pre_T.data, 128, 22500, 64); 
    Add_bias(res_8.data, 22500 , d_feat_conv2_bias, 128);
    // Vector_Show(res_8.data, 100);
    // 行沒問題
    BatchNorm(res_8.data, res_8.size, d_feat_bn2_running_mean, d_feat_bn2_running_var, 
        d_feat_bn2_weight, d_feat_bn2_bias, 128);
    Relu(res_8.data, res_8.size);
    // Vector_Show_col(res_8.data, res_8.size, 22500, 64);
    // 列沒問題

    GPU_pointer res_9(1024 * 22500);
    matMulNaive(res_9.data, d_feat_conv3_weight, res_8.data, 1024, 22500, 128); 
    Add_bias(res_9.data, 22500 , d_feat_conv3_bias, 1024);
    BatchNorm(res_9.data, res_9.size, d_feat_bn3_running_mean, d_feat_bn3_running_var, 
        d_feat_bn3_weight, d_feat_bn3_bias, 1024);

    GPU_pointer res_10(1024);

    Stn_Max_CUDA_Wrapper(res_9.data, res_10.data, 1024, 22500);
    // Vector_Show(res_10.data, 100);
    
    GPU_pointer res_11(512);
    matMulNaive(res_11.data, d_fc1_weight, res_10.data, 512, 1, 1024); 
    Add_bias(res_11.data, 1 , d_fc1_bias, 512);
    BatchNorm(res_11.data, res_11.size, d_bn1_running_mean, d_bn1_running_var, 
        d_bn1_weight, d_bn1_bias, 512);
    // Vector_Show(res_11.data, 100);
    Relu(res_11.data, res_11.size);

    GPU_pointer res_12(256);
    matMulNaive(res_12.data, d_fc2_weight, res_11.data, 256, 1, 512); 
    Add_bias(res_12.data, 1 , d_fc2_bias, 256);
    BatchNorm(res_12.data, res_12.size, d_bn2_running_mean, d_bn2_running_var, 
        d_bn2_weight, d_bn2_bias, 256);
    Relu(res_12.data, res_12.size);

    GPU_pointer res_13(10);
    matMulNaive(res_13.data, d_fc3_weight, res_12.data, 10, 1, 256); 
    Add_bias(res_13.data, 1 , d_fc3_bias, 10);
    // Vector_Show(res_13.data, 10);
    // 沒問題
    return Select_Max(res_13.data);




}




int main(int argc, char *argv[]) {
    
    std::string dir = argv[1];  // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集点云数据和标签
    // std::string dir = "/home/course/LJY/project1/epoch30";
    // std::string dir = "/home/course/LJY/project1/model"; 
    // 读取模型参数
    auto params = read_params(dir);

    std::string file_path = "./data/test_point_clouds.h5";
    // std::string file_path = "/home/course/LJY/project1/3D_MNIST/test_point_clouds.h5";


    std::vector<std::vector<float>> list_of_points;//存储点坐标xyz
    std::vector<int> list_of_labels;
    // 读取训练集数据
    read_h5_file(file_path, list_of_points, list_of_labels);

    
    
    //stn3d
    //feat_stn_conv1_res = conv1d(points.T,"feat.stn.conv1.weight","feat.stn.conv1.bias",3,64) 
    cudaMalloc((void**)&d_feat_stn_conv1_weight, sizeof(float) * 64 * 3);
    cudaMalloc((void**)&d_feat_stn_conv1_bias, sizeof(float) * 64);
    cudaMemcpy(d_feat_stn_conv1_weight, params["feat.stn.conv1.weight"].data(), sizeof(float) * 64 * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_conv1_bias, params["feat.stn.conv1.bias"].data(), sizeof(float) * 64, cudaMemcpyHostToDevice);
    // feat_stn_bn1_res = bacthnorm1d(feat_stn_conv1_res,"feat.stn.bn1.weight","feat.stn.bn1.bias","feat.stn.bn1.running_mean","feat.stn.bn1.running_var",64)
    cudaMalloc((void**)&d_feat_stn_bn1_weight,  sizeof(float) * 64);
    cudaMalloc((void**)&d_feat_stn_bn1_bias, sizeof(float) * 64);
    cudaMalloc((void**)&d_feat_stn_bn1_running_mean,  sizeof(float) * 64);
    cudaMalloc((void**)&d_feat_stn_bn1_running_var, sizeof(float) * 64);
    cudaMemcpy(d_feat_stn_bn1_weight, params["feat.stn.bn1.weight"].data(), sizeof(float) * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_bn1_bias, params["feat.stn.bn1.bias"].data(), sizeof(float) * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_bn1_running_mean, params["feat.stn.bn1.running_mean"].data(), sizeof(float) * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_bn1_running_var, params["feat.stn.bn1.running_var"].data(), sizeof(float) * 64, cudaMemcpyHostToDevice);
    // feat_stn_conv2_res = conv1d(feat_stn_relu_bn1_res,"feat.stn.conv2.weight","feat.stn.conv2.bias",64,128)
    cudaMalloc((void**)&d_feat_stn_conv2_weight, sizeof(float) * 128 * 64);
    cudaMalloc((void**)&d_feat_stn_conv2_bias, sizeof(float) * 128);
    cudaMemcpy(d_feat_stn_conv2_weight, params["feat.stn.conv2.weight"].data(), sizeof(float) * 128 * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_conv2_bias, params["feat.stn.conv2.bias"].data(), sizeof(float) * 128, cudaMemcpyHostToDevice);
    //feat_stn_bn2_res = bacthnorm1d(feat_stn_conv2_res,"feat.stn.bn2.weight","feat.stn.bn2.bias","feat.stn.bn2.running_mean","feat.stn.bn2.running_var",128)
    cudaMalloc((void**)&d_feat_stn_bn2_running_mean, sizeof(float) * 128);
    cudaMalloc((void**)&d_feat_stn_bn2_running_var, sizeof(float) * 128);
    cudaMalloc((void**)&d_feat_stn_bn2_weight, sizeof(float) * 128);
    cudaMalloc((void**)&d_feat_stn_bn2_bias, sizeof(float) * 128);
    cudaMemcpy(d_feat_stn_bn2_running_mean, params["feat.stn.bn2.running_mean"].data(), sizeof(float) * 128, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_bn2_running_var, params["feat.stn.bn2.running_var"].data(), sizeof(float) * 128, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_bn2_weight, params["feat.stn.bn2.weight"].data(), sizeof(float) * 128, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_bn2_bias, params["feat.stn.bn2.bias"].data(), sizeof(float) * 128, cudaMemcpyHostToDevice);
    // feat_stn_conv3_res = conv1d(feat_stn_relu_bn2_res,"feat.stn.conv3.weight","feat.stn.conv3.bias",128,1024)
    cudaMalloc((void**)&d_feat_stn_conv3_weight, sizeof(float) * 1024 * 128);
    cudaMalloc((void**)&d_feat_stn_conv3_bias, sizeof(float) * 1024);
    cudaMemcpy(d_feat_stn_conv3_weight, params["feat.stn.conv3.weight"].data(), sizeof(float) * 1024 * 128, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_conv3_bias, params["feat.stn.conv3.bias"].data(), sizeof(float) * 1024, cudaMemcpyHostToDevice);
    // feat_stn_bn3_res = bacthnorm1d(feat_stn_conv3_res,"feat.stn.bn3.weight","feat.stn.bn3.bias","feat.stn.bn3.running_mean","feat.stn.bn3.running_var",1024)
    cudaMalloc((void**)&d_feat_stn_bn3_running_mean, sizeof(float) * 1024);
    cudaMalloc((void**)&d_feat_stn_bn3_running_var, sizeof(float) * 1024);
    cudaMalloc((void**)&d_feat_stn_bn3_weight, sizeof(float) * 1024);
    cudaMalloc((void**)&d_feat_stn_bn3_bias, sizeof(float) * 1024);
    cudaMemcpy(d_feat_stn_bn3_running_mean, params["feat.stn.bn3.running_mean"].data(), sizeof(float) * 1024, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_bn3_running_var, params["feat.stn.bn3.running_var"].data(), sizeof(float) * 1024, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_bn3_weight, params["feat.stn.bn3.weight"].data(), sizeof(float) * 1024, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_bn3_bias, params["feat.stn.bn3.bias"].data(), sizeof(float) * 1024, cudaMemcpyHostToDevice);
    //feat_stn_fc1_res = linear(feat_stn_max_res,"feat.stn.fc1.weight","feat.stn.fc1.bias",1024,512)
    cudaMalloc((void**)&d_feat_stn_fc1_weight,sizeof(float) * 512 * 1024); 
    cudaMalloc((void**)&d_feat_stn_fc1_bias, sizeof(float) * 512);
    cudaMemcpy((void**)d_feat_stn_fc1_bias, params["feat.stn.fc1.bias"].data(), sizeof(float) * 512, cudaMemcpyHostToDevice);
    cudaMemcpy((void**)d_feat_stn_fc1_weight, params["feat.stn.fc1.weight"].data(), sizeof(float) * 512 * 1024, cudaMemcpyHostToDevice);
    //feat_stn_bn4_res = bacthnorm1d(feat_stn_fc1_res.reshape(-1,1),"feat.stn.bn4.weight","feat.stn.bn4.bias","feat.stn.bn4.running_mean","feat.stn.bn4.running_var",512)
    cudaMalloc((void**)&d_feat_stn_bn4_running_mean, sizeof(float) * 512);
    cudaMalloc((void**)&d_feat_stn_bn4_running_var, sizeof(float) * 512);
    cudaMalloc((void**)&d_feat_stn_bn4_weight, sizeof(float) * 512);
    cudaMalloc((void**)&d_feat_stn_bn4_bias, sizeof(float) * 512);
    cudaMemcpy(d_feat_stn_bn4_running_mean, params["feat.stn.bn4.running_mean"].data(), sizeof(float) * 512, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_bn4_running_var, params["feat.stn.bn4.running_var"].data(), sizeof(float) * 512, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_bn4_weight, params["feat.stn.bn4.weight"].data(), sizeof(float) * 512, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_bn4_bias, params["feat.stn.bn4.bias"].data(), sizeof(float) * 512, cudaMemcpyHostToDevice);
    //feat_stn_fc2_res = linear(feat_stn_relu_bn4_res,"feat.stn.fc2.weight","feat.stn.fc2.bias",512,256)
    cudaMalloc((void**)&d_feat_stn_fc2_weight,sizeof(float) * 256 * 512); 
    cudaMalloc((void**)&d_feat_stn_fc2_bias, sizeof(float) * 256);
    cudaMemcpy(d_feat_stn_fc2_weight, params["feat.stn.fc2.weight"].data(), sizeof(float) * 256 * 512, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_fc2_bias, params["feat.stn.fc2.bias"].data(), sizeof(float) * 256, cudaMemcpyHostToDevice);
    //feat_stn_bn5_res = bacthnorm1d(feat_stn_fc2_res.reshape(-1,1),"feat.stn.bn5.weight","feat.stn.bn5.bias","feat.stn.bn5.running_mean","feat.stn.bn5.running_var",256)
    cudaMalloc((void**)&d_feat_stn_bn5_running_mean, sizeof(float) * 256);
    cudaMalloc((void**)&d_feat_stn_bn5_running_var, sizeof(float) * 256);
    cudaMalloc((void**)&d_feat_stn_bn5_weight, sizeof(float) * 256);
    cudaMalloc((void**)&d_feat_stn_bn5_bias, sizeof(float) * 256);
    cudaMemcpy(d_feat_stn_bn5_running_mean, params["feat.stn.bn5.running_mean"].data(), sizeof(float) * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_bn5_running_var, params["feat.stn.bn5.running_var"].data(), sizeof(float) * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_bn5_weight, params["feat.stn.bn5.weight"].data(), sizeof(float) * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_bn5_bias, params["feat.stn.bn5.bias"].data(), sizeof(float) * 256, cudaMemcpyHostToDevice);
    //feat_stn_fc3_res = linear(feat_stn_relu_bn5_res,"feat.stn.fc3.weight","feat.stn.fc3.bias",256,9)
    cudaMalloc((void**)&d_feat_stn_fc3_weight,sizeof(float) * 256 * 9); 
    cudaMalloc((void**)&d_feat_stn_fc3_bias, sizeof(float) * 256);
    cudaMemcpy(d_feat_stn_fc3_weight, params["feat.stn.fc3.weight"].data(), sizeof(float) * 256 * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_stn_fc3_bias, params["feat.stn.fc3.bias"].data(), sizeof(float) * 256, cudaMemcpyHostToDevice);
    //end stn3d

    //feat_conv1_res = conv1d(feat_bmm_res.T,"feat.conv1.weight","feat.conv1.bias",3,64)
    cudaMalloc((void**)&d_feat_conv1_weight, sizeof(float) * 64 * 3);
    cudaMalloc((void**)&d_feat_conv1_bias, sizeof(float) * 64);
    cudaMemcpy(d_feat_conv1_weight, params["feat.conv1.weight"].data(), sizeof(float) * 64 * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_conv1_bias, params["feat.conv1.bias"].data(), sizeof(float) * 64, cudaMemcpyHostToDevice);
    //feat_bn1_res = bacthnorm1d(feat_conv1_res,"feat.bn1.weight","feat.bn1.bias","feat.bn1.running_mean","feat.bn1.running_var",64)
    cudaMalloc((void**)&d_feat_bn1_running_mean, sizeof(float) * 64);
    cudaMalloc((void**)&d_feat_bn1_running_var, sizeof(float) * 64);
    cudaMalloc((void**)&d_feat_bn1_weight, sizeof(float) * 64);
    cudaMalloc((void**)&d_feat_bn1_bias, sizeof(float) * 64);
    cudaMemcpy(d_feat_bn1_running_mean, params["feat.bn1.running_mean"].data(), sizeof(float) * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_bn1_running_var, params["feat.bn1.running_var"].data(), sizeof(float) * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_bn1_weight, params["feat.bn1.weight"].data(), sizeof(float) * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_bn1_bias, params["feat.bn1.bias"].data(), sizeof(float) * 64, cudaMemcpyHostToDevice);
    
    
    // stnkd
    // feat_fstn_conv1_res = conv1d(feat_relu_bn1_res,"feat.fstn.conv1.weight","feat.fstn.conv1.bias",64,64)
    cudaMalloc((void**)&d_fstn_conv1_weight, sizeof(float) * 64 * 64);
    cudaMalloc((void**)&d_fstn_conv1_bias, sizeof(float) * 64);
    cudaMemcpy(d_fstn_conv1_weight, params["feat.fstn.conv1.weight"].data(), sizeof(float) * 64 * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fstn_conv1_bias, params["feat.fstn.conv1.bias"].data(), sizeof(float) * 64, cudaMemcpyHostToDevice);
    // feat_fstn_bn1_res = bacthnorm1d(feat_fstn_conv1_res,"feat.fstn.bn1.weight","feat.fstn.bn1.bias","feat.fstn.bn1.running_mean","feat.fstn.bn1.running_var",64)
    cudaMalloc((void**)&d_feat_fstn_bn1_running_mean, sizeof(float) * 64);
    cudaMalloc((void**)&d_feat_fstn_bn1_running_var, sizeof(float) * 64);
    cudaMalloc((void**)&d_feat_fstn_bn1_weight, sizeof(float) * 64);
    cudaMalloc((void**)&d_feat_fstn_bn1_bias, sizeof(float) * 64);
    cudaMemcpy(d_feat_fstn_bn1_running_mean, params["feat.fstn.bn1.running_mean"].data(), sizeof(float) * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_fstn_bn1_running_var, params["feat.fstn.bn1.running_var"].data(), sizeof(float) * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_fstn_bn1_weight, params["feat.fstn.bn1.weight"].data(), sizeof(float) * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_fstn_bn1_bias, params["feat.fstn.bn1.bias"].data(), sizeof(float) * 64, cudaMemcpyHostToDevice);
    // feat_fstn_conv2_res = conv1d(feat_fstn_relu_bn1_res,"feat.fstn.conv2.weight","feat.fstn.conv2.bias",64,128)
    cudaMalloc((void**)&d_feat_fstn_conv2_weight, sizeof(float) * 64 * 128);
    cudaMalloc((void**)&d_feat_fstn_conv2_bias, sizeof(float) * 128);
    cudaMemcpy(d_feat_fstn_conv2_weight, params["feat.fstn.conv2.weight"].data(), sizeof(float) * 64 * 128, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_fstn_conv2_bias, params["feat.fstn.conv2.bias"].data(), sizeof(float) * 128, cudaMemcpyHostToDevice);
    // feat_fstn_bn2_res = bacthnorm1d(feat_fstn_conv2_res,"feat.fstn.bn2.weight","feat.fstn.bn2.bias","feat.fstn.bn2.running_mean","feat.fstn.bn2.running_var",128)




    cudaMalloc((void**)&d_feat_fstn_bn2_running_mean, sizeof(float) * 128);
    cudaMalloc((void**)&d_feat_fstn_bn2_running_var, sizeof(float) * 128);
    cudaMalloc((void**)&d_feat_fstn_bn2_weight, sizeof(float) * 128);
    cudaMalloc((void**)&d_feat_fstn_bn2_bias, sizeof(float) * 128);
    // 似乎有問題 params["feat.fstn.bn2.weight"]
    cudaMemcpy(d_feat_fstn_bn2_running_mean, params["feat.fstn.bn2.running_mean"].data(), sizeof(float) * 128, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_fstn_bn2_running_var, params["feat.fstn.bn2.running_var"].data(), sizeof(float) * 128, cudaMemcpyHostToDevice);
    
    // Vector_show_cpu(params["feat.fstn.bn2.weight"]);
    // 是正確的數據
    
    
    
    cudaMemcpy(d_feat_fstn_bn2_weight, params["feat.fstn.bn2.weight"].data(), sizeof(float) * 128, cudaMemcpyHostToDevice);
    // Vector_Show(d_feat_fstn_bn2_weight, 128);
    // 是正確的數據

    cudaMemcpy(d_feat_fstn_bn2_bias, params["feat.fstn.bn2.bias"].data(), sizeof(float) * 128, cudaMemcpyHostToDevice);

    // feat_fstn_conv3_res = conv1d(feat_fstn_relu_bn2_res,"feat.fstn.conv3.weight","feat.fstn.conv3.bias",128,1024)
    cudaMalloc((void**)&d_feat_fstn_conv3_weight, sizeof(float) * 128 * 1024);
    cudaMalloc((void**)&d_feat_fstn_conv3_bias, sizeof(float) * 1024);
    cudaMemcpy(d_feat_fstn_conv3_weight, params["feat.fstn.conv3.weight"].data(), sizeof(float) * 128 * 1024, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_fstn_conv3_bias, params["feat.fstn.conv3.bias"].data(), sizeof(float) * 1024, cudaMemcpyHostToDevice);
    // feat_fstn_bn3_res = bacthnorm1d(feat_fstn_conv3_res,"feat.fstn.bn3.weight","feat.fstn.bn3.bias","feat.fstn.bn3.running_mean","feat.fstn.bn3.running_var",1024)
    cudaMalloc((void**)&d_feat_fstn_bn3_running_mean, sizeof(float) * 1024);
    cudaMalloc((void**)&d_feat_fstn_bn3_running_var, sizeof(float) * 1024);
    cudaMalloc((void**)&d_feat_fstn_bn3_weight, sizeof(float) * 1024);
    cudaMalloc((void**)&d_feat_fstn_bn3_bias, sizeof(float) * 1024);
    cudaMemcpy(d_feat_fstn_bn3_running_mean, params["feat.fstn.bn3.running_mean"].data(), sizeof(float) * 1024, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_fstn_bn3_running_var, params["feat.fstn.bn3.running_var"].data(), sizeof(float) * 1024, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_fstn_bn3_weight, params["feat.fstn.bn3.weight"].data(), sizeof(float) * 1024, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_fstn_bn3_bias, params["feat.fstn.bn3.bias"].data(), sizeof(float) * 1024, cudaMemcpyHostToDevice);
    // feat_fstn_fc1_res = linear(feat_fstn_max_res,"feat.fstn.fc1.weight","feat.fstn.fc1.bias",1024,512)
    cudaMalloc((void**)&d_feat_fstn_fc1_weight, sizeof(float) * 1024 * 512);
    cudaMalloc((void**)&d_feat_fstn_fc1_bias, sizeof(float) * 512);
    cudaMemcpy(d_feat_fstn_fc1_weight, params["feat.fstn.fc1.weight"].data(), sizeof(float) * 512 * 1024, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_fstn_fc1_bias, params["feat.fstn.fc1.bias"].data(), sizeof(float) * 512, cudaMemcpyHostToDevice);
    // feat_fstn_bn4_res = bacthnorm1d(feat_fstn_fc1_res.reshape(-1,1),"feat.fstn.bn4.weight","feat.fstn.bn4.bias","feat.fstn.bn4.running_mean","feat.fstn.bn4.running_var",512)
    cudaMalloc((void**)&d_feat_fstn_bn4_weight, sizeof(float) * 512);
    cudaMalloc((void**)&d_feat_fstn_bn4_bias,sizeof(float) * 512);
    cudaMalloc((void**)&d_feat_fstn_bn4_mean, sizeof(float) * 512);
    cudaMalloc((void**)&d_feat_fstn_bn4_var,sizeof(float) * 512);
    cudaMemcpy(d_feat_fstn_bn4_weight, params["feat.fstn.bn4.weight"].data(), sizeof(float) * 512, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_fstn_bn4_bias, params["feat.fstn.bn4.bias"].data(), sizeof(float) * 512, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_fstn_bn4_mean,params["feat.fstn.bn4.running_mean"].data(), sizeof(float) * 512, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_fstn_bn4_var, params["feat.fstn.bn4.running_var"].data(), sizeof(float) * 512, cudaMemcpyHostToDevice);
    // feat_fstn_fc2_res = linear(feat_fstn_relu_bn4_res,"feat.fstn.fc2.weight","feat.fstn.fc2.bias",512,256)
    cudaMalloc((void**)&d_feat_fstn_fc2_weight, sizeof(float) * 512 * 256); 
    cudaMalloc((void**)&d_feat_fstn_fc2_bias, sizeof(float) * 256);
    cudaMemcpy(d_feat_fstn_fc2_weight, params["feat.fstn.fc2.weight"].data(), sizeof(float) * 256 * 512, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_fstn_fc2_bias, params["feat.fstn.fc2.bias"].data(), sizeof(float) * 256, cudaMemcpyHostToDevice);
    // feat_fstn_bn5_res = bacthnorm1d(feat_fstn_fc2_res.reshape(-1,1),"feat.fstn.bn5.weight","feat.fstn.bn5.bias","feat.fstn.bn5.running_mean","feat.fstn.bn5.running_var",256)
    cudaMalloc((void**)&d_feat_fstn_bn5_weight, sizeof(float) * 256);
    cudaMalloc((void**)&d_feat_fstn_bn5_bias,sizeof(float) * 256);
    cudaMalloc((void**)&d_feat_fstn_bn5_mean, sizeof(float) * 256);
    cudaMalloc((void**)&d_feat_fstn_bn5_var,sizeof(float) * 256);
    cudaMemcpy(d_feat_fstn_bn5_weight, params["feat.fstn.bn5.weight"].data(), sizeof(float) * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_fstn_bn5_bias, params["feat.fstn.bn5.bias"].data(), sizeof(float) * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_fstn_bn5_mean,params["feat.fstn.bn5.running_mean"].data(), sizeof(float) * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_fstn_bn5_var, params["feat.fstn.bn5.running_var"].data(), sizeof(float) * 256, cudaMemcpyHostToDevice);
    // feat_fstn_fc3_res = linear(feat_fstn_relu_bn5_res,"feat.fstn.fc3.weight","feat.fstn.fc3.bias",256,64*64)
    cudaMalloc((void**)&d_feat_fstn_fc3_weight,sizeof(float) * 256 * 64 * 64); 
    cudaMalloc((void**)&d_feat_fstn_fc3_bias, sizeof(float) * 64 * 64);
    cudaMemcpy(d_feat_fstn_fc3_weight, params["feat.fstn.fc3.weight"].data(), sizeof(float) * 256 * 64 * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_fstn_fc3_bias, params["feat.fstn.fc3.bias"].data(), sizeof(float) * 64 * 64, cudaMemcpyHostToDevice);
    //   =========end STNkd=========
    // feat_conv2_res = conv1d(feat_bmm_res2.T, "feat.conv2.weight","feat.conv2.bias",64,128)
    cudaMalloc((void**)&d_feat_conv2_weight, sizeof(float) * 128 * 64);
    cudaMalloc((void**)&d_feat_conv2_bias, sizeof(float) * 128);
    cudaMemcpy(d_feat_conv2_weight, params["feat.conv2.weight"].data(), sizeof(float) * 128 * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_conv2_bias, params["feat.conv2.bias"].data(), sizeof(float) * 128, cudaMemcpyHostToDevice);
    // feat_bn2_res = bacthnorm1d(feat_conv2_res,"feat.bn2.weight","feat.bn2.bias","feat.bn2.running_mean","feat.bn2.running_var",128)


    cudaMalloc((void**)&d_feat_bn2_running_mean, sizeof(float) * 128);
    cudaMalloc((void**)&d_feat_bn2_running_var, sizeof(float) * 128);
    cudaMalloc((void**)&d_feat_bn2_weight, sizeof(float) * 128);
    cudaMalloc((void**)&d_feat_bn2_bias, sizeof(float) * 128);
    cudaMemcpy(d_feat_bn2_running_mean, params["feat.bn2.running_mean"].data(), sizeof(float) * 128, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_bn2_running_var, params["feat.bn2.running_var"].data(), sizeof(float) * 128, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_bn2_weight, params["feat.bn2.weight"].data(), sizeof(float) * 128, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_bn2_bias, params["feat.bn2.bias"].data(), sizeof(float) * 128, cudaMemcpyHostToDevice);


    // feat_conv3_res = conv1d(feat_relu_bn2_res,"feat.conv3.weight","feat.conv3.bias",128,1024)
    cudaMalloc((void**)&d_feat_conv3_weight, sizeof(float) * 1024 * 128);
    cudaMalloc((void**)&d_feat_conv3_bias, sizeof(float) * 1024);
    cudaMemcpy(d_feat_conv3_weight, params["feat.conv3.weight"].data(), sizeof(float) * 1024 * 128, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_conv3_bias, params["feat.conv3.bias"].data(), sizeof(float) * 1024, cudaMemcpyHostToDevice);
    // feat_bn3_res = bacthnorm1d(feat_conv3_res,"feat.bn3.weight","feat.bn3.bias","feat.bn3.running_mean","feat.bn3.running_var",1024)
    cudaMalloc((void**)&d_feat_bn3_running_mean, sizeof(float) * 1024);
    cudaMalloc((void**)&d_feat_bn3_running_var, sizeof(float) * 1024);
    cudaMalloc((void**)&d_feat_bn3_weight, sizeof(float) * 1024);
    cudaMalloc((void**)&d_feat_bn3_bias, sizeof(float) * 1024);
    cudaMemcpy(d_feat_bn3_running_mean, params["feat.bn3.running_mean"].data(), sizeof(float) * 1024, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_bn3_running_var, params["feat.bn3.running_var"].data(), sizeof(float) * 1024, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_bn3_weight, params["feat.bn3.weight"].data(), sizeof(float) * 1024, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_bn3_bias, params["feat.bn3.bias"].data(), sizeof(float) * 1024, cudaMemcpyHostToDevice);
    //     # ============end PointNetEncoder============
    // fc1_res = linear(feat_max_res,"fc1.weight","fc1.bias",1024,512)
    cudaMalloc((void**)&d_fc1_weight,sizeof(float) * 512 *1024); 
    cudaMalloc((void**)&d_fc1_bias, sizeof(float) * 512);
    cudaMemcpy(d_fc1_weight, params["fc1.weight"].data(), sizeof(float) * 512 *1024, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_bias, params["fc1.bias"].data(), sizeof(float) * 512, cudaMemcpyHostToDevice);
    // bn1_res = bacthnorm1d(fc1_res.reshape(-1,1),"bn1.weight","bn1.bias","bn1.running_mean","bn1.running_var",512)
    cudaMalloc((void**)&d_bn1_weight, sizeof(float) * 512);
    cudaMalloc((void**)&d_bn1_bias,sizeof(float) * 512);
    cudaMalloc((void**)&d_bn1_running_mean, sizeof(float) * 512);
    cudaMalloc((void**)&d_bn1_running_var,sizeof(float) * 512);
    cudaMemcpy(d_bn1_weight, params["bn1.weight"].data(), sizeof(float) * 512, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn1_bias, params["bn1.bias"].data(), sizeof(float) * 512, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn1_running_mean,params["bn1.running_mean"].data(), sizeof(float) * 512, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn1_running_var, params["bn1.running_var"].data(), sizeof(float) * 512, cudaMemcpyHostToDevice);
    // fc2_res = linear(relu_bn1_res,"fc2.weight","fc2.bias",512,256)
    cudaMalloc((void**)&d_fc2_weight,sizeof(float) * 256 * 512); 
    cudaMalloc((void**)&d_fc2_bias, sizeof(float) * 256);
    cudaMemcpy(d_fc2_weight, params["fc2.weight"].data(), sizeof(float) * 256 * 512, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_bias, params["fc2.bias"].data(), sizeof(float) * 256, cudaMemcpyHostToDevice);
    // bn2_res = bacthnorm1d(fc2_res.reshape(-1,1),"bn2.weight","bn2.bias","bn2.running_mean","bn2.running_var",256)
    cudaMalloc((void**)&d_bn2_weight, sizeof(float) * 256);
    cudaMalloc((void**)&d_bn2_bias,sizeof(float) * 256);
    cudaMalloc((void**)&d_bn2_running_mean, sizeof(float) * 256);
    cudaMalloc((void**)&d_bn2_running_var,sizeof(float) * 256);
    cudaMemcpy(d_bn2_weight, params["bn2.weight"].data(), sizeof(float) * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn2_bias, params["bn2.bias"].data(), sizeof(float) * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn2_running_mean,params["bn2.running_mean"].data(), sizeof(float) * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn2_running_var, params["bn2.running_var"].data(), sizeof(float) * 256, cudaMemcpyHostToDevice);
    // fc3_res = linear(relu_bn2_res,"fc3.weight","fc3.bias",256,10)
    cudaMalloc((void**)&d_fc3_weight,sizeof(float) * 10 * 256); 
    cudaMalloc((void**)&d_fc3_bias, sizeof(float) * 10);
    cudaMemcpy(d_fc3_weight, params["fc3.weight"].data(), sizeof(float) * 10 * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc3_bias, params["fc3.bias"].data(), sizeof(float) * 10, cudaMemcpyHostToDevice);
    
    int correct_num = 0;
    float *d_input;

    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < list_of_points.size(); i++) {

        int point_size = list_of_points[i].size();
        if (point_size > 67500) {
            list_of_points[i].erase(list_of_points[i].begin() + 67500, list_of_points[i].end());
            list_of_points[i].resize(67500);
        } else {
            list_of_points[i].resize(67500, 0.0f);
        }


        point_size = 67500;

        // Vector_Show(list_of_points[i].data(), 67500);
        // 将数组大小变为合适

        float *input = list_of_points[i].data();

        cudaMalloc((void**)&d_input, point_size * sizeof(float));
        cudaMemcpy(d_input, input, point_size * sizeof(float), cudaMemcpyHostToDevice);

        //模型推理
        int pre = transpose(d_input);
        // std::cout << "预测值 :  " << pre << ' ' << "标签值 : " << list_of_labels[i] << std::endl;
        if(pre == list_of_labels[i]) {
            correct_num++;
        }

        // int predicted_label = argmax(h_softmax_output, 9);

        // if (predicted_label == list_of_labels[i])
        // {
        //     correct_num++;
        // }
    }

    cudaFree(d_feat_stn_conv1_weight);
    cudaFree(d_feat_stn_conv1_bias);
    cudaFree(d_feat_stn_bn1_weight);
    cudaFree(d_feat_stn_bn1_bias);
    cudaFree(d_feat_stn_bn1_running_mean);
    cudaFree(d_feat_stn_bn1_running_var);
    cudaFree(d_feat_stn_conv2_weight);
    cudaFree(d_feat_stn_conv2_bias);
    cudaFree(d_feat_stn_bn2_running_mean);
    cudaFree(d_feat_stn_bn2_running_var);
    cudaFree(d_feat_stn_bn2_weight);
    cudaFree(d_feat_stn_bn2_bias);
    cudaFree(d_feat_stn_conv3_weight);
    cudaFree(d_feat_stn_conv3_bias);
    cudaFree(d_feat_stn_bn3_running_mean);
    cudaFree(d_feat_stn_bn3_running_var);
    cudaFree(d_feat_stn_bn3_bias);
    cudaFree(d_feat_stn_bn3_weight);
    cudaFree(d_feat_stn_fc1_weight);
    cudaFree(d_feat_stn_fc1_bias);
    cudaFree(d_feat_stn_bn4_running_mean);
    cudaFree(d_feat_stn_bn4_running_var);
    cudaFree(d_feat_stn_bn4_bias);
    cudaFree(d_feat_stn_bn4_weight);
    cudaFree(d_feat_stn_fc2_weight);
    cudaFree(d_feat_stn_fc2_bias);
    cudaFree(d_feat_stn_bn5_running_mean);
    cudaFree(d_feat_stn_bn5_running_var);
    cudaFree(d_feat_stn_bn5_bias);
    cudaFree(d_feat_stn_bn5_weight);
    cudaFree(d_feat_stn_fc3_weight);
    cudaFree(d_feat_stn_fc3_bias);
    cudaFree(d_feat_conv1_weight);
    cudaFree(d_feat_conv1_bias);
    cudaFree(d_feat_bn1_running_mean);
    cudaFree(d_feat_bn1_running_var);
    cudaFree(d_feat_bn1_weight);
    cudaFree(d_feat_bn1_bias);
    cudaFree(d_fstn_conv1_weight);
    cudaFree(d_fstn_conv1_bias);
    cudaFree(d_feat_fstn_bn1_running_mean); 
    cudaFree(d_feat_fstn_bn1_running_var);
    cudaFree(d_feat_fstn_bn1_weight); 
    cudaFree(d_feat_fstn_bn1_bias);
    cudaFree(d_feat_fstn_conv2_weight);
    cudaFree(d_feat_fstn_conv2_bias);
    cudaFree(d_feat_fstn_bn2_running_mean);
    cudaFree(d_feat_fstn_bn2_running_var);
    cudaFree(d_feat_fstn_bn2_weight);
    cudaFree(d_feat_fstn_bn2_bias);
    cudaFree(d_feat_fstn_conv3_weight);
    cudaFree(d_feat_fstn_conv3_bias);
    cudaFree(d_feat_fstn_bn3_running_mean);
    cudaFree(d_feat_fstn_bn3_running_var);
    cudaFree(d_feat_fstn_bn3_weight);
    cudaFree(d_feat_fstn_bn3_bias);
    cudaFree(d_feat_fstn_fc1_weight);
    cudaFree(d_feat_fstn_fc1_bias);
    cudaFree(d_feat_fstn_bn4_weight);
    cudaFree(d_feat_fstn_bn4_bias);
    cudaFree(d_feat_fstn_bn4_mean);
    cudaFree(d_feat_fstn_bn4_var);
    cudaFree(d_feat_fstn_fc2_weight);
    cudaFree(d_feat_fstn_fc2_bias);
    cudaFree(d_feat_fstn_bn5_weight);
    cudaFree(d_feat_fstn_bn5_bias);
    cudaFree(d_feat_fstn_bn5_mean);
    cudaFree(d_feat_fstn_bn5_var);
    cudaFree(d_feat_fstn_fc3_weight);
    cudaFree(d_feat_fstn_fc3_bias);
    cudaFree(d_feat_conv2_weight);
    cudaFree(d_feat_conv2_bias);
    cudaFree(d_feat_bn2_running_mean);
    cudaFree(d_feat_bn2_running_var);
    cudaFree(d_feat_bn2_weight);
    cudaFree(d_feat_bn2_bias);
    cudaFree(d_feat_conv3_weight);
    cudaFree(d_feat_conv3_bias);
    cudaFree(d_feat_bn3_running_mean);
    cudaFree(d_feat_bn3_running_var);
    cudaFree(d_feat_bn3_weight);
    cudaFree(d_feat_bn3_bias);
    cudaFree(d_fc1_weight);
    cudaFree(d_fc1_bias);
    cudaFree(d_bn1_weight);
    cudaFree(d_bn1_bias);
    cudaFree(d_bn1_running_mean);
    cudaFree(d_bn1_running_var);
    cudaFree(d_fc2_weight);
    cudaFree(d_fc2_bias);
    cudaFree(d_bn2_weight);
    cudaFree(d_bn2_bias);
    cudaFree(d_bn2_running_mean);
    cudaFree(d_bn2_running_var);
    cudaFree(d_fc3_weight);
    cudaFree(d_fc3_bias);

    double accuracy = static_cast<double>(correct_num) / list_of_points.size();

    // 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
    cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ':' << accuracy;

    return 0;
}

// 编译的命令为：nvcc test_new.cu -o test_new -Xcompiler "-O3 -std=c++14" -g -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp