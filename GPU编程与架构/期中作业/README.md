# 国科大作业-GPU编程与架构-期中作业
    new_py.py：是课程用于读取点云数据并进行训练模型的代码
    model_graph.txt：是这次训练用的模型的层数的具体参数说明，注意矩阵的格式
        （eg： (conv2): Conv1d(64, 128, kernel_size=(1,), stride=(1,)) 这个矩阵是128 * 64）
    torch_forward.py：是我自己写的加载已保存模型并进行推理的代码
    test_new：这份.cu代码完成的是读入pointnet测试集并进行推理的内容，主要内容是cuda编程（矩阵乘法，最大池化，向量归一化）
