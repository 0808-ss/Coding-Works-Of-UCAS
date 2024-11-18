# 定义编译器
CC = gcc

# 定义编译选项
CFLAGS = -Wall -g
LDFLAGS = -pthread -L/usr/lib/x86_64-linux-gnu
LDLIBS = -lssl -lcrypto

# 定义目标文件名
TARGET = http-server

# 默认目标
all: $(TARGET)

# 编译源代码文件
$(TARGET): $(TARGET).c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# 清理编译生成的文件
clean:
	rm -f $(TARGET)

# 声明伪目标，以便在没有文件名时执行默认目标
.PHONY: all clean