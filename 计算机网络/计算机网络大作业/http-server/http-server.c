#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/sendfile.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <openssl/ssl.h>
#include <openssl/err.h>

#define PORT_HTTP 8000
#define PORT_HTTPS 4430
#define resource_dir "./resource_dir"
#define NotFundpage "./resource_dir/404.html"

// 定义请求方法的枚举类型
typedef enum {
    HTTP_UNKNOWN, // 未知请求类型
    HTTP_GET,
    HTTP_POST
} HttpRequestMethod;

typedef struct {
    HttpRequestMethod method;  // 请求方法
    char path[256];            // 请求路径
    char version[16];          // HTTP版本
} HttpRequestLine;

// 定义Range结构体
typedef struct {
    int start; // 范围的起始字节
    int end;   // 范围的结束字节
} Range;

const char *status_line_200 = "HTTP/1.1 200 OK"; // 例如："HTTP/1.1 404 Not Found"
const char *status_line_301 = "HTTP/1.1 301 Moved Permanently";
const char *status_line_404 = "HTTP/1.1 404 Not Found";
const char *status_line_206 = "HTTP/1.1 206 Partial Content";



void parse_http_request(char *buffer, int size, HttpRequestLine *RequestLine, Range* Range_info) {
    // 本次计网实验解析请求行即可

    // 似乎还得注意请求头的range信息

    // 解析请求行
    char method[16];  // 请求方法
    char path[256];   // 请求路径
    char version[16]; // HTTP版本

    // 提取请求行
    char *request_line = strtok(buffer, "\r\n"); // 获取第一行

    if (request_line) {
        // 使用sscanf进行格式化输入
        int matched = sscanf(request_line, "%15s %255s %15s", method, path, version);
        
        if (matched == 3) {
            // 根据请求方法设置枚举值
            if (strcmp(method, "GET") == 0) {
                RequestLine->method = HTTP_GET;
            } else if (strcmp(method, "POST") == 0) {
                RequestLine->method = HTTP_POST;
            } else {
                RequestLine->method = HTTP_UNKNOWN;
            }
            
            // 复制解析结果到RequestLine结构体
            strncpy(RequestLine->path, path, sizeof(RequestLine->path) - 1);
            RequestLine->path[sizeof(RequestLine->path) - 1] = '\0'; // 确保字符串以NULL结尾
            strncpy(RequestLine->version, version, sizeof(RequestLine->version) - 1);
            RequestLine->version[sizeof(RequestLine->version) - 1] = '\0'; // 确保字符串以NULL结尾

        } else {
            printf("请求行不完整 \n");
        }
    } else {
        printf("无法提取buf的第一行\n");
    }
    char *line, *next_line;

    line = strtok(NULL, "\r\n");
    int start=-1, end = -1; // 初始化end为-1，表示没有结束字节
    while (1) {

        // printf("line: %s \n", line);
        // 检查这一行是否包含"Range:"
        if (strstr(line, "Range:") != NULL) {
            // 找到包含"Range:"的行，处理这一行
            char *range_value = strstr(line, "Range: bytes=");
            if (range_value) {
                // 提取Range的值
                char *range = range_value + strlen("Range: bytes=");
                
                // 使用sscanf提取Range的值
                if (sscanf(range, "%d-%d", &start, &end) == 2) {
                    // 成功提取起始和结束字节
                    // printf("Range: %d-%d\n", start, end);


                } else if (sscanf(range, "%d", &start) == 1) {
                    // 只提取了起始字节，没有结束字节
                    // Range_info->start = start;
                    // printf("Range: %d-\n", start);
                } else {
                    printf("解析请求头部的range失败\n");
                }
            }
        }
        // 继续处理下一行
        next_line = strtok(NULL, "\r\n");
        // printf("n_line: %s \n", next_line);

        if (next_line) {
            line = next_line;
        } else {
            break;
        }
    }
    Range_info->start = start;
    Range_info->end = end;


}

// 发送HTTP响应头
int send_http_response_header(int socket_fd, const char *status_line) {
    char http_header[1024];
    int header_len = snprintf(http_header, sizeof(http_header), "%s\r\nConnection: close\r\n\r\n", status_line);
    if (header_len <= 0 || send(socket_fd, http_header, header_len, 0) != header_len) {
        return -1;
    }
    return 0;
}

int send_http_response_header_location(int socket_fd, const char *status_line, HttpRequestLine Request) {
    char http_header[1024];
    int header_len = snprintf(http_header, sizeof(http_header), "%s\r\nLocation: https://127.0.0.1:4430%s\r\nConnection: close\r\n\r\n", status_line, Request.path);
    if (header_len <= 0 || send(socket_fd, http_header, header_len, 0) != header_len) {
        return -1;
    }
    return 0;
}
// 构造了301的响应头

// 发送文件内容到socket
int send_file_to_socket(int socket_fd, int file_fd) {
    off_t offset = 0;
    ssize_t sent;
    while ((sent = sendfile(socket_fd, file_fd, &offset, 1024 * 1024)) > 0) {
        printf("文件发送成功\n");
        // 成功发送了数据
    }
    if (sent == -1) {
        // 发送失败
        perror("文件发送失败");
        return -1;
    }
    close(file_fd);
    return 0;
}

int send_http_response_with_file(int socket_fd, const char *file_path) {
    // 打开文件
    int file_fd = open(file_path, O_RDONLY);

    if (file_fd == -1) {
        perror("文件打开错误");
        // 其实这里应该返回404

        if (send_http_response_header(socket_fd, status_line_404) == -1) {
            perror("响应头发送错误");   
            return -1;
        }
        // 发送了404报头
        int file_fd_new = open(NotFundpage, O_RDONLY);

        if (send_file_to_socket(socket_fd, file_fd_new) == -1) {
            return -1;
        }
        close(file_fd_new);


        return 0;
    }

    // 获取文件大小
    off_t file_size = lseek(file_fd, 0, SEEK_END);
    lseek(file_fd, 0, SEEK_SET);

    // 发送HTTP响应头
    if (send_http_response_header(socket_fd, status_line_200) == -1) {
        perror("响应头发送错误");
       
        return -1;
    }

    // 发送文件内容
    if (send_file_to_socket(socket_fd, file_fd) == -1) {
        return -1;
    }

    // 关闭文件描述符
    close(file_fd);
    return 0;
}

int send_file_range(int new_socket, const char* filePath, Range range_info) {

    // 发送HTTP响应头
    if (send_http_response_header(new_socket, status_line_206) == -1) {
        perror("响应头发送错误");
       
        return -1;
    }

    int file_fd = open(filePath, O_RDONLY);
    if (file_fd == -1) {
        perror("打开文件失败");
        return -1;
    }

        // 定位到start字节的位置
    if (lseek(file_fd, range_info.start, SEEK_SET) == -1) {
        perror("定位文件失败");
        close(file_fd);
        return -1;
    }

    char buffer[60000]; // 足够大以容纳从start到end的字节
    int bytes_to_send;
    int bytes_sent = 0;
    if(range_info.end == -1) {
        // 这里应该是start到文件末尾
        struct stat file_struct;
        if (fstat(file_fd, &file_struct) == -1) {
            perror("获取文件状态失败");
            close(file_fd);
            return -1;
        }
        bytes_to_send = file_struct.st_size - range_info.start;

    } else {
        bytes_to_send = range_info.end - range_info.start + 1;
    }

    // 直接读取并发送指定范围的字节
    int bytes_read = read(file_fd, buffer, bytes_to_send);
    if (bytes_read > 0) {
        int bytes_written = send(new_socket, buffer, bytes_read, 0);
        if (bytes_written < 0) {
            perror("发送文件内容失败");
        } else {
            bytes_sent += bytes_written;
        }
    } else {
        perror("读取文件失败");
    }
    // 现在完成了指定字节的读取，但是问题是大小很小


    close(file_fd);
    return (bytes_sent > 0) ? 0 : -1;
}

void* handle_connection(void* arg, int port) {

    int new_socket = (int)(arg);
    char buffer[4096];
    int bytes_received = -1;

    if(port == 8000) {

    
        bytes_received = recv(new_socket, buffer, sizeof(buffer), 0);
        if (bytes_received < 0) {
            perror("socket接收错误");
            return NULL;
        }

        buffer[bytes_received] = '\0';
        HttpRequestLine get_request;
        Range range_info;
        parse_http_request(buffer, 1024, &get_request, &range_info);
        send_http_response_header_location(new_socket, status_line_301, get_request);
        printf("已经转发一个请求：%s\n", get_request.path);
        // 完成了80端口需要的转发工作
        return NULL;


    } else if (port == 4430) {


        // 现在的问题是OpenSSL库似乎没有正确安装 11/8/11：24
        // 现在的问题就是https发送过来的数据无法得到？


        bytes_received = recv(new_socket, buffer, sizeof(buffer), 0);

        if (bytes_received < 0) {
            perror("socket接收错误");
            return NULL;
        }
        buffer[bytes_received] = '\0';

        printf("Received request: %s\n", buffer);
        HttpRequestLine get_request;
        Range range_info;
        parse_http_request(buffer, 4096, &get_request, &range_info);
        // printf("Received request: %d\n", get_request.method);
        // range_info的信息符合预期
        // 接受的操作符合预期

        char filePath[4096];
        snprintf(filePath, sizeof(filePath), "%s%s", resource_dir, get_request.path);
    
        //printf("File Path: %s\n", filePath);
        if(range_info.start == -1 && range_info.end == -1) {
            if (send_http_response_with_file(new_socket, filePath) == -1) {
                printf("send_http_response_with_file函数调用失败\n");
            } else {
                printf("send_http_response_with_file函数调用成功\n");
            }
        } else if (range_info.end == -1){
             // 完成start到文件尾部的发送
            if (send_file_range(new_socket, filePath, range_info) == -1) {
                printf("start到尾部发送文件失败\n");
            } else {
                printf("start到尾部发送文件成功\n");
            }
        } else {
            // range_info.end--; // 因为end是独占的，所以需要减1
            // 这行似乎没必要
            if (send_file_range(new_socket, filePath, range_info) == -1) {
                printf("start到end发送文件失败\n");
            } else {
                printf("start到end发送文件成功\n");
            }

        }


        return NULL;

    } else {
        printf("使用了错误的端口号");
    }

    // 在port4430上的https请求打印不出来,应该是解密的问题

    // 这里经过测试可以收到来自浏览器的请求


}

// 线程函数，用于监听指定端口
void* listen_port(void* arg) {

    pthread_t thread_id = pthread_self();
    unsigned long thread_num = (unsigned long)thread_id;
    printf("This is ptd  %lu ...\n", thread_num);
    // 保证了线程函数的正确性

    int port = *((int*)arg);
    int server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);

    // 创建socket文件描述符
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("create socket failed");
        exit(EXIT_FAILURE);
    }
    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt failed");
        exit(EXIT_FAILURE);
    }
    // 设置快速可重用

    // 绑定socket到端口
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = inet_addr("127.0.0.1");
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address))<0) {
        perror("socket bind failed");
        exit(EXIT_FAILURE);
    }

    // 监听端口
    if (listen(server_fd, 3) < 0) {
        perror("listen error");
        exit(EXIT_FAILURE);
    }

    printf("Listening on port %d...\n", port);

    int num = 1;

    // 接受连接
    while ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen))>0 && num > 0) {
        printf("Connection accepted\n");
        // num--;
        

        // 这里可以处理新的连接，例如创建新的线程来处理
        handle_connection((void *)(new_socket), port);
        // 已经完成了连接

        close(new_socket); // 示例中我们只是关闭连接
    }

    return NULL;
}

void* listen_port_https(void* arg) { 
    pthread_t thread_id = pthread_self();
    unsigned long thread_num = (unsigned long)thread_id;
    printf("This is ptd  %lu ...\n", thread_num);

    int sockfd, new_fd;
    socklen_t len;
    struct sockaddr_in my_addr, their_addr;
    unsigned int myport, lisnum;
    
    myport = 4430;

    char buf[4096];

    SSL_CTX *ctx;

 /* SSL 库初始化 */
    SSL_library_init();
    /* 载入所有 SSL 算法 */
    OpenSSL_add_all_algorithms();
    /* 载入所有 SSL 错误消息 */
    SSL_load_error_strings();
    /* 以 SSL V2 和 V3 标准兼容方式产生一个 SSL_CTX ，即 SSL Content Text */
    ctx = SSL_CTX_new(SSLv23_server_method());
    /* 也可以用 SSLv2_server_method() 或 SSLv3_server_method() 单独表示 V2 或 V3标准 */
    if (ctx == NULL) {
        ERR_print_errors_fp(stdout);
        exit(1);
    }
    /* 载入用户的数字证书， 此证书用来发送给客户端。 证书里包含有公钥 */
    if (SSL_CTX_use_certificate_file(ctx, "./CAcert.pem", SSL_FILETYPE_PEM) <= 0) {
        ERR_print_errors_fp(stdout);
        exit(1);
    }
    /* 载入用户私钥 */
    if (SSL_CTX_use_PrivateKey_file(ctx, "./privkey.pem", SSL_FILETYPE_PEM) <= 0){
        ERR_print_errors_fp(stdout);
        exit(1);
    }
    /* 检查用户私钥是否正确 */
    if (!SSL_CTX_check_private_key(ctx)) {
        ERR_print_errors_fp(stdout);
        exit(1);
    }

    /* 开启一个 socket 监听 */
    if ((sockfd = socket(PF_INET, SOCK_STREAM, 0)) == -1) {
        perror("获得socket错误");
        exit(1);
    } else
        // printf("监听4430端口socket created\n");
    bzero(&my_addr, sizeof(my_addr));

    my_addr.sin_family = PF_INET;
    my_addr.sin_port = htons(4430);
    my_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    int yes = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1) {
        perror("设置socket快速可重用错误");
        exit(1);
    }

    if (bind(sockfd, (struct sockaddr *) &my_addr, sizeof(struct sockaddr))
        == -1) {
        perror("bind 错误\n");
        exit(1);
    } else {
        // printf("bind 成功\n");
    }

    if (listen(sockfd, 2) == -1) {
        perror("listen 错误");
        exit(1);
    } else
        printf("Listening on port 4430...\n");

    while (1) {

        SSL *ssl;
        len = sizeof(struct sockaddr);
        /* 等待客户端连上来 */
        if ((new_fd = accept(sockfd, (struct sockaddr *)&my_addr, &len)) == -1) {
            
            perror("accept 错误\n");
            exit(errno);

        } else
            printf("got connection from %s, port %d, socket %d\n",
                   inet_ntoa(my_addr.sin_addr),
                   ntohs(my_addr.sin_port), new_fd);
        /* 基于 ctx 产生一个新的 SSL */

        ssl = SSL_new(ctx);
        /* 将连接用户的 socket 加入到 SSL */
        SSL_set_fd(ssl, new_fd);
        /* 建立 SSL 连接 */
        if (SSL_accept(ssl) == -1) {
            perror("建立 SSL 连接错误\n");
            close(new_fd);
            break;
        }
        bzero(buf, 4096);
        len = SSL_read(ssl, buf, 4096);
        // 接收一个消息
        buf[len] = '\0';
        
        // printf("接收消息成功:%s\n",buf);
        // 已经验证过正确性

        HttpRequestLine get_request;
        Range range_info;
        parse_http_request(buf, 4096, &get_request, &range_info);
        char filePath[4096];
        snprintf(filePath, sizeof(filePath), "%s%s", resource_dir, get_request.path);
        // printf("File Path: %s\n", filePath);
        // 已经验证过正确性
        char http_header[1024];

        int file_fd = open(filePath, O_RDONLY);

        if (file_fd == -1) {
            perror("文件打开错误");
            // 其实这里应该返回404
            bzero(http_header, 1024);
            int header_len = snprintf(http_header, sizeof(http_header), "%s\r\nConnection: close\r\n\r\n", status_line_404);
            len = SSL_write(ssl, http_header, strlen(http_header));
            // 发送了404请求头
            close(file_fd);

        } else {
            printf("请求文件存在并准备发送！\n");
            bzero(http_header, 1024);

            if(range_info.end == -1 && range_info.start == -1) {
                int header_len = snprintf(http_header, sizeof(http_header), "%s\r\nConnection: close\r\n\r\n", status_line_200);
                len = SSL_write(ssl, http_header, strlen(http_header));
                // 这里面就是正常的200
                char buf_send[60000];

                bzero(buf_send, 60000);
                ssize_t bytes_read = read(file_fd, buf_send, sizeof(buf_send));
                
                len = SSL_write(ssl, buf_send, bytes_read);
                if(len > 0) {
                    printf("文件发送成功\n");
                }
                close(file_fd);

            } else {
                printf("依据头部信息进行部分发送！\n");
                int header_len = snprintf(http_header, sizeof(http_header), "%s\r\nConnection: close\r\n\r\n", status_line_206);
                // 这里面就是正常的206
                len = SSL_write(ssl, http_header, strlen(http_header));

                if(range_info.end == -1 ) {
                    // 从start到末尾
                    char buf_send[60000];
                    bzero(buf_send, 60000);
                    char buf_send_range[60000];
                    bzero(buf_send_range, 60000);
                    ssize_t bytes_read = read(file_fd, buf_send, sizeof(buf_send));
                    memcpy(buf_send_range, buf_send + range_info.start, bytes_read);
                    len = SSL_write(ssl, buf_send_range, (bytes_read - range_info.start));
                    
                    if(len > 0) {
                        printf("文件发送成功\n");
                    }

                } else {
                    // 从start到end
            
                    char buf_send[60000];
                    bzero(buf_send, 60000);
                    char buf_send_range[60000];
                    bzero(buf_send_range, 60000);
                    ssize_t bytes_read = read(file_fd, buf_send, sizeof(buf_send));

                    memcpy(buf_send_range, buf_send + range_info.start, (range_info.end - range_info.start + 1));
                    len = SSL_write(ssl, buf_send_range, (range_info.end - range_info.start + 1));
    
                }

                close(file_fd);
            }
            



        }






        close(new_fd);

        SSL_shutdown(ssl);
        /* 释放 SSL */
        SSL_free(ssl);
        /* 关闭 socket */
        close(new_fd);

    }
    
    


    // https://blog.csdn.net/sardden/article/details/42705897
    // 参考链接
    
    


}


int main() {

    pthread_t thread_id;
    int HTTP = PORT_HTTP;
    int HTTPS = PORT_HTTPS;

    // 创建监听4430端口的线程
    pthread_create(&thread_id, NULL, listen_port_https, (void*)&HTTPS);

    // 在主线程中监听8000端口
    listen_port((void*)&HTTP);

    pthread_join(thread_id, NULL);

    return 0;
}
