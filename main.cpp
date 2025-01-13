#include "MessageManager.hpp"
#include "SerialPort.hpp"
#include "camera.hpp"
#include "globalParam.hpp"
#include "globalText.hpp"
#include <AimAuto.hpp>
#include <UIManager.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <glog/logging.h>
#include <monitor.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <pthread.h>
#include <unistd.h>
#include <filesystem>

#define RESIZE 0.5

// 全局变量参数，这个参数存储着全部的需要的参数
GlobalParam gp;
// 通信类
MessageManager MManager(gp);
#ifndef VIRTUALGRAB
// 相机类
Camera camera(gp);
#endif
#ifdef NOPORT
const int COLOR = RED;
#endif // NOPORT

// 定义双缓冲区
struct DataBuffer{
    Translator translator;
    cv::Mat pic;
    bool data_ready; // 标志数据是否准备好
    double time_stamp; // 时间戳
};
DataBuffer buffers[2]; // 两个缓冲区

// 定义线程锁和条件变量
pthread_mutex_t Mutex= PTHREAD_MUTEX_INITIALIZER;
pthread_barrier_t Barrier;

// 定义退出标志位
bool exit_flag = false;

// 读线程，负责读取串口信息以及取流
void *ReadFunction(void *arg);
// 运算线程，负责对图片进行处理，并且完成后续的要求
void *OperationFunction(void *arg);

int main(int argc, char **argv)
{
    // 初始化Glog并设置部分标志位
    // google::InitGoogleLogging(argv[0]);
    // 设置Glog输出的log文件写在address中log_address对应的地址下
    // FLAGS_log_dir = "../log";
    printf("welcome\n");
    // 实例化通信串口类
    printf("222\n");
    SerialPort *serialPort = new SerialPort(argv[1]);
    printf("333\n");
    // 设置通信串口对象初始值
    serialPort->InitSerialPort(int(*argv[2] - '0'), 8, 1, 'N');
    printf("555\n");
#ifndef NOPORT
    Translator temp_translator;
    printf("666\n");
    MManager.read(temp_translator, *serialPort);
    printf("555\n");
    // 通过电控发来的标志位是0～4还是5～9来确定是红方还是蓝方，其中0～4是红方，5～9是蓝方
    MManager.initParam(temp_translator.message.status / 5 == 0 ? RED : BLUE);
#else
    // 再没有串口的时候直接设定颜色，这句代码可以根据需要进行更改
    MManager.initParam(COLOR);
#endif // NOPORT

    // 初始化线程锁和条件变量
    pthread_barrier_init(&Barrier, nullptr, 2);
    // 输出日志，开始初始化
    pthread_t readThread;
    pthread_t operationThread;

    // 开启线程
    pthread_create(&readThread, NULL, ReadFunction, serialPort);
    pthread_create(&operationThread, NULL, OperationFunction, serialPort);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);

    // 等待线程结束
    pthread_join(readThread, NULL);
    pthread_join(operationThread, NULL);
    
    // 销毁线程锁和条件变量
    pthread_mutex_destroy(&Mutex);
    pthread_barrier_destroy(&Barrier);
    
    return 0;
}

void *ReadFunction(void *arg) // 读线程
{
    int current_buffer = 0; // 当前使用的缓冲区
    int cnt = 0;
#ifndef VIRTUALGRAB
    camera.init();
#endif 
    // 传入的参数赋给串口，以获得串口数据
    SerialPort *serialPort = (SerialPort *)arg;
    while (1)
    {   
        pthread_mutex_lock(&Mutex);
        // usleep(1000);
        // printf("read thread is running %d %d\n", ++cnt, current_buffer);
        MManager.read(buffers[current_buffer].translator, *serialPort);
        if (buffers[current_buffer].translator.message.status % 5 != 0)
        {
#ifndef VIRTUALGRAB
            camera.change_attack_mode(ENERGY, gp);
#endif
            gp.attack_mode = ENERGY;
        }
        else
        {
#ifndef VIRTUALGRAB
            camera.change_attack_mode(ARMOR, gp);
#endif
            gp.attack_mode = ARMOR;
        }
#ifndef NOPORT
        MManager.LogMessage(buffers[current_buffer].translator, gp);
        // translator.message.status =3;
        if (buffers[current_buffer].translator.message.status / 5 != gp.color)
        {
            gp.initGlobalParam(buffers[current_buffer].translator.message.status / 5);
        }
        if (buffers[current_buffer].translator.message.armor_flag != gp.armorStat)
        {
            MManager.ChangeBigArmor(buffers[current_buffer].translator);
        }
#endif// NOPORT

        buffers[current_buffer].time_stamp = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

#ifndef VIRTUALGRAB

#ifdef DEBUGMODE
        camera.set_param_mult(gp);
#endif
        camera.get_pic(&buffers[current_buffer].pic, gp);
        buffers[current_buffer].data_ready = true;

#else
        MManager.getFrame(buffers[current_buffer].pic, buffers[current_buffer].translator);
#endif
        // usleep(200 * 1000);
        pthread_mutex_unlock(&Mutex);
        // 切换缓冲区
        current_buffer = (current_buffer + 1) % 2;
        
        
        pthread_barrier_wait(&Barrier);
        // printf("read thread is end %d %d\n", cnt, current_buffer);
    }
    return NULL;
}

void *OperationFunction(void *arg)
{
    SerialPort *serialPort = (SerialPort *)arg;
    // 实例化自瞄类
    AimAuto aim(&gp);
    // 实例化UI类
    UIManager UI(gp);
    cv::Mat pic;
    Translator translator;
    double dt = 0;
    double last_time_stamp = 0;
#ifdef SHOW_FPS
    int frame_count = 0;
    double fps_time_stamp = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
#endif
#ifdef DEBUGMODE
    //=====动态调参使用参数======//
    // 当前按键
    int key = 0;
    // debug时waitKey时间，也就是整体的运行速率
    int debug_t = 1;
    // 储存相机坐标系下的点，用于绘图
    std::deque<cv::Point3f> points3d;
    // 储存当前时间，用于绘图
    std::deque<double> times;

#endif // DEBUGMODE
    //========================//
    uint8_t error_times{0};
    int processing_buffer = 1; // 当前处理的缓冲区
    int cnt = 0;
    int empty_frame_count = 0;
    // cv::waitKey(200);
    while (1)
    {
        // usleep(1000);
        // printf("operation thread is running %d  %d\n", ++cnt, processing_buffer);
#ifndef NOPORT
        translator = buffers[processing_buffer].translator;
        // translator.message.pitch = 0;
        // translator.message.yaw = 0;
#else
        MManager.FakeMessage(translator);
#endif // NOPORT
        pic = buffers[processing_buffer].pic.clone();

        buffers[processing_buffer].data_ready = false;

        if (last_time_stamp == 0) dt = 0;
        else dt = buffers[processing_buffer].time_stamp - last_time_stamp;
        last_time_stamp = buffers[processing_buffer].time_stamp;

        processing_buffer = (processing_buffer + 1) % 2;
        // 如果图片为空，不执行
        if (pic.empty()){
#ifdef VIRTUALGRAB
            pic = cv::Mat(gp.height, gp.width, CV_8UC3, cv::Scalar(0, 0, 0));
#else       
            empty_frame_count++;
            if (empty_frame_count > 3){
                printf("pic is empty\n");
                exit(0);
            }else{
                pic = cv::Mat(gp.height, gp.width, CV_8UC3, cv::Scalar(0, 0, 0));
            }
#endif
        }else{
            empty_frame_count = 0;
        }
#ifdef RECORDVIDEO // 如果开启录制视频，使用MManager类进行录制
        MManager.recordFrame(pic);
#endif
        // 自瞄模式
        if (translator.message.status % 5 == 0)
        {
            aim.auto_aim(pic, translator, dt);
            double time_stamp = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            translator.message.latency = (time_stamp - last_time_stamp) * 1000;
            MManager.write(translator, *serialPort);
// #ifdef DEBUGMODE
            // drawStat(points3d, times, translator);
            UI.receive_pic(pic);
            UI.windowsManager(key, debug_t);
            cv::Mat tmp;
            cv::resize(pic, tmp, cv::Size((int)pic.size[1] * RESIZE, (int)pic.size[0] * RESIZE), cv::INTER_LINEAR);
            cv::imshow("aimauto__", tmp);
            // usleep(200 * 1000);
// #endif

#ifndef DEBUGMODE
#ifdef SSH
            std::vector<uchar> buf;
            cv::imencode(".jpg", pic, buf); // 将帧编码为 JPEG 格式

            std::string header = "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
            send(new_socket, header.c_str(), header.size(), 0);

            std::string response = "--frame\r\nContent-Type: image/jpeg\r\n\r\n";
            response.insert(response.end(), buf.begin(), buf.end());
            response += "\r\n\r\n";
            send(new_socket, response.c_str(), response.size(), 0);
#endif
#endif

#ifdef DEBUGMODE
            key = cv::waitKey(debug_t);
            if (key == ' ')
                key = cv::waitKey(0);
            if (key == 27 || key == 'q')
                exit(0);
#endif // DEBUGMODE
        }
        if (translator.message.status == 99)
            exit(0);

#ifndef NOPORT
#ifdef SSH
        close(new_socket);
        close(server_fd);
#endif
#endif // NOPORT
#ifdef SHOW_FPS
        frame_count++;
        auto now_time_stamp = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        if (now_time_stamp - fps_time_stamp >= 1)
        {
            printf("FPS: %d  \tLatency: %.3f ms\n", frame_count, translator.message.latency);
            frame_count = 0;
            fps_time_stamp = now_time_stamp;
        }
#endif
        pthread_barrier_wait(&Barrier);
        // printf("operation thread is end %d  %d\n", cnt, processing_buffer);
        // printf(gp.color == RED ? "RED" : "BLUE");
    }
    return NULL;
}