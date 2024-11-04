#include "MessageManager.hpp"
#include "SerialPort.hpp"
#include "WMIdentify.hpp"
#include "WMPredict.hpp"
#include "camera.hpp"
#include "globalParam.hpp"
#include "globalParamInit.hpp"
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
#define RESIZE 0.5
// 全局变量
// 全局变量参数，这个参数存储着全部的需要的参数
GlobalParam gp;
MessageManager MManager(gp);
// 全局地址参数，这个参数存储着全部的需要的地址
address addr;
// 全局图片，负责将图片从取流的读取线程读取到运算线程
cv::Mat pic;
// 读信息，是从电控接受的信息，在读线程被赋值，之后交给运算线程
Translator translator;
// 临时信息，为了防止数据的堵塞，使用临时信息反复的读取，在需要真正获取信息的时候将信息传给translator
Translator temp;
std::chrono::microseconds last_tick = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::system_clock::now().time_since_epoch());
// 声明线程
// 读线程，负责读取串口信息以及取流
void *ReadFunction(void *arg);
// 运算线程，负责对图片进行处理，并且完成后续的要求
void *OperationFunction(void *arg);
int main(int argc, char **argv)
{
    // 初始化Glog并设置部分标志位
    google::InitGoogleLogging(argv[0]);
    // 设置Glog输出的log文件写在address中log_address对应的地址下
    FLAGS_log_dir = addr.log_address;
    printf("welcome\n");
    // 实例化通信串口类
    SerialPort *serialPort = new SerialPort(argv[1]);
    // 设置通信串口对象初始值
    serialPort->InitSerialPort(int(*argv[2] - '0'), 8, 1, 'N');
#ifndef NOPORT
    MManager.read(temp, *serialPort);
#ifdef THREADANALYSIS
    printf("init status is: %d\n", temp.message.status);
#endif
    // 通过电控发来的标志位是0～4还是5～9来确定是红方还是蓝方，其中0～4是红方，5～9是蓝方
    MManager.initParam(temp.message.status / 5 == 0 ? RED : BLUE);

#else
    // 再没有串口的时候直接设定颜色，这句代码可以根据需要进行更改
    MManager.initParam(BLUE);
    translator.message.predict_time = 0;
#endif // NOPORT
    // 输出日志，开始初始化
    pthread_t readThread;
    pthread_t WMIThread;
    // 开启线程
    pthread_create(&readThread, NULL, ReadFunction, serialPort);
    pthread_create(&WMIThread, NULL, OperationFunction, serialPort);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset); // 将线程绑定到CPU 0

    if (pthread_setaffinity_np(WMIThread, sizeof(cpu_set_t), &cpuset) != 0) {
        perror("Failed to set affinity for WMIThread");
        return 1;
    }
    
    // 开启线程成功，输出日志
    pthread_join(WMIThread, NULL);
    return 0;
}

void *ReadFunction(void *arg)
{
#ifdef THREADANALYSIS
    printf("read function init successful\n");
#endif
    // 传入的参数赋给串口，以获得串口数据
    SerialPort *serialPort = (SerialPort *)arg;
    while (1)
    {
        MManager.read(temp, *serialPort);
        usleep(100);
    }
    return NULL;
}

void *OperationFunction(void *arg)
{

#ifdef THREADANALYSIS
    printf("operation function init successful\n");
#endif
    SerialPort *serialPort = (SerialPort *)arg;
    // 实例化能量机关识别类
    WMIdentify WMI(gp);
    // 实例化自瞄类
    AimAuto aim(&gp);
    // 实例化UI类
    UIManager UI;
    // 重置能量机关识别类
    WMI.clear();
    // 实例化能量机关预测类
    WMIPredict WMIPRE;
#ifndef VIRTUALGRAB
    // 假如是现实取流，初始化相机
    Camera camera(gp);
    camera.init();
#endif // VIRTUALGRAB
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
    while (1)
    {
        // std::chrono::milliseconds start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        // std::chrono::system_clock::now().time_since_epoch());
#ifndef NOPORT
        MManager.copy(temp, translator);

#else

        MManager.FakeMessage(translator);
#endif // NOPORT
        if (MManager.CheckCrc(translator, 61))
        {

#ifndef NOPORT
            MManager.LogMessage(translator, gp);
            // translator.message.status =3;
            if (translator.message.status / 5 != gp.color)
            {
                initGlobalParam(gp, addr, translator.message.status / 5);
            }
            if (translator.message.armor_flag != gp.armorStat)
            {
                MManager.ChangeBigArmor(translator);
            }
#endif
            if (translator.message.status % 5 != 0)
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

#ifndef VIRTUALGRAB
            // camera.getFrame(pic);
#ifdef SHOW_FPS
            std::chrono::microseconds s = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch());
#endif
#ifdef DEBUGMODE
            camera.getFrame(pic);
#else
            camera.get_pic(&pic, gp);
#endif
#ifdef SHOW_FPS
            std::chrono::microseconds e = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch());
            aim.setTime(static_cast<double>(e.count()) * 1e-6);
            float t = (e - s).count();
            system("clear");
            printf("Get Frame ms:%.4f\n", t * 1e-3);
            if (t * 1e-3 > 8.0)
            {
                error_times++;
            }
            if (error_times > 20)
            {   
                printf("low grabbing\n");
                exit(-1);
            }
#endif
#else
            MManager.getFrame(pic, translator);
#endif
        }
        else
        {
#ifdef THREADANALYSIS
            // crc校验 串口发来的数据
            printf("crc was wrong\n");
#endif
            continue;
        }
        // 如果图片为空，不执行
        if (pic.empty() == 1)
        {
#ifdef THREADANALYSIS
            printf("pic is empty\n");
#endif
            exit(-1);
        }
#ifdef RECORDVIDEO
        MManager.recordFrame(pic);
#endif

        WMI.JudgeClear(translator);

        // 打击小符或者大符
        if (translator.message.status % 5 == 1 || translator.message.status % 5 == 3)
        {

#ifdef DEBUGMODE
            times.push_back((double)translator.message.predict_time / 1000);
#endif // DEBUGMODE
#ifdef THREADANALYSIS
            printf("status == 1 || status == 3\n");
#endif

            if (!WMIPRE.BulletSpeedProcess(translator, gp))
            {
                translator.message.status = 102;
            }

#ifdef USEWMNET
            // 进行识别
            WMI.startWMINet(pic, translator);
#else
            WMI.startWMIdentify(pic, translator);
#endif
            // 进行预测

            if (!WMIPRE.StartPredict(translator, gp, WMI))
            {
                translator.message.status = 102;
                std::cout << "WM indentity failed" << std::endl;
            }

#ifdef DEBUGMODE
            // 如果开启DEBUGMODE，使用UI类在图片上绘制UI
            UI.receive_pic(pic);
            // 通过按键进行调参，这里的顺序必须是先这个再按键
            UI.windowsManager(gp, key, debug_t);
#ifdef THREADANALYSIS
            printf("picture showed\n");
#endif
            cv::imshow("result", pic);

            // 获取按键，用于动态调参
            key = cv::waitKey(debug_t);
            if (key == ' ')
                cv::waitKey(0);
            if (key == 27)
                return nullptr;
#endif // DEBUGMODE
        }
        else if (translator.message.status % 5 == 0)
        {
            //    std::cout<<1111<<std::endl;
#ifdef THREADANALYSIS
            // printf("status == 0\n");
#endif
#ifdef DEBUGMODE

            times.push_back((double)translator.message.predict_time / 1000);
            // uint32_t time_stamp = translator.message.predict_time;
#endif // DEBUGMODE
            aim.AimAutoYHY(pic, translator);

            aim.NewTracker(translator, pic);
            MManager.HoldMessage(translator);

#ifdef DEBUGMODE
            drawStat(points3d, times, translator);
            UI.receive_pic(pic);
            UI.windowsManager(gp, key, debug_t);
            cv::Mat tmp;
            cv::resize(pic, tmp, cv::Size((int)pic.size[1] * RESIZE, (int)pic.size[0] * RESIZE), cv::INTER_LINEAR);
            // cv::resize(pic, tmp, cv::Size((int)pic.size[1] * 0.75, (int)pic.size[0] * 0.75), cv::INTER_LINEAR);
            cv::imshow("aimauto__", tmp);
#endif // DEBUGMODE

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
            if (key == 27)
                return nullptr;
#endif // DEBUGMODE
        }
#ifdef SHOW_FPS
        // std::chrono::milliseconds end_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        //     std::chrono::system_clock::now().time_since_epoch());
        // float total_time = (end_ms - start_ms).count() / 1000.0;
        // alltime += total_time;
        // times_++;
        // printf("Current fps:%.4f | Average fps:%.4f\n", 1 / total_time, 1 / alltime * times_++);
        std::chrono::microseconds this_tick = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch());
        double real_time = (this_tick - last_tick).count();
        last_tick = this_tick;
        // alltime += real_time;
        // printf("real_ms:%.4f|real_fps:%.4f|average_ms:%.4lf\n", real_time * 1e-3, 1 / real_time * 1e6, alltime / ++times_);
        double real_fps = 1 / real_time * 1e6;
        // translator.message.vx_c = real_fps;
        printf("real_ms:%.4f|real_fps:%.4f\n", real_time * 1e-3, real_fps);
#endif
        if (translator.message.status == 99)
            return nullptr;
            // std::chrono::milliseconds end_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            //     std::chrono::system_clock::now().time_since_epoch());
            // float total_time = (end_ms - start_ms).count() / 1000.0;
            // std::cout << "fpss: " << 1 / total_time << std::endl;

#ifndef NOPORT
        MManager.UpdateCrc(translator, 61);
        MManager.write(translator, *serialPort);
        // std::cout<<"status:"<<+translator.message.status <<std::endl;

#ifdef SSH
        close(new_socket);
        close(server_fd);
#endif
#endif // NOPORT
    }
    return NULL;
}