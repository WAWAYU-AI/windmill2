#ifndef _LIB_OVALENE
#define _LIB_OVALENE
#include "opencv2/core/types.hpp"
#include <chrono>
#include <globalParam.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <random>
#include <stdlib.h>
#include <time.h>
#include <vector>

// 在循环之前声明实例

// 当有检测到有两个装甲板的时候，传入
class ovalene
{
private:
    GlobalParam gp;

    cv::Mat empty_mat;
    // cv::Mat tvec, rvec, rotation_matrix;

    cv::Point2f camera;
    cv::Point2f center_point;

    // 需要调整这个来更改显示效果
    double meter2pixel = 1200.0f;

    // 调参部分,建议仅调节Z_zoom
    const double Z_zoom = 0.42;
    const double xy_zoom = 1.0;

    // const double height_of_armor = 57.0;
    // const double width_of_armor = 135.0;

    double R1 = 0;
    double R2 = 0;
    double angle_origin = 0;

    bool global_draw_line = 0;
    bool global_output = 0;
    bool is2armors;
    bool reachAble = 1;

    // double start_time = 0.0;
    // double curent_time = 0.0;
    double FrameTime = 6;
    std::vector<std::vector<cv::Mat>> pnp_objects;
    struct YawState
    {
    public:
        double last_yaw;
        double current_yaw;
        // double next_yaw;
        double speed;

        double yaw;
        // static const int _size = 10;
        std::vector<double> yaw_list;
        std::vector<double> time_list;
        int index = 0;

        YawState()
        {
            last_yaw = 0;
            current_yaw = 0;
            // next_yaw = 0;
        }
    };
    YawState yaw_state;

public:
    double get_vyaw()
    {
        if (this->reachAble)
        {
            return this->yaw_state.speed;
        }
        else
        {
            return 1.14514e-5;
        }
    }

    double calculate_vyaw(double yaw, double time)
    {
        // this->curent_time = time;
        this->yaw_state.yaw_list.push_back(yaw);
        this->yaw_state.time_list.push_back(time);

        bool switch_flag = abs((yaw - yaw_state.yaw_list.back()) / yaw) > 0.5;

        if (yaw_state.yaw_list.size() != 0 and switch_flag)
        {
            this->yaw_state.speed = (yaw_state.yaw_list[0] - yaw_state.yaw_list[yaw_state.yaw_list.size() - 2]) / (yaw_state.time_list[0] - yaw_state.time_list[yaw_state.time_list.size() - 2]);
            yaw_state.yaw_list.clear();
            yaw_state.time_list.clear();
        }

        return this->yaw_state.speed;
    }
    // void adaptive_para_adjust(double yaw, double time)
    // {
    //     this->curent_time = time;
    //     // double yaw = ts.message.yaw_a;
    //     // //std::cout << "time:::::::::::" << time
    //     //           << std::endl;
    //     yaw_state.last_yaw = yaw_state.current_yaw;
    //     yaw_state.current_yaw = yaw;

    //     if (this->yaw_state.yaw_list.size() == 0)
    //     {
    //         this->start_time = this->curent_time;
    //     }

    //     //std::cout << "tagdansndlaksndklanskdnl";
    //     //std::cout << this->yaw_state.current_yaw << "  " << this->yaw_state.last_yaw;

    //     if (abs((this->yaw_state.current_yaw - this->yaw_state.last_yaw) / this->yaw_state.current_yaw) > 0.5 and this->yaw_state.yaw_list.size() > 5)
    //     {
    //         // 检测到大变动，先结算
    //         //std::cout << "tagdansndlaksndklanskdnl";
    //         this->calculate_speed();
    //         yaw_state.yaw_list.clear();
    //         yaw_state.yaw_list.push_back(yaw_state.current_yaw);
    //     }
    //     else if(yaw!=0)
    //     {
    //         yaw_state.yaw_list.push_back(yaw);
    //     }
    //     //std::cout << "speed:::::::::" << yaw_state.speed << std::endl;
    // }

    // void calculate_speed()
    // {
    //     int len = this->yaw_state.yaw_list.size() - 1;
    //     double speed = (this->yaw_state.yaw_list[0] - this->yaw_state.yaw_list[len - 1]) / (this->curent_time - this->start_time);
    //     this->yaw_state.speed = speed * 1000;
    //     //std::cout << "duration   " << this->curent_time - this->start_time << std::endl;
    // }

    void init(cv::Size2d size)
    {
        empty_mat = cv::Mat(size, CV_8UC3, cv::Scalar(255, 255, 255));
    }

    void clear()
    {
        pnp_objects.clear();
    } // 每一帧之前clear一下

    void input(cv::Mat tvec, cv::Mat rvec, cv::Mat rotation_matrix)
    {
        pnp_objects.push_back(
            {tvec, rvec, rotation_matrix});

    } // 用于传入参数

    double get_R()
    {
        return (R1 + R2) / 2.0f / meter2pixel;
    }

    void draw()
    {
        // cv::imshow("ressss", empty_mat);
    }

    void Analysis2d()
    {
        /*
        pnp_objects.push_back({tvec, rvec, rotationMatrix, pointMat});*/

        int local_index = 0;

        camera = cv::Point2f(100, 600);

        cv::circle(empty_mat, camera, 5, cv::Scalar(255, 0, 0), 2);

        double old = 0;

        is2armors = (pnp_objects.size() == 2);
        cv::Point2f camera2armor;
        std::vector<std::vector<cv::Point2f>> cross_info_list;

        for (auto pnp_object : pnp_objects)
        {
            cv::Scalar point_color;
            if (local_index == 0)
            {
                point_color = cv::Scalar(255, 0, 0);
            }
            else
            {
                point_color = cv::Scalar(0, 0, 255);
            }
            cv::Mat tvec = pnp_object[0];

            camera2armor = cv::Point2f(
                (tvec.at<double>(2, 0)) * meter2pixel * Z_zoom / 1000,
                tvec.at<double>(0, 0) * meter2pixel * xy_zoom / 1000);

            cv::Mat thirdColumn = pnp_object[2].col(2);

            cv::Point2f vec_XandZ(thirdColumn.at<double>(2, 0), thirdColumn.at<double>(0, 0));
            // 存储深度方向和水平方向的偏移量

            this->angle_origin = std::atan2(thirdColumn.at<double>(2, 0), thirdColumn.at<double>(0, 0));
            // 计算pnp给出的yaw角度

            double angle_in_degree = -this->angle_origin / 3.1415926535f * 180.0f;
            // 计算并转换角度

            cv::circle(empty_mat, camera2armor + camera, 2, point_color, 2);

            const double len_of_line = 0.50;
            // 一个本地参数,用于计算控制显示的line长度

            cv::Point2f end_of_line = camera2armor + camera - vec_XandZ * len_of_line * meter2pixel;
            cv::Point2f lim = vec_XandZ * meter2pixel;

            if (global_draw_line)
            {
                cv::line(empty_mat, camera2armor + camera, end_of_line, cv::Scalar(0, 0, 255), 1);
            }

            std::vector<cv::Point2f> tmp =
                {camera2armor + camera,
                 end_of_line};
            //  起始点和终点
            cross_info_list.push_back(tmp);
            // 加入到交点列表

            local_index++;

            if (cross_info_list.size() == 2 && is2armors)
            {
                this->cross_point(cross_info_list, empty_mat);
                cross_info_list.clear();
            }
        }
    }
    void cross_point(std::vector<std::vector<cv::Point2f>> jiaodian, cv::Mat &mat)
    {
        cv::Point2f line1_p1 = jiaodian[0][0];
        cv::Point2f line1_p2 = jiaodian[0][1];
        cv::Point2f line2_p1 = jiaodian[1][0];
        cv::Point2f line2_p2 = jiaodian[1][1];

        // 计算直线的参数
        double A1 = line1_p2.y - line1_p1.y;
        double B1 = line1_p1.x - line1_p2.x;
        double C1 = A1 * line1_p1.x + B1 * line1_p1.y;

        double A2 = line2_p2.y - line2_p1.y;
        double B2 = line2_p1.x - line2_p2.x;
        double C2 = A2 * line2_p1.x + B2 * line2_p1.y;

        double det = A1 * B2 - A2 * B1;
        double x = (B2 * C1 - B1 * C2) / det;
        double y = (A1 * C2 - A2 * C1) / det;
        // 计算交点坐标
        this->center_point = cv::Point2f(x, y);

        R1 = std::sqrt((line1_p1.x - x) * (line1_p1.x - x) + (line1_p1.y - y) * (line1_p1.y - y));
        R2 = std::sqrt((line2_p1.x - x) * (line2_p1.x - x) + (line2_p1.y - y) * (line2_p1.y - y));

        if (global_output)
        {
            ////std::cout << "lenth:" << R1 / meter2pixel << " " << R2 / meter2pixel << std::endl;
        }

        double gap = abs(R1 / meter2pixel - R2 / meter2pixel);

        bool local_output = 0;

        if (global_output or local_output)
        {
            ////std::cout << "delta_width:" << gap << std::endl;
        }

        if (1)
        {
            cv::circle(mat, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
        }
        else
        {
            ////std::cout << "Lines are parallel, no intersection." << std::endl;
        }
    }
};

#endif //_LIB_OVALENEs