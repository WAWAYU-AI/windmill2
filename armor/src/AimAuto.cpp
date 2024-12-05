#include "Eigen/Eigen"
#include "Eigen/src/Core/Matrix.h"
#include "globalParam.hpp"
#include "globalText.hpp"
#include "monitor.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/hal/interface.h"
#include <AimAuto.hpp>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// 相机到云台转轴的平移向量
#define VECTOR_X 0.015
#define VECTOR_Y 0
#define VECTOR_Z 0.095
#define DIM_ERROR_DEEP 1.0
#define V_ZOOM 1.0
#define VYAW_ZOOM 1.0

void convertNumber(const std::string &number_s, int &number_i)
{
    if (number_s == "outpost")
    {
        number_i = 0;
    }
    else if (number_s == "guard")
    {
        number_i = 6;
    }
    else if (number_s == "base")
    {
        number_i = 7;
    }
    else if (number_s == "1" || number_s == "2" || number_s == "3" || number_s == "4" || number_s == "5")
    {
        number_i = number_s[number_s.size() - 1] - '0';
    }
    else
    {
        // continue;
        number_i = 3;
    }
}

AimAuto::AimAuto(GlobalParam *gp)
{
    // 保存全局参数及其他初始化
    detector = new Detector(*gp); // 初始化检测器
    tracker = new Tracker(*gp); // 初始化跟踪器
    this->gp = gp;
}
AimAuto::~AimAuto()
{
    delete detector;
    delete tracker;
}
void AimAuto::auto_aim(cv::Mat &src, Translator &ts, double dt)
{
    std::vector<Armor> tar_list;
    auto armors = detector->detect(src, gp->color);
    std::sort(armors.begin(), armors.end(), [&](const UnsolvedArmor &la, const UnsolvedArmor &lb)
              { return abs((double)src.cols / 2 - ((la.left_light.top + la.right_light.top + la.left_light.bottom + la.right_light.bottom) / 4).x) < abs((double)src.cols / 2 - ((lb.left_light.top + lb.right_light.top + lb.left_light.bottom + lb.right_light.bottom) / 4).x); });
    for (auto armor : armors)
    {
        int number = -1;
        convertNumber(armor.number, number);
        if (number == 7){ // base
            continue;
        }
        Armor tar;
        pnp_solve(armor, ts, src, tar, number);
        tar_list.push_back(tar);
#ifdef DEBUGMODE
        int tickness{1};
        cv::line(src, tar.apex[0], tar.apex[1], cv::Scalar(0, 0, 255), tickness);
        cv::line(src, tar.apex[1], tar.apex[2], cv::Scalar(0, 0, 255), tickness);
        cv::line(src, tar.apex[2], tar.apex[3], cv::Scalar(0, 0, 255), tickness);
        cv::line(src, tar.apex[3], tar.apex[0], cv::Scalar(0, 0, 255), tickness);
        cv::circle(src, (tar.apex[0] + tar.apex[1] + tar.apex[2] + tar.apex[3]) / 4, 5, cv::Scalar(193, 182, 255), -1);
        cv::circle(src, tar.apex[0], 3, cv::Scalar(193, 182, 255), -1);
        cv::circle(src, tar.apex[1], 3, cv::Scalar(193, 182, 255), -1);
        cv::circle(src, tar.apex[2], 3, cv::Scalar(193, 182, 255), -1);
        cv::circle(src, tar.apex[3], 3, cv::Scalar(193, 182, 255), -1);
#endif // DEBUGMODE
    }

    tracker->track(tar_list, ts, dt);
    
}
void AimAuto::pnp_solve(UnsolvedArmor &armor, Translator &ts, cv::Mat &src, Armor &tar, int number)
{
    //===============pnp解算===============//
    std::vector<cv::Point3f> objPoints;
    if (!gp->isBigArmor[number])
    { // 如果 ?小 ? 甲 ?
        if (gp->color == 1)
        {
            // printf("BLUE\n");
            objPoints = std::vector<cv::Point3f>{
                cv::Point3f(-67.50F, 28.50F, 0), // 2,3,4,1象限顺序
                cv::Point3f(-67.50F, -28.50F, 0),
                cv::Point3f(67.50F, -28.50F, 0),
                cv::Point3f(67.50F, 28.50F, 0),
            };
        }
        else
        {
            // printf("RED\n");
            objPoints = std::vector<cv::Point3f>{
                cv::Point3f(-67.50F, 28.5F, 0), // 2,3,4,1象限顺序
                cv::Point3f(-67.50F, -28.5F, 0),
                cv::Point3f(67.50F, -28.5F, 0),
                cv::Point3f(67.50F, 28.5F, 0),
            };
        }
    }
    else // 如果不是小 ? 甲 ?
        objPoints = std::vector<cv::Point3f>{
            cv::Point3f(-112.50F, 28.50F, 0), // 2,3,4,1象限顺序
            cv::Point3f(-112.50F, -28.50F, 0),
            cv::Point3f(112.50F, -28.50F, 0),
            cv::Point3f(112.50F, 28.50F, 0),
        };
    cv::Mat tVec, rVec, _K, _dist;
    tVec.create(3, 1, CV_64F);
    rVec.create(3, 1, CV_64F);
    _K = (cv::Mat_<double>(3, 3) << (float)gp->fx, 0, (float)gp->cx, 0, (float)gp->fy, (float)gp->cy, 0, 0, 1);
    _dist = (cv::Mat_<double>(1, 5) << (float)gp->k1, (float)gp->k2, (float)gp->p1, (float)gp->p2, (float)gp->k3);
    std::vector<cv::Point2f> tmp = {armor.left_light.top, armor.left_light.bottom, armor.right_light.bottom, armor.right_light.top};
    bool use_ippe_sq = 0;
    if (use_ippe_sq)
    {
        std::swap(objPoints[1], objPoints[3]);
        cv::solvePnP(objPoints, tmp, _K, _dist, rVec, tVec, false, cv::SOLVEPNP_IPPE_SQUARE);
    }
    else
    {
        cv::solvePnP(objPoints, tmp, _K, _dist, rVec, tVec, false, cv::SOLVEPNP_IPPE);
    }

    //=================坐标系转换================//
    tar.center = cv::Point3f(tVec.at<double>(0), tVec.at<double>(1), tVec.at<double>(2));
    cv::Mat rotation_matrix;
    double theta = rVec.at<double>(2);
    cv::Rodrigues(rVec, rotation_matrix);
    double yaw = std::atan2(rotation_matrix.at<double>(0, 2), rotation_matrix.at<double>(2, 2));
    if (yaw >= 0)
    {
        yaw = -(3.14 - yaw);
    }
    else
    {
        yaw = 3.14 + yaw;
    }
    //================数据 ? ?===================//

    tar.yaw = yaw;
    tar.angle = cv::Point3f(rVec.at<double>(0), rVec.at<double>(1), rVec.at<double>(2));
    tar.color = gp->color;
    tar.type = number;
    tar.apex[0] = armor.left_light.top;
    tar.apex[1] = armor.left_light.bottom;
    tar.apex[2] = armor.right_light.bottom;
    tar.apex[3] = armor.right_light.top;
    tar.distance_to_image_center = abs((double)src.cols / 2 - ((tar.apex[0] + tar.apex[1] + tar.apex[2] + tar.apex[3]) / 4).x);
#ifdef DEBUGMODE
    cv::putText(src, "PnpYaw:" + std::to_string(yaw), cv::Point(500, 200), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 2);
#endif
    if (number == 7)
    // base
    {
        tar.type = 10;
    }
    else if (number == 6)
    // sentry
    {
        tar.type = 7;
    }
    else if (number == 0)
    {
        tar.type = 11;
    }
    // ts.message.armor_flag = tar.type;
    Eigen::MatrixXd m_pitch(3, 3);
    Eigen::MatrixXd m_yaw(3, 3);
    double raw_yaw = ts.message.yaw;
    ts.message.yaw = fmod(ts.message.yaw, 2 * M_PI);
    m_yaw << cos(ts.message.yaw), -sin(ts.message.yaw), 0, sin(ts.message.yaw), cos(ts.message.yaw), 0, 0, 0, 1;
    m_pitch << cos(ts.message.pitch), 0, sin(ts.message.pitch), 0, 1, 0, -sin(ts.message.pitch), 0, cos(ts.message.pitch);
    Eigen::Vector3d temp;
    temp = Eigen::Vector3d(tar.center.z + VECTOR_X, -tar.center.x + VECTOR_Y, -tar.center.y + VECTOR_Z);
    tar.yaw -= ts.message.yaw;
    tar.yaw *= -1;
    tar.position = m_yaw * m_pitch * temp;
    // this->position_save = tar.center;
    // position is the world axis
    //=========================================//
}