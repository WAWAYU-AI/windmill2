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
#define VECTOR_X 75
#define VECTOR_Y 0
#define VECTOR_Z 111
#define DIM_ERROR_DEEP 1.0
#define V_ZOOM 1.0
#define VYAW_ZOOM 1.0

const std::vector<cv::Point3f> small_armor = {
    cv::Point3f(-67.50F, 28.50F, 0), // 2,3,4,1象限顺序
    cv::Point3f(-67.50F, -28.50F, 0),
    cv::Point3f(67.50F, -28.50F, 0),
    cv::Point3f(67.50F, 28.50F, 0),
};
const std::vector<cv::Point3f> big_armor = {
    cv::Point3f(-112.50F, 28.50F, 0), // 2,3,4,1象限顺序
    cv::Point3f(-112.50F, -28.50F, 0),
    cv::Point3f(112.50F, -28.50F, 0),
    cv::Point3f(112.50F, 28.50F, 0),
};

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

void AimAuto::draw_armor_back(cv::Mat &pic, Armor &armor, int number){
    std::vector<cv::Point3f> objPoints;
    if (!gp->isBigArmor[number])
        objPoints = {
            cv::Point3f(-67.50F, 62.50F, 0), // 2,3,4,1象限顺序
            cv::Point3f(-67.50F, -62.50F, 0),
            cv::Point3f(67.50F, -62.50F, 0),
            cv::Point3f(67.50F, 62.50F, 0),
        };
    else
        objPoints = big_armor;
    std::vector<cv::Point2f> imgPoints;
    cv::Mat rVec = (cv::Mat_<double>(3, 1) << armor.angle.x, armor.angle.y, armor.angle.z);
    cv::Mat tVec = (cv::Mat_<double>(3, 1) << armor.center.x, armor.center.y, armor.center.z);
    cv::Mat _K = (cv::Mat_<double>(3, 3) << (float)gp->fx, 0, (float)gp->cx, 0, (float)gp->fy, (float)gp->cy, 0, 0, 1);
    std::vector<float> _dist = {(float)gp->k1, (float)gp->k2, (float)gp->p1, (float)gp->p2, (float)gp->k3};
    cv::projectPoints(objPoints, rVec, tVec, _K, _dist, imgPoints);
    cv::line(pic, imgPoints[0], imgPoints[1], cv::Scalar(255,255,255), 1);
    cv::line(pic, imgPoints[1], imgPoints[2], cv::Scalar(255,255,255), 1);
    cv::line(pic, imgPoints[2], imgPoints[3], cv::Scalar(255,255,255), 1);
    cv::line(pic, imgPoints[3], imgPoints[0], cv::Scalar(255,255,255), 1);
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
        int number = 0;
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
        draw_armor_back(src, tar, number);
#endif // DEBUGMODE
    }
#ifdef DEBUGMODE
    Armor armor;
    if (tar_list.size() > 0) armor = tar_list[0];
    cv::putText(src, "X: " + std::to_string(armor.center.x), cv::Point(20, 200), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 255), 1);
    cv::putText(src, "Y: " + std::to_string(armor.center.y), cv::Point(20, 250), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 255), 1);
    cv::putText(src, "Z: " + std::to_string(armor.center.z), cv::Point(20, 300), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 255), 1);
    cv::putText(src, "Yaw: " + std::to_string(armor.yaw), cv::Point(15, 350), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 255), 1);
    cv::putText(src, "x: " + std::to_string(armor.position(0)), cv::Point(15, 400), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 255), 1);
    cv::putText(src, "y: " + std::to_string(armor.position(1)), cv::Point(15, 450), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 255), 1);
    cv::putText(src, "z: " + std::to_string(armor.position(2)), cv::Point(15, 500), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 255), 1);
    cv::putText(src, "pitch: " + std::to_string(ts.message.pitch), cv::Point(15, 550), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 255), 1);
    cv::putText(src, "yaw: " + std::to_string(ts.message.yaw), cv::Point(15, 600), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 255), 1);
#endif // DEBUGMODE

    tracker->track(tar_list, ts, dt);

#ifdef DEBUGMODE
    tracker -> draw(tar_list);
    if (ts.message.crc){
        cv::putText(src, "latency: " + std::to_string(ts.message.latency), cv::Point(1050, 150), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 1);
        cv::putText(src, "xc: " + std::to_string(ts.message.x_c), cv::Point(1130, 200), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 1);
        cv::putText(src, "vx: " + std::to_string(ts.message.v_x), cv::Point(1130, 250), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 1);
        cv::putText(src, "yc: " + std::to_string(ts.message.y_c), cv::Point(1130, 300), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 1);
        cv::putText(src, "vy: " + std::to_string(ts.message.v_y), cv::Point(1130, 350), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 1);
        cv::putText(src, "z1: " + std::to_string(ts.message.z1 ), cv::Point(1130, 400), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 1);
        cv::putText(src, "z2: " + std::to_string(ts.message.z2 ), cv::Point(1130, 450), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 1);
        cv::putText(src, "vz: " + std::to_string(ts.message.v_z), cv::Point(1130, 500), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 1);
        cv::putText(src, "r1: " + std::to_string(ts.message.r1 ), cv::Point(1130, 550), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 1);
        cv::putText(src, "r2: " + std::to_string(ts.message.r2 ), cv::Point(1130, 600), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 1);
        cv::putText(src, "yaw: " + std::to_string(ts.message.yaw_a), cv::Point(1110, 650), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 1);
        cv::putText(src, "vyaw: " + std::to_string(ts.message.vyaw), cv::Point(1100, 700), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 1);
    }
#endif // DEBUGMODE
    
}
void AimAuto::pnp_solve(UnsolvedArmor &armor, Translator &ts, cv::Mat &src, Armor &tar, int number)
{
    //===============pnp解算===============//
    std::vector<cv::Point3f> objPoints;
    if (!gp->isBigArmor[number])
        objPoints = small_armor;
    else 
        objPoints = big_armor;
    cv::Mat rVec, tVec, _K, _dist;
    tVec.create(3, 1, CV_64F);
    rVec.create(3, 1, CV_64F);
    _K = (cv::Mat_<double>(3, 3) << (float)gp->fx, 0, (float)gp->cx, 0, (float)gp->fy, (float)gp->cy, 0, 0, 1);//相机的内参矩阵
    _dist = (cv::Mat_<float>(1, 5) << (float)gp->k1, (float)gp->k2, (float)gp->p1, (float)gp->p2, (float)gp->k3);//相机的畸变系数
    std::vector<cv::Point2f> tmp = {armor.left_light.top, armor.left_light.bottom, armor.right_light.bottom, armor.right_light.top};
    cv::solvePnP(objPoints,tmp,_K,_dist,rVec,tVec,false,cv::SOLVEPNP_IPPE);
    
    //=================坐标系转换================//
    tar.center = cv::Point3f(tVec.at<double>(0), tVec.at<double>(1), tVec.at<double>(2));
    cv::Mat rotation_matrix;
    cv::Rodrigues(rVec, rotation_matrix);
    double yaw = std::atan2(rotation_matrix.at<double>(0, 2), rotation_matrix.at<double>(2, 2));//储存装甲板信息
    if (yaw < 0){
        yaw = - yaw - M_PI;
    }else{
        yaw = M_PI - yaw;
    }
    tar.angle = cv::Point3f(rVec.at<double>(0), rVec.at<double>(1), rVec.at<double>(2));
    tar.color = gp->color;
    tar.type = number;
    tar.apex[0] = armor.left_light.top;
    tar.apex[1] = armor.left_light.bottom;
    tar.apex[2] = armor.right_light.bottom;
    tar.apex[3] = armor.right_light.top;
    
#ifdef DEBUGMODE
    cv::putText(src, "PnpYaw:" + std::to_string(yaw), cv::Point(500, 200), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 2);
    // cv::imshow("result", src);
#endif
    Eigen::MatrixXd m_pitch(3, 3);//pitch旋转矩阵
    Eigen::MatrixXd m_yaw(3, 3);//yaw旋转矩阵
    ts.message.yaw = fmod(ts.message.yaw, 2 * M_PI);
    m_yaw << cos(ts.message.yaw), -sin(ts.message.yaw), 0, sin(ts.message.yaw), cos(ts.message.yaw), 0, 0, 0, 1;
    m_pitch << cos(ts.message.pitch), 0, -sin(ts.message.pitch), 0, 1, 0, sin(ts.message.pitch), 0, cos(ts.message.pitch);
    Eigen::Vector3d temp;
    temp = Eigen::Vector3d(tar.center.z + VECTOR_X, -tar.center.x + VECTOR_Y, -tar.center.y + VECTOR_Z);
    tar.yaw = - ts.message.yaw + yaw;//装甲板yaw
    Eigen::MatrixXd r_mat = m_yaw * m_pitch;//旋转矩阵
    tar.position = r_mat * temp;
    cv::Mat a(3, 3, CV_64F, r_mat.data());
    cv::Mat b = (cv::Mat_<double>(3, 3) << 0, 0, 1, -1, 0, 0, 0, -1, 0);
    rotation_matrix = a * b * rotation_matrix;
    cv::Rodrigues(rotation_matrix, rVec);
    tar.rVec = rVec;    // 世界系到车体系旋转向量
    //=========================================//
}