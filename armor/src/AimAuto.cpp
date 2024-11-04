#include "Eigen/Eigen"
#include "Eigen/src/Core/Matrix.h"
// #include "fftw_omega.hpp"
#include "gaoning.hpp"
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

int binary_threshold = 0;

AimAuto::AimAuto(GlobalParam *gp) : dt(1e-3f), level_count{0, 0, 0}, restart_time(0.0), ArmorSpeed_array(), isBigArmor(), translational_speed_level(0), rotate_speed_level(0), last_world_state(1e-3, 1e-3, 1e-3), current_world_state(1e-3, 1e-3, 1e-3), src_size(1440, 1080), closest2Armors(), config0(), config1(), time_add(0)
{
    empty = cv::Mat(1000, 1000, CV_8UC3, cv::Scalar(255, 255, 255));

    binary_threshold = gp->binary_thres;
    std::cout << binary_threshold << std::endl;
    det = initDetector(gp->color);
    z_list = new double[this->z_len + 1];
    this->getParam();
    tracker_0 = new Tracker(config0.max_match_distance, config0.max_match_yaw_diff);
    tracker_1 = new Tracker(config1.max_match_distance, config1.max_match_yaw_diff);
    auto f = [this](const Eigen::VectorXd &x)
    {
        Eigen::VectorXd x_new = x;
        x_new(0) += x(1) * dt;
        x_new(2) += x(3) * dt;
        x_new(4) += x(5) * dt;
        x_new(6) += x(7) * dt;
        return x_new;
    };
    // J_f - Jacobian of process function
    auto j_f = [this](const Eigen::VectorXd &)
    {
        Eigen::MatrixXd f(9, 9);
        // clang-format off
    f <<  1,   dt, 0,   0,   0,   0,   0,   0,   0,
          0,   1,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   1,   dt, 0,   0,   0,   0,   0, 
          0,   0,   0,   1,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   1,   dt, 0,   0,   0,
          0,   0,   0,   0,   0,   1,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   1,   dt, 0,
          0,   0,   0,   0,   0,   0,   0,   1,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   1;
        // clang-format on
        return f;
    };
    // h - Observation function
    auto h = [](const Eigen::VectorXd &x)
    {
        Eigen::VectorXd z(4);
        double xc = x(0), yc = x(2), yaw = x(6), r = x(8);
        z(0) = xc - r * cos(yaw); // xa
        z(1) = yc - r * sin(yaw); // ya
        z(2) = x(4);              // za
        z(3) = x(6);              // yaw
        return z;
    };
    // J_h - Jacobian of observation function
    auto j_h = [](const Eigen::VectorXd &x)
    {
        Eigen::MatrixXd h(4, 9);
        double yaw = x(6), r = x(8);
        // clang-format off
    //    xc   v_xc yc   v_yc za   v_za yaw         v_yaw r
    h <<  1,   0,   0,   0,   0,   0,   r*sin(yaw), 0,   -cos(yaw),
          0,   0,   1,   0,   0,   0,   -r*cos(yaw),0,   -sin(yaw),
          0,   0,   0,   0,   1,   0,   0,          0,   0,
          0,   0,   0,   0,   0,   0,   1,          0,   0;
        // clang-format on
        return h;
    };
    auto u_q_0 = [this]()
    {
        Eigen::MatrixXd q(9, 9);
        double t{dt}, x{config0.s2qxyz_}, y{config0.s2qyaw_}, r{config0.s2qr_};
        double q_x_x{pow(t, 4) / 4 * x}, q_x_vx{pow(t, 3) / 2 * x}, q_vx_vx{pow(t, 2) * x};
        double q_y_y{pow(t, 4) / 4 * y}, q_y_vy{pow(t, 3) / 2 * y}, q_vy_vy{pow(t, 2) * y};
        double q_r{pow(t, 4) / 4 * r};
        // clang-format off
    //    xc      v_xc    yc      v_yc    za      v_za    yaw     v_yaw   r
    q <<  q_x_x,  q_x_vx, 0,      0,      0,      0,      0,      0,      0,
          q_x_vx, q_vx_vx,0,      0,      0,      0,      0,      0,      0,
          0,      0,      q_x_x,  q_x_vx, 0,      0,      0,      0,      0,
          0,      0,      q_x_vx, q_vx_vx,0,      0,      0,      0,      0,
          0,      0,      0,      0,      q_x_x,  q_x_vx, 0,      0,      0,
          0,      0,      0,      0,      q_x_vx, q_vx_vx,0,      0,      0,
          0,      0,      0,      0,      0,      0,      q_y_y,  q_y_vy, 0,
          0,      0,      0,      0,      0,      0,      q_y_vy, q_vy_vy,0,
          0,      0,      0,      0,      0,      0,      0,      0,      q_r;
        // clang-format on
        return q;
    };
    auto u_q_1 = [this]()
    {
        Eigen::MatrixXd q(9, 9);
        double t{dt}, x{config1.s2qxyz_}, y{config1.s2qyaw_}, r{config1.s2qr_};
        double q_x_x{pow(t, 4) / 4 * x}, q_x_vx{pow(t, 3) / 2 * x}, q_vx_vx{pow(t, 2) * x};
        double q_y_y{pow(t, 4) / 4 * y}, q_y_vy{pow(t, 3) / 2 * y}, q_vy_vy{pow(t, 2) * y};
        double q_r{pow(t, 4) / 4 * r};
        // clang-format off
    //    xc      v_xc    yc      v_yc    za      v_za    yaw     v_yaw   r
    q <<  q_x_x,  q_x_vx, 0,      0,      0,      0,      0,      0,      0,
          q_x_vx, q_vx_vx,0,      0,      0,      0,      0,      0,      0,
          0,      0,      q_x_x,  q_x_vx, 0,      0,      0,      0,      0,
          0,      0,      q_x_vx, q_vx_vx,0,      0,      0,      0,      0,
          0,      0,      0,      0,      q_x_x,  q_x_vx, 0,      0,      0,
          0,      0,      0,      0,      q_x_vx, q_vx_vx,0,      0,      0,
          0,      0,      0,      0,      0,      0,      q_y_y,  q_y_vy, 0,
          0,      0,      0,      0,      0,      0,      q_y_vy, q_vy_vy,0,
          0,      0,      0,      0,      0,      0,      0,      0,      q_r;
        // clang-format on
        return q;
    };
    auto u_r_0 = [this](const Eigen::VectorXd &z)
    {
        Eigen::DiagonalMatrix<double, 4> r;
        double x = config0.r_xyz_factor;
        r.diagonal() << abs(x * z[0]), abs(x * z[1]), abs(1e-7), config0.r_yaw;
        return r;
    };
    auto u_r_1 = [this](const Eigen::VectorXd &z)
    {
        Eigen::DiagonalMatrix<double, 4> r;
        double x = config1.r_xyz_factor;
        r.diagonal() << abs(x * z[0]), abs(x * z[1]), abs(1e-7), config1.r_yaw;
        return r;
    };
    // P - error estimate covariance matrix
    Eigen::DiagonalMatrix<double, 9> p0;
    Eigen::DiagonalMatrix<double, 9> p1;
    p0.setIdentity();
    p1.setIdentity();
    tracker_0->ekf = ExtendedKalmanFilter{f, h, j_f, j_h, u_q_0, u_r_0, p0};
    tracker_1->ekf = ExtendedKalmanFilter{f, h, j_f, j_h, u_q_1, u_r_1, p1};
    tracker_0->tracking_thres = tracker_1->tracking_thres = 10;
    tracker_0->lost_thres = tracker_1->lost_thres = 10;

    this->gp = gp;
    this->tar_list.clear();
    this->last_time_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch());
    this->_K_.resize(3, 3);
    this->_K_ << (float)gp->fx, 0, (float)gp->cx, 0, (float)gp->fy, (float)gp->cy, 0, 0, 1;
}

AimAuto::~AimAuto()
{
    free(this->z_list);
}
void AimAuto::getParam()
{
    cv::FileStorage fs;
    std::string path = std::filesystem::current_path();
    path = path + "/AimautoConfig.yaml";
    // std::cout << path;
    fs.open(path, cv::FileStorage::READ);

    fs["max_match_distance"] >> config0.max_match_distance;
    fs["max_match_yaw_diff"] >> config0.max_match_yaw_diff;
    fs["s2qxyz"] >> config0.s2qxyz_;
    fs["s2qyaw"] >> config0.s2qyaw_;
    fs["s2qr"] >> config0.s2qr_;
    fs["r_xyz_factor"] >> config0.r_xyz_factor;
    fs["r_yaw"] >> config0.r_yaw;

    fs["max_match_distance_"] >> config1.max_match_distance;
    fs["max_match_yaw_diff_"] >> config1.max_match_yaw_diff;
    fs["s2qxyz_"] >> config1.s2qxyz_;
    fs["s2qyaw_"] >> config1.s2qyaw_;
    fs["s2qr_"] >> config1.s2qr_;
    fs["r_xyz_factor_"] >> config1.r_xyz_factor;
    fs["r_yaw_"] >> config1.r_yaw;

    fs["react_time_0"] >> react_time_[0];
    fs["react_time_1"] >> react_time_[1];
    fs["react_time_2_8rad"] >> react_time_[2];
    fs["react_time_2_10rad"] >> react_time_[3];
}
void AimAuto::AimAutoYHY(cv::Mat &src, Translator &ts)
{
    // ts.message.status = 2;
    this->is_hitting_outpose = ts.message.status % 5 == 2;
    if (det->detect_color != gp->color)
    {
        det = initDetector(gp->color);
    }
    this->time = (double)ts.message.predict_time;
    this->tar_list.clear();
    auto armors = det->detect(src);
    std::sort(armors.begin(), armors.end(), [&](const rm_auto_aim::Armor &la, const rm_auto_aim::Armor &lb)
              { return abs((double)src.cols / 2 - ((la.left_light.top + la.right_light.top + la.left_light.bottom + la.right_light.bottom) / 4).x) < abs((double)src.cols / 2 - ((lb.left_light.top + lb.right_light.top + lb.left_light.bottom + lb.right_light.bottom) / 4).x); });
    // std::cout << "armors.size():" << armors.size() << std::endl;
    for (auto armor : armors)
    {
        if (this->first_see)
        {
            this->first_see_time = this->time;
            this->first_see = 0;
        }
        int number = -1;
        convertNumber(armor.number, number);
        if (number == 7) // base
        {
            // continue;
        }
        Armor tar;
        pnp_solve(armor, ts, src, tar, number);
        tar_list.emplace_back(tar);
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

    if (y_camera_reserve.size() == 2)
    {
        if (y_camera_reserve[0].first.x < y_camera_reserve[1].first.x)
        {
            // std::swap(y_camera_reserve[0].y, y_camera_reserve[1].y);
            std::swap(y_camera_reserve[0], y_camera_reserve[1]);
        }
    }
    armors_msg.armors = tar_list;

    //=====================输出装甲板坐标并且在DEBUGMODE ?展示识别到的装甲 ?======================//
    if (tar_list.size() != 0)
    {
        ts.message.x_a = tar_list[0].position(0) * 1000;
        ts.message.y_a = tar_list[0].position(1) * 1000;
        ts.message.z_a = tar_list[0].position(2) * 1000;
    }
    else
    {
        ts.message.x_a = 0;
        ts.message.y_a = 0;
        ts.message.z_a = 0;
    }
    if (last_ts.message.armor_flag == 11)
    {
        if (this->rotate_speed_level == 0)
        {
            this->shootKeeper = 3;
        }
        else if (this->is_hitting_outpose == 1)
        {
            this->shootKeeper = 0;
        }
    }
    this->position_save = tar_list[0].center;
#ifdef DEBUGMODE
    showDist(tar_list, src);
    // cv::putText(src, std::to_string(shootKeeper), cv::Point(300, 300), 1, 2, cv::Scalar(225, 225, 0), 2);
    // cv::imshow("result", src);
#endif // DEBUGMODE
}
void AimAuto::pnp_solve(rm_auto_aim::Armor &armor, Translator &ts, cv::Mat &src, Armor &tar, int number)
{
    //===============pnp解算===============//
    if (this->isBigArmor.empty())
    {
        this->isBigArmor.push_back(gp->isBigArmor[number]);
    }
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
    tVec.create(3, 1, CV_64F);
    rVec.create(3, 1, CV_64F);
    _K = (cv::Mat_<double>(3, 3) << (float)gp->fx, 0, (float)gp->cx, 0, (float)gp->fy, (float)gp->cy, 0, 0, 1);
    _dist = {(float)gp->k1, (float)gp->k2, (float)gp->p1, (float)gp->p2, (float)gp->k3};
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
    // cv::Point p = cv::Point(tar.center.x + 500, (tar.center.z - 3100));
    // cv::circle(empty, p, 2, cv::Scalar(0, 255, 0), -1);
    // cv::imshow("e", empty);
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
    tar.color = det->detect_color;
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
    this->raw_yaw = ts.message.yaw;
    ts.message.yaw = fmod(ts.message.yaw, 2 * M_PI);
    m_yaw << cos(ts.message.yaw), -sin(ts.message.yaw), 0, sin(ts.message.yaw), cos(ts.message.yaw), 0, 0, 0, 1;
    m_pitch << cos(ts.message.pitch), 0, sin(ts.message.pitch), 0, 1, 0, -sin(ts.message.pitch), 0, cos(ts.message.pitch);
    Eigen::Vector3d temp;
    temp = Eigen::Vector3d(tar.center.z / 1000 + VECTOR_X, -tar.center.x / 1000 + VECTOR_Y, -tar.center.y / 1000 + VECTOR_Z);
    tar.yaw -= ts.message.yaw;
    tar.yaw *= -1;
    // if (y_camera_reserve.size() == 0)
    {
        y_camera_reserve.push_back(std::pair(cv::Point3f(-tar.center.x + VECTOR_Y * 1000, tar.center.y - VECTOR_Z * 1000, tar.center.z), tar.type));
    }
    // else if (abs(y_camera_reserve.back().z - float(tar.type)) < 1e-5 or 1)
    // {
    //     y_camera_reserve.push_back(cv::Point3f(-tar.center.x / 1000 + VECTOR_Y, tar.center.y - VECTOR_Z * 1000, tar.type));
    // }
    // else
    // {
    // }
    tar.position = m_yaw * m_pitch * temp;
    this->position_save = tar.center;
    // position is the world axis
    //=========================================//
}

void AimAuto::NewTracker(Translator &ts, cv::Mat &src)
{
    if (restart_time == 0)
    {
        restart_time = this->time;
        this->first_see = 1;
    }
    ts.message.armor_flag = 0;
    if (armors_msg.armors.size() == 0)
    {
        last_ts = ts;
        ts.message.status = 0;
        if (this->shootKeeper > 0)
        {
            shootKeeper--;
            ts.message.armor_flag = 11;
        }
        // printf("keeper:%d\n", shootKeeper);

#ifdef DEBUGMODE
        if (ts.message.armor_flag >= 11)
        {
            cv::putText(src, "FIRE!", cv::Point(720 - 60, 540 - 80), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 255, 0), 3);
        }
#endif
        return;
    }
    if (ts.message.bullet_v < 200.0)
    {
        ts.message.bullet_v = 233.0;
    }
    else if (ts.message.bullet_v > 300.0)
    {
        ts.message.bullet_v = 270.0;
    }
    // ts.message.bullet_v = 270.0;
    // double time_add{0};
    if (!this->updateTracker(ts, src))
        return;
    std::cout << y_camera_reserve.size();
    y_camera_reserve.erase(std::remove_if(y_camera_reserve.begin(), y_camera_reserve.end(), [this](std::pair<cv::Point3f, int> p)
                                          { return p.second != tracking_numb; }),
                           y_camera_reserve.end());
    std::cout << y_camera_reserve.size();
    std::cout << tracking_numb << std::endl;
    auto p_ = findTarget(ts, time_add, src);
    storeMessage(p_, ts);

#ifdef SENDCAMERA
    this->convertPoint(ts, p_, 1, src);
    this->fireControl(ts, ts, src);
#ifndef DEBUGMODE
    printf("x:%.3lf|y:%.3lf|z:%.3lf\n", ts.message.x_a, ts.message.y_a, ts.message.z_a);
    printf("R:%.3lf|vyaw:%.3lf|Num:%d\n", this->r * (this->isClockwise > 0 ? 1.05 : 0.95), rawOmega, tracking_numb);
    printf("yaw:%.4f|pitch:%.4f|flag:%d\n", ts.message.yaw, ts.message.pitch, ts.message.armor_flag);
    printf("bcO:%.3lf|bcV:%.3lf\n", ts.message.vz_c, ts.message.vy_c);
#endif
    last_ts = ts;
    if (this->shootKeeper > 0)
    {
        shootKeeper--;
        ts.message.armor_flag = 11;
    }
    // printf("keeper:%d\n", shootKeeper);
#ifdef DEBUGMODE
    if (ts.message.armor_flag >= 11)
    {
        cv::putText(src, "FIRE!", cv::Point(720 - 60, 540 - 80), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 255, 0), 3);
    }
#endif
    if (tracker_0->tracker_state == Tracker::LOST or tracker_0->tracker_state == Tracker::TEMP_LOST)
    {
        ts.message.armor_flag = 0;
    }

    this->closest2Armors.clear();
    y_camera_reserve.clear();
    this->isBigArmor.clear();

#endif // SENDCAMERA
}

void AimAuto::put_state(Translator &ts, cv::Mat &src)
{
    cv::Point init(0, 400);
    cv::Point line(0, 50);
    std::vector<double> show_tmp;
}

std::unique_ptr<rm_auto_aim::Detector> initDetector(int color)
{
    address addr;
    // int binary_thres = binary_threshold;
    int detect_color = color;
    double min_ratio,
        max_ratio,
        max_angle_l,
        min_light_ratio,
        min_small_center_distance,
        max_small_center_distance,
        min_large_center_distance,
        max_large_center_distance,
        max_angle_a,
        threshold;
    cv::FileStorage fs;
    fs.open(addr.yaml_address + "detect.yaml", cv::FileStorage::READ);
    fs["min_ratio"] >> min_ratio;
    fs["max_ratio"] >> max_ratio;
    fs["max_angle_l"] >> max_angle_l;
    fs["min_light_ratio"] >> min_light_ratio;
    fs["min_small_center_distance"] >> min_small_center_distance;
    fs["max_small_center_distance"] >> max_small_center_distance;
    fs["min_large_center_distance"] >> min_large_center_distance;
    fs["max_large_center_distance"] >> max_large_center_distance;
    fs["max_angle_a"] >> max_angle_a;
    fs["threshold"] >> threshold;
    rm_auto_aim::Detector::LightParams l_params = {
        .min_ratio = min_ratio,
        .max_ratio = max_ratio,
        .max_angle = max_angle_l};

    rm_auto_aim::Detector::ArmorParams a_params = {
        .min_light_ratio = 0.7,
        .min_small_center_distance = min_small_center_distance,
        .max_small_center_distance = max_small_center_distance,
        .min_large_center_distance = min_large_center_distance,
        .max_large_center_distance = max_large_center_distance,
        .max_angle = max_angle_a};

    auto detector = std::make_unique<rm_auto_aim::Detector>(binary_threshold, detect_color, l_params, a_params);

    // Init classifier
    const std::string root_path = std::filesystem::current_path();
    auto model_path = root_path + "/model/mlp.onnx";
    auto label_path = root_path + "/model/label.txt";
    std::vector<std::string> ignore_classes =
        std::vector<std::string>{"negative"};
    detector->classifier =
        std::make_unique<rm_auto_aim::NumberClassifier>(model_path, label_path, threshold, ignore_classes);

    return detector;
}

cv::Point3f AimAuto::findTarget(Translator &ts,
                                double &time_add, cv::Mat &src)
{
    double flight_time = 0.0, react_time = 0.02;
    // 2023-10-16 csy and lxw
    std::vector<cv::Point3f> armors;
    std::vector<Eigen::Vector3d> armors_cam;
    double real_time = flight_time = 0;
    if (ts.message.status % 5 != 2)
    {
        int times_ = 0;
        cv::Point3f temp_armor;

        // ======calculate speed level===== //
        this->rawOmega = tracker_0->target_state(7);
        this->isClockwise = rawOmega < 0.0 ? 1.0 : -1.0;

        double quater_time =
            (3.14159 / 2) / ((tracker_0->target_state(7)));

        double thresh1_2 = 5.0, thresh0_1 = 1.2, bias = 0.4;
        if (abs(rawOmega) > thresh1_2 + bias /* rad/s */)
        {
            rotateLevelCount.push_back(2);
            react_time = react_time_[2];
        }
        else if (abs(rawOmega) > thresh0_1 + bias and abs(rawOmega) < thresh1_2 - bias /* rad/s */)
        {
            rotateLevelCount.push_back(1);
            react_time = react_time_[1];
        }
        else if (abs(rawOmega) < thresh0_1 - bias)
        {
            rotateLevelCount.push_back(0);
            react_time = react_time_[0];
        }
        else
        {
            rotateLevelCount.push_back(rotateLevelCount.back());
        }
        int m_size = 60;
        if (rotateLevelCount.size() == m_size)
        {
            int sum_1 = 0;
            int sum_2 = 0;
            for (auto &i : rotateLevelCount)
            {
                sum_1 += i == 1;
                sum_2 += i == 2;
            }
            if (sum_2 > m_size * 0.67)
            {
                this->rotate_speed_level = 2;
                double appr = std::accumulate(filteredVyawArray.begin(), filteredVyawArray.end(), 0.0) / filteredVyawArray.size();
                // if (abs(rawOmega) > 12.0 and abs(rawOmega / appr) > 2.0)
                // {
                //     filteredVyawArray.push_back(appr);
                //     tracker_0->target_state(7) = appr;
                // }
                // else
                filteredVyawArray.push_back(rawOmega);
                tracker_0->target_state(7) = appr;
                if (filteredVyawArray.size() > 5)
                    filteredVyawArray.pop_front();
            }
            else if (sum_1 > m_size * 0.67)
            {
                this->rotate_speed_level = 1;
                filteredVyawArray.clear();
            }
            else if (sum_1 + sum_2 < m_size / 4)
            {
                this->rotate_speed_level = 0;
                filteredVyawArray.clear();
            }
            rotateLevelCount.pop_front();
            this->level_count[0] = sum_1;
            this->level_count[1] = sum_2;
        }
        double v_sum =
            std::sqrt((tracker_0->target_state(1)) * (tracker_0->target_state(1)) +
                      (tracker_0->target_state(3)) * (tracker_0->target_state(3)));

        if (v_sum > 0.3 /* m/s */)
        {
            this->transLevelCount.push_back(1);
        }
        else if (v_sum < 0.25)
        {
            this->transLevelCount.push_back(0);
        }
        else
        {
            transLevelCount.push_back(transLevelCount.back());
        }
        if (transLevelCount.size() == m_size)
        {
            int sum_1 = 0;
            for (auto &i : transLevelCount)
            {
                sum_1 += i == 1;
            }
            if (sum_1 > m_size * 0.8)
            {
                this->translational_speed_level = 1;
            }
            else
            {
                this->translational_speed_level = 0;
            }
            transLevelCount.pop_front();
            this->level_count[2] = sum_1;
        }
        // 2024-1-21 lxw

        if (v_sum < 3e-2)
        {
            this->v_zoom = 0;
        }
        else
        {
            this->v_zoom = V_ZOOM;
        }
        this->vyaw_zoom = 1.0;
#ifdef DEBUGMODE
        std::ostringstream indice_info;
#endif
        // rotate_speed_level = 2;
        double yaw_forward = 0.0, x_c_forward = 0, y_c_forward = 0, pred_yaw = 0;
        std::vector<int> indices = {0, 1, 2, 3};
        std::vector<Eigen::Vector3d> armors_camera(0);
        // Eigen::Vector3d center__;
        cv::Point3f center_;
        // react_time += time_add / 1000;
        do
        {
            armors.clear();
            armors_cam.clear();
            real_time = flight_time + react_time;
            // double x_c = tracker_1->target_state(0);
            // double y_c = tracker_1->target_state(2);
            double x_c = tracker_0->target_state(0);
            double y_c = tracker_0->target_state(2);
            double pred_x_c = 0.0, pred_y_c = 0.0;
            // do not use real_time here
            // x_c_forward = tracker_1->target_state(1) * flight_time * this->v_zoom;
            // y_c_forward = tracker_1->target_state(3) * flight_time * this->v_zoom;
            if (rotate_speed_level != 2)
            {
                x_c_forward = tracker_0->target_state(1) * real_time * this->v_zoom;
                y_c_forward = tracker_0->target_state(3) * real_time * this->v_zoom;
            }
            else
            {
                x_c_forward = tracker_0->target_state(1) * flight_time * this->v_zoom;
                y_c_forward = tracker_0->target_state(3) * flight_time * this->v_zoom;
            }
            pred_x_c = x_c + x_c_forward;
            pred_y_c = y_c + y_c_forward;

            if (abs(tracker_0->target_state(7)) < 0.2)
                tracker_0->target_state(7) = 1e-3;

            yaw_forward = tracker_0->target_state(7) * real_time * this->vyaw_zoom;
            pred_yaw = tracker_0->target_state(6) + yaw_forward;
            double rate = 1.0;
            double pred_x_a = pred_x_c - this->r * cos(pred_yaw) * rate;
            double pred_y_a = pred_y_c - this->r * sin(pred_yaw) * rate;

            cv::Point3f
                p1(pred_x_a, pred_y_a, tracker_0->target_state(4)),
                p2(pred_x_c + pred_y_c - pred_y_a, pred_y_c - pred_x_c + pred_x_a, tracker_0->target_state(4) + tracker_0->dz),
                p3(2 * pred_x_c - pred_x_a, 2 * pred_y_c - pred_y_a, tracker_0->target_state(4)),
                p4(pred_x_c - pred_y_c + pred_y_a, pred_y_c + pred_x_c - pred_x_a, tracker_0->target_state(4) + tracker_0->dz);
            armors.push_back(p1);
            armors.push_back(p2);
            armors.push_back(p3);
            armors.push_back(p4);

            // std::vector<Eigen::Vector3d> armors_cam
            armors_cam.push_back(world2camera(p1, ts, 0));
            armors_cam.push_back(world2camera(p2, ts, 0));
            armors_cam.push_back(world2camera(p3, ts, 0));
            armors_cam.push_back(world2camera(p4, ts, 0));

            center_ = cv::Point3f(pred_x_c, pred_y_c, tracker_0->target_state(4));
            predict_center = world2camera(center_, ts, 0);
            std::sort(indices.begin(), indices.end(), [&armors](int a, int b)
                      { return sqrt(armors[a].x * armors[a].x + armors[a].y * armors[a].y) < sqrt(armors[b].x * armors[b].x + armors[b].y * armors[b].y); });
            temp_armor = armors[(indices[0] + position_gap + 4) % 4];
            if (y_camera_reserve.size() == 2)
            {
                std::map<std::pair<int, int>, int> ycr_map = {
                    {{3, 0}, 0},
                    {{0, 3}, 1},
                    {{1, 0}, 1},
                    {{2, 3}, this->isClockwise > 0 ? 0 : 1},
                    {{3, 2}, this->isClockwise > 0 ? 1 : 0},
                    {{0, 1}, 0}};
                choose_ycr = ycr_map[{indices[0], indices[1]}];
            }
            else
            {
                choose_ycr = (indices[0] == 3 || indices[0] == 1) ? -1 : 0;
            }
#ifdef DEBUGMODE
            indice_info << indices[0] << indices[1] << indices[2] << indices[3] << " ";
#endif
            flight_time = timer.getFlightTime(temp_armor, (double)ts.message.bullet_v / 10) / DIM_ERROR_DEEP;
            if (times_++ > 5)
                break;

        } while (abs(flight_time + react_time - real_time) > 1e-5);
        // if (this->closest2Armors.empty())
        // {
        //     closest2Armors.push_back(armors[indices[0]]);
        //     closest2Armors.push_back(armors[indices[1]]);
        // }
        // if (abs(armors_cam[indices[0]].x() - center__.x()) > abs(armors_cam[indices[1]].x() - center__.x()) and rotate_speed_level == 1)
        // {
        //     std::swap(indices[0], indices[1]);
        //     std::cout << "swap_armors\n";
        // }
        // cv::circle(src, camera2pixel(armors_cam[indices[1]]), 10, cv::Scalar(0, 0, 255), -1);
        // if (abs(armors_cam[indices[0]].x()) >= abs(armors_cam[indices[1]].x()) and rotate_speed_level == 0 and y_camera_reserve.size() == 2)
        // {
        //     std::swap(indices[0], indices[1]);
        //     std::cout << "swap_armors\n";
        // }
#ifdef DEBUGMODE
        std::ostringstream temp0, temp1;
        temp0 << "yaw_forward:" << std::to_string(yaw_forward);
        // temp1 << std::to_string(tracker->target_state(6));
        cv::putText(src, temp0.str(), cv::Point(800, 450), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 125, 220), 2);
        // cv::putText(src, "yawModel:" + temp1.str(), cv::Point(800, 350), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 125, 220), 2);
        cv::putText(src, "indiceInfo:" + indice_info.str(), cv::Point(800, 400), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 125, 220), 2);
#endif
#ifdef POSESTIMATION
        predictPoints.push_back(predictPoint(armors[indices[0]], this->receive_second + real_time, pred_yaw));
        if (predictPoints.size() > 50)
        {
            predictPoints.pop_front();
        }
#endif
        printf("real_time:%.4lf | flight_time:%.4lf | react_time:%.4lf\n", real_time, flight_time, react_time);
        this->current_world_state = center_;
        return armors[indices[0]];
    }
    else
    {
        return cv::Point3f(1e-3, 1e-3, 1e-3);
    }
}

void AimAuto::storeMessage(cv::Point3f target, Translator &ts)
{
    ts.message.x_a = target.x * 1000;
    ts.message.y_a = target.y * 1000;
    ts.message.z_a = target.z * 1000;
    // ts.message.x_c = tracker_1->target_state(0) * 1000;
    // ts.message.y_c = tracker_1->target_state(2) * 1000;
    // ts.message.z_c = tracker_1->target_state(4) * 1000;
    ts.message.x_c = tracker_0->target_state(0) * 1000;
    ts.message.y_c = tracker_0->target_state(2) * 1000;
    ts.message.z_c = tracker_0->target_state(4) * 1000;
    ts.message.yaw_a = tracker_0->target_state(6);
    // ts.message.vx_c = tracker_1->target_state(1) * 1000;
    // ts.message.vy_c = tracker_1->target_state(3) * 1000;
    // ts.message.vz_c = tracker_1->target_state(8) * 1000;
    ts.message.vx_c = tracker_0->target_state(1) * 1000;
    ts.message.vy_c = tracker_0->target_state(3) * 1000;
    ts.message.vz_c = tracker_0->target_state(8) * 1000;
    ts.message.vyaw_a = tracker_0->target_state(7);
    // ts.message.armor_flag = tracker->tracked_armor.type;
    ts.message.yaw = this->raw_yaw;
}

void AimAuto::convertPoint(Translator &ts, cv::Point3f target, bool ycr, cv::Mat &src)
{
    if (!this->is_hitting_outpose)
    {
        Eigen::MatrixXd m_pitch(3, 3);
        Eigen::MatrixXd m_yaw(3, 3);
        auto real_yaw = ts.message.yaw + gp->camera2shootBias;
        m_yaw << cos(real_yaw), -sin(real_yaw), 0, sin(real_yaw), cos(real_yaw), 0, 0, 0, 1;
        m_pitch << cos(ts.message.pitch), 0, sin(ts.message.pitch), 0, 1, 0, -sin(ts.message.pitch), 0, cos(ts.message.pitch);
        Eigen::Vector3d temp;
        temp = Eigen::Vector3d(target.x, target.y, target.z);
        Eigen::Vector3d position = m_pitch.inverse() * m_yaw.inverse() * temp;
        ts.message.x_a = -position(1) * 1000;
        ts.message.y_a = -(position(2) * 1000);
        ts.message.z_a = position(0) * 1000;
        ts.message.vyaw_a *= -1;
        ts.message.yaw_a *= -1;
        if (this->choose_ycr != -1 and ycr)
        {
            ts.message.y_a = this->y_camera_reserve[this->choose_ycr % 2].first.y;
#ifdef DEBUGMODE

            cv::putText(src, "YCR", cv::Point(500, 250), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 125, 220), 2);
#endif

            // std::cout << "y_overrided\n"
            //   << std::endl;
        }
    }
    else
    {
        ts.message.x_a = position_save.x;
        ts.message.y_a = position_save.y;
        ts.message.z_a = position_save.z;
        this->rawYawArray.push_back(tar_list[0].yaw > last_yaw);
        if (this->rawYawArray.size() > 50)
            this->rawYawArray.pop_front();
        double aqq = std::accumulate(rawYawArray.begin(), rawYawArray.end(), 0.0);
        this->isClockwise = aqq > 12.0 ? 1 : -1;
        printf("aqq:%f\n", aqq);
        double pzx = 0;
        if (this->isClockwise > 0)
        {
            pzx = abs(ts.message.z_a) / 30000 * 500;
        }
        else if (abs(this->isClockwise) < 1e-5)
        {
            pzx = abs(ts.message.z_a) / 30000 * 0;
        }
        else
        {
            pzx = abs(ts.message.z_a) / 30000 * -500;
        }
        ts.message.x_a += pzx;
    }
}

void AimAuto::fireControl(Translator &ts0, Translator &ts1, cv::Mat &src)
{
    // ====================显示1==================== //

    cv::Scalar color_tmp /*green*/ (0, 255, 0);
    std::vector<double> center_p = {ts0.message.x_c, ts0.message.y_c, ts0.message.z_c};
    Eigen::Vector3d center_in_camera = this->world2camera(center_p, ts0);

    double temp_x, temp_y, temp_z;
    temp_x = center_in_camera(0);
    temp_y = center_in_camera(1);
    temp_z = center_in_camera(2);

#ifdef DEBUGMODE
    Eigen::Vector3d closest_predict_in_camera = Eigen::Vector3d(ts0.message.x_a, ts0.message.y_a + 1000 * VECTOR_Z, ts0.message.z_a);
    Eigen::Vector3d secondary_predict_in_camera = Eigen::Vector3d(ts1.message.x_a, ts1.message.y_a + 1000 * VECTOR_Z, ts1.message.z_a);
    cv::Point closest_predict_plot = camera2pixel(closest_predict_in_camera);
    cv::Point secondary_predict_plot = camera2pixel(secondary_predict_in_camera);
    cv::Point center_plot = camera2pixel(center_in_camera);
    cv::Point predict_center_plot = camera2pixel(predict_center);

    center_plot.y = predict_center_plot.y = 400;
    // int pixel_gap = closest_predict_plot.x - center_plot.x;
    // 水平，高度，深度
    cv::Point center(0, 800);
    cv::Point gap(0, 50);
#ifdef POSESTIMATION
    posteriorEstimation(ts0, src, closest_predict_plot.y);
#endif
#endif
    double y_a_appr(0), v_sum(0), threshold_X_a(0);
    if (!this->is_hitting_outpose)
    {
        double translational_high_speed = 400.0;
        Translator ts_temp = ts0;

        if (rotate_speed_level == 2)
        {
            //             if (abs(ts0.message.x_a) > abs(ts1.message.x_a) and abs(ts1.message.x_a - temp_x) < 750 * this->r)
            //             {
            //                 std::swap(ts0, ts1);
            // #ifdef DEBUGMODE
            //                 cv::putText(src, "TsSwap", cv::Point(500, 250), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 125, 220), 2);
            // #endif
            //             }
            yReserveArray.push_back(ts0.message.y_a);
            if (yReserveArray.size() > 20)
            {
                yReserveArray.pop_front();
            }
            y_a_appr = std::accumulate(yReserveArray.begin(), yReserveArray.end(), 0.0) / yReserveArray.size();
            ts0.message.y_a = 0.2 * y_a_appr + 0.8 * ts0.message.y_a;
        }
        // clang-format off
    std::vector<std::vector<double>> threhold_table = {
                    /*trans:0 trans:1*/
        /*rotate:0*/ {33.0, 30.0},
        /*rotate:1*/ {25.0, 30.0},
        /*rotate:2*/ {20.0, 20.0}};
        // clang-format on
        threshold_X_a = threhold_table[rotate_speed_level][translational_speed_level];
        v_sum /* mm/s */ = std::sqrt((ts0.message.vx_c) * (ts0.message.vx_c) + (ts0.message.vy_c) * (ts0.message.vy_c));
        if (v_sum > translational_high_speed)
        {
            threshold_X_a += 0.008 * (v_sum - translational_high_speed);
        }

        threshold_X_a *= this->isBigArmor[0] ? 1.5 : 1.0;

        // int numb_save = ts0.message.armor_flag;
        // double tmp_d = ts.message.x_a * this->isClockwise;
        if (rotate_speed_level != 2)
        {
            if (this->isClockwise > 0)
            {
                // clockwise,have larger thresh in right side;
                if ((ts0.message.x_a < threshold_X_a and ts0.message.x_a > -threshold_X_a / 2.5))
                {
                    ts0.message.armor_flag = 114;
                }
            }
            else
            { // not clockwise,have larger thresh in left side;
                if ((ts0.message.x_a < threshold_X_a / 2.5 and ts0.message.x_a > -threshold_X_a))
                {
                    ts0.message.armor_flag = 114;
                }
            }
            fireTimes = 0;
        }
        else if (abs(ts0.message.x_a) < threshold_X_a * 0.75)
        {
            // if (this->time - time_save > 80)
            {
                ts0.message.armor_flag = 114;
                time_save = time;
            }
            fireTimes = 0;
        }
        else
        {
            // fireTimes = 0;
        }

        printf("times:%d\n", fireTimes);
        // high:armor_flag=11,count=1
        // mid :armor_flag=12,count=3
        // slow:armor_flag=13,count=5
        bool permmit = 0;
        double shootCenterL = 0, shootCenterR = 0;
        if (rotate_speed_level == 2)
        {
            double d_ = ts0.message.x_a - temp_x;
            int max_size = 30;
            if (d_ > 0)
            {
                this->shootCenterRightArray.push_back(d_);
                if (shootCenterRightArray.size() >= max_size)
                {
                    shootCenterRightArray.pop_front();
                    permmit = 1;
                }
            }
            else
            {
                this->shootCenterLeftArray.push_back(d_);
                if (shootCenterLeftArray.size() >= max_size)
                {
                    shootCenterLeftArray.pop_front();
                    permmit = 1;
                }
            }
            if (permmit)
            {
                shootCenterL = std::accumulate(shootCenterLeftArray.begin(), shootCenterLeftArray.end(), 0.0) / max_size;
                shootCenterR = std::accumulate(shootCenterRightArray.begin(), shootCenterRightArray.end(), 0.0) / max_size;
                double shootCenter = temp_x + ((this->isClockwise > 0) ? shootCenterL : shootCenterR);
                ts0.message.x_a = shootCenter;
                // if (fireTimes >= 1)
                // {
                // ts0.message.x_a += rawOmega * 10;
                // #ifdef DEBUGMODE
                //                     cv::putText(src, "makebias", cv::Point(500, 250), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 125, 220), 2);
                // #endif
                // if (fireTimes < 3)
                // {
                //     ts0.message.armor_flag = 114;
                // }
                // fireTimes--;
                // }
            }
            // printf("xa:%.4lf,times:%d\n", ts0.message.x_a, fireTimes);
            // if (abs(ts0.message.x_a - shootCenter) > 20.0)
            // {
            //     ts0.message.x_a = shootCenter + this->isClockwise * 30.0;
            // }
            // else
            // {
            //     ts0.message.armor_flag = 114;
            // }

            // if (ts0.message.x_a * this->isClockwise > 0)
            // {
            //     // ts0.message.x_a = 1e-5;
            // }
        }
        else
        {
            shootCenterLeftArray.clear();
            shootCenterRightArray.clear();
        }
        if (v_sum > 300.0 and rotate_speed_level == 2)
        {
            ts0.message.x_a = temp_x;
        }
        if (ts0.message.armor_flag == 114)
        {
            // clang-format off
        std::vector<std::vector<int>> tablet = {
                    /*trans:0 trans:1*/
        /*rotate:0*/ {11,12},
        /*rotate:1*/ {12, 12},
        /*rotate:2*/ {12, 12}};
            // clang-format on
            ts0.message.armor_flag = tablet[this->rotate_speed_level][this->translational_speed_level];
        }
        // std::cout << "rotate speed level:" << rotate_speed_level << std::endl;
        float ratio = 1.0;

        if (this->rotate_speed_level == 1)
        {
            if ((ts0.message.x_a - temp_x) > this->r * 1e3 * 0.8)
            {
                ts0.message.x_a = temp_x;
            }
            else if (ts0.message.x_a - temp_x < -this->r * 1e3 * 0.8)
            {
                ts0.message.x_a = temp_x;
            }
        }
        ts0.message.vy_c = 1 * this->updateArmorSpeed(src, ts0);
        if (rotate_speed_level != 2)
        {
            auto bonus = ts0.message.x_a * ts0.message.vy_c;
            auto rate = sqrt(abs(ts0.message.x_a) / 5.0);
            if (rate < 1)
                rate = 1;
            if (bonus < 0) // do not follow
            {
                ts0.message.vy_c *= rate;
            }
            else // over follow
            {
                ts0.message.vy_c /= rate;
            }
            // ts0.message.vy_c += ts0.message.x_a * 0.5
        }
        ts0.message.vz_c = -rawOmega * this->r * 1.5;
    }
    else
    {
        ts0.message.vy_c = 0;
        ts0.message.vyaw_a = 0.8 * M_PI * this->isClockwise;
        if (abs(ts0.message.x_a) < 25)
            ts0.message.armor_flag = 11;
    }
#ifdef DEBUGMODE
    Eigen::Vector3d send_in_camera = Eigen::Vector3d(ts0.message.x_a, ts0.message.y_a + 1000 * VECTOR_Z, ts0.message.z_a * DIM_ERROR_DEEP);
    cv::Point send_plot = camera2pixel(send_in_camera);
    // send_plot.y = 400;

    cv::circle(src, closest_predict_plot, 10, cv::Scalar(0, 255, 0), 2);
    // cv::circle(src, secondary_predict_plot, 10, cv::Scalar(0, 255, 0), -1);
    cv::circle(src, center_plot, 8, cv::Scalar(88, 59, 255), 2);
    cv::circle(src, predict_center_plot, 6, cv::Scalar(88, 59, 255), -1);
    cv::circle(src, send_plot, 8, cv::Scalar(128, 128, 128), -1);

    // ===============DISPLAY-2=============== //
    std::ostringstream temp1, temp2, temp3, temp4, temp5, temp6, temp7;
    temp1
        << "sp" << std::to_string(ts0.message.x_a)
        << "|cz" << std::to_string(ts0.message.y_a)
        << "|sd" << std::to_string(ts0.message.z_a)
        // << "|y_appr" << std::to_string(y_a_appr)
        << "|centerx" << std::to_string(temp_x);
    temp2
        << "xc" << std::to_string(ts0.message.x_c)
        << "yc" << std::to_string(ts0.message.y_c)
        << "zc" << std::to_string(ts0.message.z_c)
        << "vs" << std::to_string(v_sum)
        << "pitch:" << ts0.message.pitch
        << "yaw:" << ts0.message.yaw;
    temp3
        << "hk-yz:" << std::to_string(threshold_X_a)
        << " yaw_a:" << ts0.message.yaw_a
        << " vyaw_a:" << ts0.message.vyaw_a
        << " slevel" << this->rotate_speed_level
        << " rawOmega:"
        << rawOmega;
    temp4
        << "r_lv:" << (rotate_speed_level)
        << "t_lv:" << (translational_speed_level)
        << "clkw:" << this->isClockwise
        << "rlv_ct:"
        << this->level_count[0] << "|"
        << this->level_count[1] << "|"
        << this->level_count[2];
    temp5
        << " v_a:" << ts0.message.vy_c
        << "|vsum:" << v_sum
        << "|bulv:" << ts0.message.bullet_v
        // << "|r:" << this->r * (this->isClockwise > 0 ? 1.05 : 0.95)
        << "|r:" << this->r;
    ;
    //     << "|ctrlCenter:" << shootCenterBias << "|" << ratio;
    temp6
        << " ycr:" << this->y_camera_reserve.size()
        << " ycr_index:" << this->choose_ycr;
    // if (this->y_camera_reserve.size() >= 2)
    {
        temp6 << " ori_n:";
        for (auto i : y_camera_reserve)
        {
            temp6 << i.second << "|";
        }
    }
    // << " ori_n" << this->ori_y[0].z << this->ori_y[1].z;
    cv::putText(src, temp1.str(), center, cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 125, 220), 2);
    cv::putText(src, temp2.str(), center + gap, cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 125, 220), 2);
    cv::putText(src, temp3.str(), center + 2 * gap, cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 125, 220), 2);
    cv::putText(src, temp4.str(), center + 3 * gap, cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 125, 220), 2);
    cv::putText(src, temp5.str(), center + 4 * gap, cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 125, 220), 2);
    cv::putText(src, temp6.str(), center + 5 * gap, cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 125, 220), 2);
    cv::putText(src, temp7.str(), center + 6 * gap, cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 125, 220), 2);

    int midx = gp->realy_mid;
    int midy = this->src_size.height / 2;
    cv::Point startPoint(midx, 0);
    cv::Point endPoint(midx, 2 * midy);
    cv::line(src, startPoint, endPoint, cv::Scalar(255, 0, 255), 2);                                   //  ?心线
    cv::line(src, cv::Point(midx - 50, midy), cv::Point(midx + 50, midy), cv::Scalar(255, 0, 255), 2); //  ?心线
    if (ts0.message.armor_flag >= 11)
    {
        cv::putText(src, "FIRE!", cv::Point(midx - 60, midy - 80), cv::FONT_HERSHEY_PLAIN, 3, color_tmp, 3);
    }
// #else
//     printf("Omega:%.2lf,R:%.3lf\n", -rawOmega, this->r);
// printf("Omega:%.2lf\n",-rawOmega);
// ===============DISPLAY-2-end=============== //
// cv::putText(src, "PhysicCenter", cv::Point(midx - 50, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 255));
#endif
    // if (!this->is_hitting_outpose)
    {
        if (rotate_speed_level == 2)
        {
            ts0.message.vz_c = 0;
            ts0.message.vy_c = 0;
            ts0.message.vyaw_a = 0;
        }
        else
        {
            // ts0.message.vz_c = 0;
            // ts0.message.vyaw_a *= 0.75;
        }
        ts0.message.y_a -= abs(ts0.message.z_a) * gp->pzy;
    }
}
double AimAuto::cal_ArmorSpeed(cv::Mat &src)
{
    // // TODO;潜在bug，绕后可能 ? 致预测方向相反
    std::chrono::milliseconds this_time_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch());
    double time_ = (double)(this_time_ - this->last_time_).count();
    double theta1, theta2, k1, k2;
    double x1 = current_world_state.x, y1 = current_world_state.y;
    double x2 = last_world_state.x, y2 = last_world_state.y;
    this->last_time_ = this_time_;
    double alpha, a, b, c, temp;

    a = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    b = sqrt(x1 * x1 + y1 * y1);
    c = sqrt(x2 * x2 + y2 * y2);

    // printf("%.4lf %.4lf %.4lf\n", a, b, c);
    temp = (b * b + c * c - a * a) / (2 * b * c);

    if (temp > 1)
    {
        temp = 1;
    }
    else if (temp < -1)
    {
        temp = -1;
    }
    alpha = std::acos(temp);
    // #ifdef DEBUGMODE
    //     std::ostringstream temp_;
    //     temp_ << x1 << "|" << y1 << " " << x2 << "|" << y2;
    //     cv::putText(src, temp_.str(), cv::Point(300, 60),
    //                 cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 100, 0));
    // #endif
    last_world_state = current_world_state;
    if (abs(x1) < 1e-2 and abs(x2) < 1e-2)
    {
        // printf("x1 x2 small\n");
        return 1e-3;
    }
    else if (abs(x1) < 1e-2)
    { // x1 is small
        // printf("x1 is small\n");
        k2 = y2 / x2;
        theta1 = M_PI_2, theta2 = atan(k2);
        alpha *= (theta1 - theta2) > 0 ? 1.0 : -1.0;
        return 1e5 * alpha / time_;
    }
    else if (abs(x2) < 1e-2)
    { // x2 is small
        // printf("x2 is small\n");
        k1 = y1 / x1;
        theta2 = M_PI_2, theta1 = atan(k1);
        alpha *= (theta1 - theta2) > 0 ? 1.0 : -1.0;
        return 1e5 * alpha / time_;
    }
    else
    {
        // printf("normal\n");
        k1 = y1 / x1, k2 = y2 / x2;
        theta1 = atan(k1), theta2 = atan(k2);
        alpha *= (theta1 - theta2) > 0 ? 1.0 : -1.0;
        return 1e5 * alpha / time_;
    }
}
int AimAuto::autoZoneChoose(cv::Mat &src, std::vector<double> &zone_, float value, double center)
{
    // this->PredPos_array.push_back(value - center);
    double d = value - center;
    int max_size = 60;
    if (PredPosIndex_array.size() != zone_.size() - 1)
    {
        PredPosIndex_array.clear();
        PredPosIndex_array.resize(zone_.size() - 1);
        for (auto i : PredPosIndex_array)
        {
            i = 0;
        }
        // init
    }
    for (int i = 0; i < zone_.size() - 1; i++)
    {
        if (d > zone_[i] and d < zone_[i + 1])
        {
            this->PredPos_array.push_back(i);
            break;
        }
    }
    if (PredPos_array.size() == max_size)
    {
        for (auto &i : PredPos_array)
        {
            for (int j = 0; j < zone_.size() - 1; j++)
            {
                if (i == j)
                {
                    PredPosIndex_array[j]++;
                }
            }
        }
        PredPos_array.pop_front();
    }
#ifdef DEBUGMODE
    std::ostringstream temp_zone;
#endif
    for (auto i : this->PredPosIndex_array)
    {
        // std::cout << i << "|";
#ifdef DEBUGMODE
        temp_zone << i << "|";
#endif
    }
    int max_index = 0;
    for (int i = 0; i < zone_.size() - 1; i++)
    {
        if (PredPosIndex_array[i] > 1.25 * PredPosIndex_array[max_index])
        {
            max_index = i;
        }
    }
    PredPosIndex_array.clear();
    return max_index;
}

double AimAuto::updateArmorSpeed(cv::Mat &src, Translator &ts)
{
    this->ArmorSpeed_array.push_back(cal_ArmorSpeed(src));
    int max_size = 20;
    if (this->ArmorSpeed_array.size() > max_size)
    {
        this->ArmorSpeed_array.pop_front();
    }

    double appr = 0;
    std::vector<double> value = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    for (int i = 0; i < max_size; i++)
    {
        appr += 0.05 * this->ArmorSpeed_array[i];
    }
    std::vector<double> vec(ArmorSpeed_array.begin(), ArmorSpeed_array.end());
    std::sort(vec.begin(), vec.end());

    double min_value = vec.front();
    double max_value = vec.back();
    double thresh = 2.0;
    if (abs(ts.message.vyaw_a) > thresh)
    {
        appr *= pow(thresh, 4);
        appr /= pow(appr, 4);
    }
    bool wrap = min_value - appr < 0 and
                max_value - appr > 0 and
                (max_value - min_value) > 50.0;
    // std::cout << "appr " << appr << std::endl;
    if (abs(appr) < 16.5f or
        abs(ts.message.z_a) < 1500.0 or
        rotate_speed_level == 1 or
        rotate_speed_level == 2 or
        isnan(appr)
        // or wrap
    )
    {
        appr = 0.0;
    }
    return appr;
}

cv::Point AimAuto::camera2pixel(Eigen::Vector3d &point)
{
    Eigen::Vector3d temp = this->_K_ * point;
    int PictureX = (int)(temp[0] / temp[2]);
    int PictureY = (int)(temp[1] / temp[2]);
    return cv::Point(PictureX, /*this->src_size.height -*/ PictureY);
}

void AimAuto::camera2plot(Eigen::Vector3d &point, cv::Mat &src)
{
    cv::Scalar color(255, 153, 0);
#ifdef DEBUGMODE
    cv::circle(src, this->camera2pixel(point), 3, color, 2);
#endif
}

Eigen::Vector3d AimAuto::world2camera(std::vector<double> &input, Translator &ts)
{
    if (input.size() != 3)
    {
        return Eigen::Vector3d(0, 0, 0);
    }
    else
    {
        Eigen::MatrixXd m_pitch(3, 3);
        Eigen::MatrixXd m_yaw(3, 3);
        m_yaw << cos(ts.message.yaw), -sin(ts.message.yaw), 0, sin(ts.message.yaw), cos(ts.message.yaw), 0, 0, 0, 1;
        m_pitch << cos(ts.message.pitch), 0, sin(ts.message.pitch), 0, 1, 0, -sin(ts.message.pitch), 0, cos(ts.message.pitch);
        Eigen::Vector3d temp_position;
        temp_position = Eigen::Vector3d(input[0], input[1], input[2]);
        Eigen::Vector3d position = m_pitch.inverse() * m_yaw.inverse() * temp_position;
        input = {-position(1), position(2), position(0)};
        return Eigen::Vector3d(-position(1), position(2), position(0));
    }
}

Eigen::Vector3d AimAuto::world2camera(cv::Point3f input, Translator &ts, bool to_shoot)
{

    Eigen::MatrixXd m_pitch(3, 3);
    Eigen::MatrixXd m_yaw(3, 3);
    auto real_yaw = ts.message.yaw;
    if (to_shoot)
    {
        real_yaw += gp->camera2shootBias / 180.0 * M_PI;
    }
    m_yaw << cos(real_yaw), -sin(real_yaw), 0, sin(real_yaw), cos(real_yaw), 0, 0, 0, 1;
    m_pitch << cos(ts.message.pitch), 0, sin(ts.message.pitch), 0, 1, 0, -sin(ts.message.pitch), 0, cos(ts.message.pitch);
    m_pitch << cos(ts.message.pitch), 0, sin(ts.message.pitch), 0, 1, 0, -sin(ts.message.pitch), 0, cos(ts.message.pitch);
    Eigen::Vector3d temp_position;
    temp_position = Eigen::Vector3d(input.x, input.y, input.z);
    Eigen::Vector3d position = m_pitch.inverse() * m_yaw.inverse() * temp_position;
    return Eigen::Vector3d(-position(1), -position(2), position(0));
}

void AimAuto::showDist(std::deque<Armor> &tar_list, cv::Mat &src)
{
    if (tar_list.size() != 0)
    {
        cv::putText(src, "X:" + std::to_string(tar_list[0].center.x), cv::Point(150, 60),
                    cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 255, 0));
        cv::putText(src, "Y:" + std::to_string(tar_list[0].center.y), cv::Point(150, 90),
                    cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 255, 0));
        cv::putText(src, "Z:" + std::to_string(tar_list[0].center.z), cv::Point(150, 120),
                    cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 255, 0));
        cv::putText(src, "type:" + std::to_string(tar_list[0].type), cv::Point(150, 150),
                    cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 255, 0));
    }
    else
    {
        cv::putText(src, "X: NOT FOUND", cv::Point(150, 60),
                    cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 255, 0));
        cv::putText(src, "Y: NOT FOUND", cv::Point(150, 90),
                    cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 255, 0));
        cv::putText(src, "Z: NOT FOUND", cv::Point(150, 120),
                    cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 255, 0));
    }
}
bool AimAuto::updateTracker(Translator &ts, cv::Mat &src)
{
    std::string t = "";
    if (tracker_0->tracker_state == Tracker::LOST)
    {
        t = "LOST";
    }
    else if (tracker_0->tracker_state != Tracker::TEMP_LOST)
    {
        t = "TRAC";
    }
    cv::putText(src, t, cv::Point(50, 200), 1, 2, cv::Scalar(0, 255, 0));
    if (!armors_msg.armors.empty())
    {
        if (tracker_0->tracker_state == Tracker::LOST)
        {
            dt = 0.01;
            tracker_0->init(armors_msg);
            tracker_1->init(armors_msg);
        }
        else
        {
            dt = this->time - this->last_time;
            dt /= 1000;
            int id = 0;
            tracking_numb = tar_list[0].type;
            tracker_0->update(armors_msg, src, tracking_numb);
            tracker_1->update(armors_msg, src, tracking_numb);
            last_time = this->time;
        }
        // std::cout << tracker_0->target_state << std::endl;
        this->r = tracker_0->target_state(8);
        return true;
    }
    else
    {
        if (tracker_0->tracker_state == Tracker::LOST)
        {
            return false;
        }
        else
        {
            time_add = this->time - this->last_time;
            return true;
        }
    }
}

void AimAuto::setTime(double t)
{
    this->receive_second = t;
}

void AimAuto::posteriorEstimation(Translator &ts, cv::Mat &src, float y)
{
    // find the closest item

    if (!predictPoints.empty())
    {
        printf("tnow:%.4f,t_f:%.4f,t_e:%.4f\n", this->receive_second, predictPoints.front().time_second, predictPoints.back().time_second);
        std::sort(predictPoints.begin(), predictPoints.end(), [&](predictPoint &a, predictPoint &b)
                  { return (a.time_second - this->receive_second) < (b.time_second - this->receive_second); });
        for (auto i : predictPoints)
        {
            double t = i.time_second - this->receive_second;
            if (t < 0)
            {
                continue;
            }
            else
            {
                printf("time_gap:%.4f\n", i.time_second - this->receive_second);
                auto temp_point = world2camera(i.point, ts, 0);
                cv::circle(src, cv::Point(camera2pixel(temp_point).x, y), 20, cv::Scalar(255, 255, 0), 2);
                return;
            }
        }
        printf("No-Maching-Points,with-time-gap::%.4f", predictPoints.back().time_second - this->receive_second);
    }
}