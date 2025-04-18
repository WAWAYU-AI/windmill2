/*
 * @Author: zxh 1608278840@qq.com
 * @Date: 2023-11-08 03:11:49
 * @FilePath: /DX_aimbot/windmill/src/WMPredict.cpp
 * @Description:能量机关预测
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved.
 */

#include "WMPredict.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <utility>
// #include "SerialPort.hpp"
#include "WMIdentify.hpp"
#include "globalParam.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/types_c.h"
#include <angular_velocity_fitter.hpp>
#include <ceres/ceres.h>
#include <ceres/loss_function.h>
#include <complex>
#include <deque>
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <ostream>
#include <type_traits>
#include <unistd.h>
#include <vector>

//=====常量=====//
static double g = 9.8;        // 重力加速度
static double k = 0.05;       // 空气阻力系数  //考虑了子弹质量？
static double r = 0.7;        // 符的半径
static double s = 999999.3;   // 车距离符的水平距离
static double Exp = 2.71823;  // 自然常数e
static double h0 = 1.1747;    // 符的中心点高度减去车的高度
static double pi = 3.14159;   // 圆周率
static double delta_t = 0.10; // 超前滞后
static double diff_w = 0.01;
static double w_low = 1.5;
static double w_up = 2.5;
static double yaw_fix = 0.07; // 相机yaw的修正常数

// bool debug = false;

bool got_angle_velocity = false;
int fit_count = 0;
const int MAX_FIT_COUNT = 300;
std::vector<double> w_big_fits;
std::vector<double> A0_fits;
std::vector<double> fai_fits;
std::vector<double> b_fits;
// 新增标志位，表示是否已固定w、A、b参数
bool params_fixed = false;
// 固定后的参数值
double fixed_w_big = 0.0;
double fixed_A0 = 0.0;
double fixed_b = 0.0;

WMPredict::WMPredict() {
  this->direction = 1;
  this->smoothData_img = cv::Mat::zeros(400, 800, CV_8UC3);
  //====大符速度参数======//
  this->A0 = 1;
  this->w_big = 2;
  this->b = 2.09 - A0;
  this->fai = 0;
  this->now_time = 0;
  this->Fire_time = 0;
  this->First_fit = 1;
}

// 自定义 ceres_fmod
// 函数，用于残差结构体中计算残差的模版函数中（标准的std::fmod()不支持自动微分）
template <typename T> T ceres_fmod(const T &a, const T &b) {
  return a - b * ceres::floor(a / b);
}

// Ceres 残差结构体
struct WindmillResidual {
  WindmillResidual(double t0, double t_i, double theta_i)
      : t0(t0), t_i(t_i), theta_i(theta_i) {}

  template <typename T>
  bool operator()(const T *const params, T *residual) const {
    T A = params[0];
    T w = params[1];
    T fai = params[2];
    T A0 = params[3];

    T dt = T(t_i) - T(t0);
    T dangle = A0 * dt + (A / w) * (ceres::cos(fai) - ceres::cos(w * dt + fai));
    T angle_now = dangle / T(3.1415926) * T(180.0);
    if (angle_now < T(0.0)) {
      angle_now = T(360.0) + angle_now;
    }
    if (angle_now > T(360.0)) {
      angle_now = ceres_fmod(angle_now, T(360.0));
    }

    // 定义残差值为图片中提取的角度与拟合角速度计算出的当前角度的差值
    residual[0] = T(theta_i) - angle_now;
    return true;
  }

  double t0;
  double t_i;
  double theta_i;
};

// 自定义迭代终止(弃用)
struct CustomTerminationCallback : public ceres::IterationCallback {
  const double *params;
  const double true_A, true_w, true_fai, true_A0;

  CustomTerminationCallback(const double *params, double true_A, double true_w,
                            double true_fai, double true_A0)
      : params(params), true_A(true_A), true_w(true_w), true_fai(true_fai),
        true_A0(true_A0) {}

  ceres::CallbackReturnType
  operator()(const ceres::IterationSummary &summary) override {
    double A_error_curr = abs(params[0] - true_A) / true_A;
    double w_error_curr = abs(params[1] - true_w) / true_w;
    // double fai_error_curr = abs(params[2] - true_fai) / true_fai;
    double A0_error_curr = abs(params[3] - true_A0) / true_A0;

    // 所有误差都小于5%，终止
    if (A_error_curr < 0.05 && w_error_curr < 0.05 && A0_error_curr < 0.05) {

      return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
    }
    return ceres::SOLVER_CONTINUE;
  }
};

/**
 * @description: 打符主流程
 * @param {Translator} &translator  串口消息
 * @param {GlobalParam} &gp  传入全局变量
 * @param {WMIdentify} &WMI  传入识别类
 * @return {*}
 */
int WMPredict::StartPredict(Translator &translator, GlobalParam &gp,
                            WMIdentify &WMI) {

  if (WMI.getListStat() == 0) {
    LOG_IF(INFO, gp.switch_INFO) << "识别失败 或者 数据不足，不预测";
    translator.message.yaw = translator.message.yaw;
    return 0;
  } else {
    LOG_IF(INFO, gp.switch_INFO) << "识别成功，开始预测";
  }

  this->UpdateData(WMI, translator);

  // 如果击打大符
  // if (translator.message.status % 5 == 3)
  if (1) {
    // 如果角速度数据数量够，进行拟合
    if (WMI.getAngleVelocityList().size() >= gp.list_size) {
      if (!got_angle_velocity) {
        // if (fit_count < MAX_FIT_COUNT) {
        if (true) {
          std::cout << "拟合 (" << fit_count + 1 << "/" << MAX_FIT_COUNT << ")"
                    << std::endl;

          this->ConvexOptimization(WMI.getTimeList(),
                                   WMI.getAngleVelocityList(), gp, translator);

          w_big_fits.push_back(this->w_big);
          A0_fits.push_back(this->A0);
          fai_fits.push_back(this->fai);
          b_fits.push_back(this->b);

          fit_count++;

          if (fit_count >= MAX_FIT_COUNT) {
            double w_big_sum = 0.0, A0_sum = 0.0, fai_sum = 0.0, b_sum = 0.0;

            for (int i = 0; i < MAX_FIT_COUNT; i++) {
              w_big_sum += w_big_fits[i];
              A0_sum += A0_fits[i];
              fai_sum += fai_fits[i];
              b_sum += b_fits[i];
            }

            fixed_w_big = w_big_sum / MAX_FIT_COUNT;
            fixed_A0 = A0_sum / MAX_FIT_COUNT;
            fixed_b = b_sum / MAX_FIT_COUNT;

            params_fixed = false;

            std::cout << "完成拟合，固定参数：" << std::endl;
            std::cout << "固定 w_big: " << fixed_w_big << std::endl;
            std::cout << "固定 A0: " << fixed_A0 << std::endl;
            std::cout << "固定 b: " << fixed_b << std::endl;
            std::cout << "当前 fai: " << this->fai << std::endl;

            w_big_fits.clear();
            A0_fits.clear();
            fai_fits.clear();
            b_fits.clear();
          }
        }
      } else {
        LOG_IF(INFO, gp.switch_INFO)
            << "使用已拟合参数: w=" << this->w_big << ", A0=" << this->A0;
      }
    } else {
      LOG_IF(INFO, gp.switch_INFO)
          << "数据不够，不拟合 getAngleVelocityList().size() : "
          << WMI.getAngleVelocityList().size();
      return 0;
    }
    // this->NewtonDspBig(WMI.getLastRotAngle(), WMI.getAlpha(), translator, gp,
    // WMI.getR_yaw());
    // this->NewtonDspBigAnyPos(WMI.getTransformationMatrix(), translator, gp,
    //                          WMI.getLastAngle(), WMI.getLastRotAngle());
    this->NewtonDspSmallAnyPos(WMI.getTransformationMatrix(), translator, gp,
                               WMI.getLastAngle());
  }
  // 如果击打小符
  else
    this->NewtonDspSmall(WMI.getLastRotAngle(), WMI.getAlpha(), translator, gp,
                         WMI.getPhi());
  // this->NewtonDspSmall(WMI.getLastAngle(), WMI.getAlpha(), translator, gp,
  //                      WMI.getPhi());

  translator.messageWM.predict_time = WMI.GetFanChangeTime();
  // this->ResultLog(translator, gp, WMI.getYaw());

  return 1;
}
/**
 * @description: 根据识别结果对预测的数据更新
 * @param {double} direction 能量机关旋转方向
 * @param {double} Radius   能量机关半径
 * @param {Point2d} R_center    能量机关中心点
 * @param {Mat} debugImg        当前识别的图片
 * @param {Mat} data_img        角速度队列绘制结果
 * @param {Translator} translator
 * @return {*}
 */
void WMPredict::UpdateData(WMIdentify &WMI, Translator translator) {
  // this->w = abs(direction) > 20 ? 1.047197551 : 0; // 小符角速度
  this->w = -1.047197551;
  // this->w=0;
  this->w = 0;
  this->direction = WMI.getDirection() > 0 ? 1 : -1;

  this->rvec = WMI.getRvec();
  this->tvec = WMI.getTvec();
  this->dist_coeffs = WMI.getDist_coeffs();
  this->camera_matrix = WMI.getCamera_matrix();

  // this->now_time = (double)translator.messageWM.predict_time / 1000;
  this->now_time = WMI.getTimeList()[WMI.getTimeList().size() - 1];

  this->debugImg = WMI.getImg0();
  this->data_img = WMI.getData_img();
}

/**
 * @description: 迭代求解子弹飞行时间与云台下一时刻开火的位姿
 * @param {double} theta_0  当前能量机关角度
 * @param {Translator} &translator
 * @param {GlobalParam} &gp
 * @param {double} R_yaw   当前云台相对于R标（能量机关中心）的yaw角
 * @return {*}
 */

void WMPredict::NewtonDspSmall(double theta_0, double alpha,
                               Translator &translator, GlobalParam &gp,
                               double R_yaw) {
  double P0 = 12 * pi / 180;
  double fly_t0 = 0.3;
  w = this->direction > 0 ? abs(w) : -abs(w);
  int n = 0; // 迭代次数
  translator.messageWM.predict_time =
      translator.messageWM.predict_time + 1000 * delta_t; // 开火时间
  theta_0 = theta_0 + w * delta_t; // 开火时的待击打点角度
  theta_0 += theta_0 < 0 ? 2 * pi : 0;
  theta_0 -= theta_0 > 2 * pi
                 ? 2 * pi
                 : 0; // theta0 范围锁定（0，2pi）
                      // std::cout<<"theta: "<<180*theta_0/pi<<std::endl;
  double v0 = translator.messageWM.bullet_v; // 弹速
  cv::Mat P_t = cv::Mat::zeros(2, 1, CV_64F);
  cv::Mat temp =
      (cv::Mat_<double>(2, 2) << this->f1P(P0, fly_t0, theta_0, v0),
       this->f1t(P0, fly_t0, theta_0, v0), this->f2P(P0, fly_t0, theta_0, v0),
       this->f2t(P0, fly_t0, theta_0, v0));
  cv::Mat temp_inv = cv::Mat::zeros(2, 2, CV_64F);
  cv::Mat b = (cv::Mat_<double>(2, 1) << this->f1(P0, fly_t0, theta_0, v0),
               this->f2(P0, fly_t0, theta_0, v0));
  double P1 = 0;
  double fly_t1 = 0;
  do {
    n++;
    P1 = P0;
    fly_t1 = fly_t0;
    // std::cout<<"P0:"<<P0；
    // std::cout<<" t0:"<<t0<<std::endl;
    //======这里对雅可比矩阵的更新要尽可能的少，不然解变化太快容易求出无意义解（t<0)======//
    temp.at<double>(0, 0) = this->f1P(P0, fly_t0, theta_0, v0);
    // temp.at<double>(0, 1) = f1t(P0, t0, theta_0,v0);
    // temp.at<double>(1, 0) = f2P(P0, t0, theta_0,v0);
    temp.at<double>(1, 1) = this->f2t(P0, fly_t0, theta_0, v0);
    // std::cout<<"temp: "<<temp<<std::endl;
    cv::invert(temp, temp_inv);
    // std::cout<<"temp_inv: "<<temp_inv<<std::endl;
    b.at<double>(0, 0) = this->f1(P0, fly_t0, theta_0, v0);
    b.at<double>(1, 0) = this->f2(P0, fly_t0, theta_0, v0);
    P_t = P_t - temp_inv * b;
    P0 = P_t.at<double>(0, 0);
    fly_t0 = P_t.at<double>(1, 0);
    if (n > 50)
      break;
  } while (abs(fly_t0 - fly_t1) > 1e-5 ||
           abs(P0 - P1) > 1e-5); // 当前解与上次迭代解差距很小时

  double yaw = atan(r * cos(theta_0 + w * fly_t0) / s);
  // cv::putText(this->debugImg, "yaw:" + std::to_string(180*yaw/pi),
  // cv::Point(30, 30), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 100, 0));
#ifdef DEBUGHIT

  double theta_guess = theta_0 + w * fly_t0;
  cv::circle(this->debugImg, CalPointGuess(theta_0), 5, cv::Scalar(0, 255, 255),
             -1);
  cv::circle(this->debugImg, CalPointGuess(theta_guess), 5,
             cv::Scalar(255, 0, 255), -1);

#endif
  //   translator.message.x_a = translator.message.yaw;
  translator.message.pitch = 180 * P0 / pi;
  translator.message.yaw = 180 * (yaw + R_yaw) / pi;
  // std::cout<< translator.message.yaw<<std::endl;
  // 相机中心的偏置修正
  translator.message.yaw += yaw_fix;
  // std::cout << " 开火时待打击点角度： " << 180 / pi * theta_0;
  // LOG_IF(INFO, gp.switch_INFO) << "小符角速度" << w;
  // LOG_IF(INFO, gp.switch_INFO) << " 开火时待打击点角度： " << 180 / pi *
  // theta_0; LOG_IF(INFO, gp.switch_INFO) << " 对应的子弹飞行时间 " << fly_t0;
  // LOG_IF(INFO, gp.switch_INFO) << " " << delta_t << "s后要调整的相对R的yaw: "
  // << 180 / pi * yaw; LOG_IF(INFO, gp.switch_INFO) << " " << delta_t <<
  // "s后要调整的pitch: " << translator.message.pitch;
}
void WMPredict::NewtonDspBig(double theta_0, double alpha,
                             Translator &translator, GlobalParam &gp,
                             double R_yaw) {
  double P0 = 12 * pi / 180;
  double fly_t0 = 0.3;
  int n = 0; // 迭代次数
  theta_0 = -theta_0;
  // t
  cv::putText(this->debugImg,
              "theta_____ORI: " + std::to_string(theta_0 * 180 / pi),
              cv::Point(10, 500), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  // t
  // t
  double delta_theta_delay = ThetaToolForBig(delta_t, this->now_time);
  theta_0 = theta_0 + delta_theta_delay; // 开火时的待击打点角度
  // theta_0 = theta_0 + 7 * pi / 18 - 0.73 * pi;
  // theta_0 = 2 * pi - theta_0;

  // std::cout<<"time:"<<now_time<<std::endl;
  this->Fire_time = this->now_time + delta_t;
  // theta_0 += theta_0 < 0 ? 2 * pi : 0;
  // theta_0 -= theta_0 > 2 * pi ? 2 * pi : 0;
  // std::cout << "theta:" << theta_0 << std::endl;
  // std::cout << "w_big:" << w_big << std::endl;
  // double v0 = translator.messageWM.bullet_v; // 弹速
  double v0 = 270; // 弹速
  cv::Mat P_t = cv::Mat::zeros(2, 1, CV_64F);
  cv::Mat temp =
      (cv::Mat_<double>(2, 2) << this->F1P(P0, fly_t0, theta_0, v0),
       this->F1t(P0, fly_t0, theta_0, v0), this->F2P(P0, fly_t0, theta_0, v0),
       this->F2t(P0, fly_t0, theta_0, v0));
  cv::Mat temp_inv = cv::Mat::zeros(2, 2, CV_64F);
  cv::Mat b = (cv::Mat_<double>(2, 1) << this->F1(P0, fly_t0, theta_0, v0),
               this->F2(P0, fly_t0, theta_0, v0));
  double P1 = 0;
  double fly_t1 = 0;

  do {
    n++;
    P1 = P0;
    fly_t1 = fly_t0;
    //======这里对雅可比矩阵的更新要尽可能的少，不然解变化太快容易求出无意义解（t<0)======//
    temp.at<double>(0, 0) = this->F1P(P0, fly_t0, theta_0, v0);
    // temp.at<double>(0, 1) = f1t(P0, t0, theta_0,v0);
    // temp.at<double>(1, 0) = f2P(P0, t0, theta_0,v0);
    temp.at<double>(1, 1) = this->F2t(P0, fly_t0, theta_0, v0);
    cv::invert(temp, temp_inv);
    b.at<double>(0, 0) = this->F1(P0, fly_t0, theta_0, v0);
    b.at<double>(1, 0) = this->F2(P0, fly_t0, theta_0, v0);
    P_t = P_t - temp_inv * b;
    P0 = P_t.at<double>(0, 0);
    fly_t0 = P_t.at<double>(1, 0);
    if (n > 50)
      break;
  } while (abs(fly_t0 - fly_t1) > 1e-5 ||
           abs(P0 - P1) > 1e-5); // 当前解与上次迭代解差距很小时
  // double yaw =
  //     atan(r * cos(theta_0 + ThetaToolForBig(fly_t0, this->Fire_time)) / s);
  double yaw =
      asin(r * cos(theta_0 + ThetaToolForBig(fly_t0, this->Fire_time)) / s);

  double delta_theta = ThetaToolForBig(fly_t0, this->Fire_time);
  theta_0 = theta_0 + delta_theta;
  // t
  LOG_IF(INFO, gp.switch_INFO)
      << "fire_time!!!!!!!!!!!!!!!!!!: " << this->Fire_time;

  LOG_IF(INFO, gp.switch_INFO)
      << "delta_theta!!!!!!!!!!!!!!!!!!: " << delta_theta * 180 / pi;
  // t

  std::vector<cv::Point3f> world_points_single;
  world_points_single.push_back(
      cv::Point3f(r * cos(delta_theta + delta_theta_delay),
                  -r * sin(delta_theta + delta_theta_delay), 0));
  std::vector<cv::Point2f> image_points_single;
  cv::projectPoints(world_points_single, rvec, tvec, camera_matrix, dist_coeffs,
                    image_points_single);
  cv::circle(this->debugImg, image_points_single[0], 5,
             cv::Scalar(255, 255, 255), -1);

  cv::putText(this->debugImg, "Predicted Point",
              image_points_single[0] + cv::Point2f(10, 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
  cv::putText(this->debugImg, "pitch: " + std::to_string(P0 * 180 / pi),
              cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg,
              "Rel_yaw: " + std::to_string((yaw + R_yaw) * 180 / pi),
              cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg, "theta: " + std::to_string(theta_0 * 180 / pi),
              cv::Point(10, 190), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg, "alpha: " + std::to_string(alpha * 180 / pi),
              cv::Point(10, 260), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg, "phi: " + std::to_string(R_yaw * 180 / pi),
              cv::Point(10, 330), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg,
              "Received Yaw: " +
                  std::to_string(translator.message.yaw * 180 / pi),
              cv::Point(10, 630), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg,
              "Received Pitch: " +
                  std::to_string(translator.message.pitch * 180 / pi),
              cv::Point(10, 700), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);

  // translator.message.x_a = translator.message.yaw;
  translator.message.r2 = (P0 * 180 / pi);
  // std::cout << fly_t0 << std::endl;
  translator.messageWM.send_yaw =
      (translator.message.yaw - (yaw + R_yaw)) * 180 / pi;
  // translator.message.yaw += yaw_fix;
  cv::putText(this->debugImg,
              "Abs_yaw: " + std::to_string(translator.message.yaw),
              cv::Point(10, 400), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);

  cv::Size newSize(640, 480);
  cv::resize(debugImg, debugImg, newSize, 0, 0, cv::INTER_LINEAR);
  cv::imshow("debugImg", this->debugImg);
  cv::waitKey(1);

  translator.messageWM.predict_time =
      translator.messageWM.predict_time + 1000 * delta_t; // 发给电控的开火时间

  // LOG_IF(INFO, gp.switch_INFO) << " 对应的子弹飞行时间 " << fly_t0;
  // LOG_IF(INFO, gp.switch_INFO) << " " << delta_t << "s后要调整到的yaw: " <<
  // 180 / pi * yaw; LOG_IF(INFO, gp.switch_INFO) << " " << delta_t <<
  // "s后要调整的pitch: " << translator.message.pitch; LOG_IF(INFO,
  // gp.switch_INFO) << " 开火时待打击点角度： " << 180 / pi * theta_0;
  // LOG_IF(INFO, gp.switch_INFO) << " 对应的子弹飞行时间 " << fly_t0;
}
void WMPredict::NewtonDspSmallAnyPos(cv::Mat world2car, Translator &translator,
                                     GlobalParam &gp, double R_yaw) {
  double P0 = 12 * pi / 180;
  double fly_t0 = 0.3;
  int n = 0; // 迭代次数
  double delta_theta_delay = w * delta_t;

  // cv::Mat world_point = (cv::Mat_<double>(4, 1) << r *
  // cos(delta_theta_delay), -r * sin(delta_theta_delay), 0, 1.0);
  cv::Mat world_point = (cv::Mat_<double>(4, 1) << r, 0, 0, 1.0);
  cv::Mat world_point_car = world2car * world_point;
  double x_car = world_point_car.at<double>(0, 0);
  double y_car = world_point_car.at<double>(1, 0);
  double z_car = world_point_car.at<double>(2, 0);
  double distance = sqrt(x_car * x_car + z_car * z_car);
  this->Fire_time = this->now_time + delta_t;
  double v0 = 23.5; // 弹速
  cv::Mat P_t = cv::Mat::zeros(2, 1, CV_64F);
  cv::Mat temp =
      (cv::Mat_<double>(2, 2) << this->f1PA(P0, x_car, fly_t0, distance, v0),
       this->f1tA(P0, delta_theta_delay, world2car, fly_t0, distance, v0),
       this->f2PA(P0, z_car, fly_t0, v0),
       this->f2tA(P0, delta_theta_delay, world2car, fly_t0, v0));
  cv::Mat temp_inv = cv::Mat::zeros(2, 2, CV_64F);
  cv::Mat b =
      (cv::Mat_<double>(2, 1)
           << this->f1A(P0, delta_theta_delay, world2car, fly_t0, distance, v0),
       this->f2A(P0, delta_theta_delay, world2car, fly_t0, v0));
  double P1 = 0;
  double fly_t1 = 0;

  do {
    n++;
    P1 = P0;
    fly_t1 = fly_t0;
    //======这里对雅可比矩阵的更新要尽可能的少，不然解变化太快容易求出无意义解（t<0)======//
    temp.at<double>(0, 0) = this->f1PA(P0, x_car, fly_t0, distance, v0);
    temp.at<double>(1, 1) =
        this->f2tA(P0, delta_theta_delay, world2car, fly_t0, v0);
    cv::invert(temp, temp_inv);
    b.at<double>(0, 0) =
        this->f1A(P0, delta_theta_delay, world2car, fly_t0, distance, v0);
    b.at<double>(1, 0) =
        this->f2A(P0, delta_theta_delay, world2car, fly_t0, v0);
    P_t = P_t - temp_inv * b;
    P0 = P_t.at<double>(0, 0);
    fly_t0 = P_t.at<double>(1, 0);
    if (n > 50)
      break;
  } while (abs(fly_t0 - fly_t1) > 1e-5 ||
           abs(P0 - P1) > 1e-5); // 当前解与上次迭代解差距很小时

  // double yaw = atan2(x_car, z_car);

  double delta_theta = w * fly_t0 + delta_theta_delay;

  // // t
  // LOG_IF(INFO, gp.switch_INFO)
  //     << "fire_time!!!!!!!!!!!!!!!!!!: " << this->Fire_time;

  // LOG_IF(INFO, gp.switch_INFO)
  //     << "delta_theta!!!!!!!!!!!!!!!!!!: " << delta_theta * 180 / pi;
  // // t
  std::vector<cv::Point3f> world_points_single;
  cv::Point3f point3f(r * cos(delta_theta + delta_theta_delay),
                      -r * sin(delta_theta + delta_theta_delay), 0);
  // cv::Point3f point3f(r, 0, 0);
  world_points_single.push_back(point3f);
  std::vector<cv::Point2f> image_points_single;
  cv::projectPoints(world_points_single, rvec, tvec, camera_matrix, dist_coeffs,
                    image_points_single);
  cv::circle(this->debugImg, image_points_single[0], 5,
             cv::Scalar(255, 255, 255), -1);

  cv::Mat world_point_goal =
      (cv::Mat_<double>(4, 1) << r * cos(delta_theta + delta_theta_delay),
       -r * sin(delta_theta + delta_theta_delay), 0, 1.0);
  // cv::Mat world_point_goal = (cv::Mat_<double>(4, 1) << r, 0, 0, 1.0);
  cv::Mat world_point_goal_car = world2car * world_point_goal;

  // std::cout << -world_point_goal_car.at<double>(0, 0) <<
  // world_point_goal_car.at<double>(1, 0) << std::endl;

  double yaw = atan2(-world_point_goal_car.at<double>(0, 0),
                     world_point_goal_car.at<double>(2, 0));

  // std::cout << yaw * 180 / pi << std::endl;

  cv::putText(this->debugImg, "Predicted Point",
              image_points_single[0] + cv::Point2f(10, 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
  cv::putText(this->debugImg, "pitch: " + std::to_string(P0 * 180 / pi),
              cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg, "Abs_yaw: " + std::to_string(yaw * 180 / pi),
              cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);

  cv::putText(this->debugImg,
              "x_car: " + std::to_string(world_point_goal_car.at<double>(0, 0)),
              cv::Point(10, 260), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg,
              "y_car: " + std::to_string(world_point_goal_car.at<double>(1, 0)),
              cv::Point(10, 330), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg,
              "z_car: " + std::to_string(world_point_goal_car.at<double>(2, 0)),
              cv::Point(10, 400), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg, "distance: " + std::to_string(distance),
              cv::Point(10, 470), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg, "R_yaw: " + std::to_string(R_yaw * 180 / pi),
              cv::Point(10, 540), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg,
              "receivedyaw: " +
                  std::to_string(translator.message.yaw * 180 / pi),
              cv::Point(10, 610), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg,
              "receivedpitch: " +
                  std::to_string(translator.message.pitch * 180 / pi),
              cv::Point(10, 680), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg,
              "rel_yaw: " +
                  std::to_string((yaw - translator.message.yaw) * 180 / pi),
              cv::Point(10, 750), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg,
              "rel_pitch: " +
                  std::to_string((P0 - translator.message.pitch) * 180 / pi),
              cv::Point(10, 820), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);

  this->resultYaw = yaw * 180 / pi;
  this->resultPitch = P0 * 180 / pi;

  translator.messageWM.send_yaw = yaw * 180 / pi;
  translator.message.r2 = P0 * 180 / pi;

  // translator.message.x_a = translator.message.yaw;
  // translator.message.pitch = (P0 * 180 / pi);
  // std::cout << fly_t0 << std::endl;
  // translator.message.yaw = (translator.message.yaw + (yaw - R_yaw)) * 180 /
  // pi; translator.message.yaw += yaw_fix;

  cv::Size newSize(630, 465);
  cv::resize(debugImg, debugImg, newSize, 0, 0, cv::INTER_LINEAR);
  cv::resize(debugImg, debugImg, newSize, 0, 0, cv::INTER_LINEAR);

  cv::imshow("debugImg", this->debugImg);
  cv::waitKey(1);

  translator.messageWM.predict_time =
      translator.messageWM.predict_time + 1000 * delta_t; // 发给电控的开火时间

  LOG_IF(INFO, gp.switch_INFO) << " 对应的子弹飞行时间 " << fly_t0;
  LOG_IF(INFO, gp.switch_INFO)
      << " " << delta_t << "s后要调整到的yaw: " << 180 / pi * yaw;
  LOG_IF(INFO, gp.switch_INFO)
      << " " << delta_t << "s后要调整的pitch: " << translator.message.pitch;

  LOG_IF(INFO, gp.switch_INFO) << " 对应的子弹飞行时间: " << fly_t0;
}
void WMPredict::NewtonDspBigAnyPos(cv::Mat world2car, Translator &translator,
                                   GlobalParam &gp, double R_yaw,
                                   double R_distance) {
  double P0 = 12 * pi / 180;
  double fly_t0 = 0.3;
  int n = 0; // 迭代次数
  double delta_theta_delay = ThetaToolForBig(delta_t, this->now_time);

  // cv::Mat world_point = (cv::Mat_<double>(4, 1) << r *
  // cos(delta_theta_delay), -r * sin(delta_theta_delay), 0, 1.0);
  cv::Mat world_point = (cv::Mat_<double>(4, 1) << r, 0, 0, 1.0);
  cv::Mat world_point_car = world2car * world_point;
  double x_car = world_point_car.at<double>(0, 0);
  double y_car = world_point_car.at<double>(1, 0);
  double z_car = world_point_car.at<double>(2, 0);
  double distance = sqrt(x_car * x_car + y_car * y_car + z_car * z_car);
  this->Fire_time = this->now_time + delta_t;
  double v0 = 23.5; // 弹速
  cv::Mat P_t = cv::Mat::zeros(2, 1, CV_64F);
  cv::Mat temp =
      (cv::Mat_<double>(2, 2) << this->F1PA(P0, x_car, fly_t0, distance, v0),
       this->F1tA(P0, delta_theta_delay, world2car, fly_t0, distance, v0),
       this->F2PA(P0, z_car, fly_t0, v0),
       this->F2tA(P0, delta_theta_delay, world2car, fly_t0, v0));
  cv::Mat temp_inv = cv::Mat::zeros(2, 2, CV_64F);
  cv::Mat b =
      (cv::Mat_<double>(2, 1)
           << this->F1A(P0, delta_theta_delay, world2car, fly_t0, distance, v0),
       this->F2A(P0, delta_theta_delay, world2car, fly_t0, v0));
  double P1 = 0;
  double fly_t1 = 0;

  do {
    n++;
    P1 = P0;
    fly_t1 = fly_t0;
    //======这里对雅可比矩阵的更新要尽可能的少，不然解变化太快容易求出无意义解（t<0)======//
    temp.at<double>(0, 0) = this->F1PA(P0, x_car, fly_t0, distance, v0);
    temp.at<double>(1, 1) =
        this->F2tA(P0, delta_theta_delay, world2car, fly_t0, v0);
    cv::invert(temp, temp_inv);
    b.at<double>(0, 0) =
        this->F1A(P0, delta_theta_delay, world2car, fly_t0, distance, v0);
    b.at<double>(1, 0) =
        this->F2A(P0, delta_theta_delay, world2car, fly_t0, v0);
    P_t = P_t - temp_inv * b;
    P0 = P_t.at<double>(0, 0);
    fly_t0 = P_t.at<double>(1, 0);
    if (n > 50)
      break;
  } while (abs(fly_t0 - fly_t1) > 1e-5 ||
           abs(P0 - P1) > 1e-5); // 当前解与上次迭代解差距很小时

  // double yaw = atan2(x_car, z_car);

  double delta_theta = ThetaToolForBig(fly_t0, this->Fire_time);

  // // t
  // LOG_IF(INFO, gp.switch_INFO)
  //     << "fire_time!!!!!!!!!!!!!!!!!!: " << this->Fire_time;

  // LOG_IF(INFO, gp.switch_INFO)
  //     << "delta_theta!!!!!!!!!!!!!!!!!!: " << delta_theta * 180 / pi;
  // // t
  std::vector<cv::Point3f> world_points_single;
  cv::Point3f point3f(r * cos(delta_theta + delta_theta_delay),
                      -r * sin(delta_theta + delta_theta_delay), 0);
  world_points_single.push_back(point3f);
  std::vector<cv::Point2f> image_points_single;
  cv::projectPoints(world_points_single, rvec, tvec, camera_matrix, dist_coeffs,
                    image_points_single);
  cv::circle(this->debugImg, image_points_single[0], 5,
             cv::Scalar(255, 255, 255), -1);

  cv::Mat world_point_goal =
      (cv::Mat_<double>(4, 1) << r * cos(delta_theta + delta_theta_delay),
       -r * sin(delta_theta + delta_theta_delay), 0, 1.0);
  cv::Mat world_point_goal_car = world2car * world_point_goal;

  // std::cout << -world_point_goal_car.at<double>(0, 0) <<
  // world_point_goal_car.at<double>(1, 0) << std::endl;

  double yaw = atan2(-world_point_goal_car.at<double>(0, 0),
                     world_point_goal_car.at<double>(2, 0));

  // std::cout << yaw * 180 / pi << std::endl;

  cv::putText(this->debugImg, "Predicted Point",
              image_points_single[0] + cv::Point2f(10, 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
  cv::putText(this->debugImg, "pitch: " + std::to_string(P0 * 180 / pi),
              cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg, "Abs_yaw: " + std::to_string(yaw * 180 / pi),
              cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);

  cv::putText(this->debugImg,
              "x_car: " + std::to_string(world_point_goal_car.at<double>(0, 0)),
              cv::Point(10, 260), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg,
              "y_car: " + std::to_string(world_point_goal_car.at<double>(1, 0)),
              cv::Point(10, 330), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);

  cv::putText(this->debugImg,
              "z_car: " + std::to_string(world_point_goal_car.at<double>(2, 0)),
              cv::Point(10, 400), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg,
              "distance: " + std::to_string(R_distance * 180 / pi),
              cv::Point(10, 470), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg, "Rotangle: " + std::to_string(R_yaw * 180 / pi),
              cv::Point(10, 540), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg,
              "receiveyaw: " +
                  std::to_string(translator.message.yaw * 180 / pi),
              cv::Point(10, 610), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg,
              "receivepitch: " +
                  std::to_string(translator.message.pitch * 180 / pi),
              cv::Point(10, 680), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  // cv::putText(this->debugImg, "rel_yaw: " + std::to_string((yaw -
  // translator.message.yaw) * 180 / pi), cv::Point(10, 750),
  // cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg,
              "rel_yaw: " +
                  std::to_string((translator.message.yaw - yaw) * 180 / pi),
              cv::Point(10, 750), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg,
              "rel_pitch: " +
                  std::to_string((P0 - translator.message.pitch) * 180 / pi),
              cv::Point(10, 820), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg, "w_big: " + std::to_string(this->w_big),
              cv::Point(810, 610), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg, "A0: " + std::to_string(this->A0),
              cv::Point(810, 680), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg, "b: " + std::to_string((this->b) * 180 / pi),
              cv::Point(810, 750), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(this->debugImg, "phi: " + std::to_string((cos(this->fai))),
              cv::Point(810, 820), cv::FONT_HERSHEY_SIMPLEX, 2,
              cv::Scalar(0, 255, 0), 2);

  this->resultYaw = yaw * 180 / pi;
  this->resultPitch = P0 * 180 / pi;

  translator.messageWM.send_yaw = yaw * 180 / pi;
  translator.message.r2 = P0 * 180 / pi;

  // translator.message.x_a = translator.message.yaw;
  // translator.message.pitch = (P0 * 180 / pi);
  // std::cout << fly_t0 << std::endl;
  // translator.message.yaw = (translator.message.yaw + (yaw - R_yaw)) * 180 /
  // pi; translator.message.yaw += yaw_fix;
  cv::Size newSize(840, 620);
  cv::resize(debugImg, debugImg, newSize, 0, 0, cv::INTER_LINEAR);

  static std::vector<double> fai_history;
  static std::vector<double> time_points;
  static std::vector<double> yaw_history;
  static std::vector<double> pitch_history;
  static double start_time = cv::getTickCount() / cv::getTickFrequency();
  double current_time =
      cv::getTickCount() / cv::getTickFrequency() - start_time;

  fai_history.push_back(this->fai * 180 / pi);
  yaw_history.push_back(this->resultYaw);
  pitch_history.push_back(this->resultPitch);
  time_points.push_back(current_time);

  if (fai_history.size() > 200) {
    fai_history.erase(fai_history.begin());
    yaw_history.erase(yaw_history.begin());
    pitch_history.erase(pitch_history.begin());
    time_points.erase(time_points.begin());
  }

  cv::Mat curve(300, 600, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat yaw_curve(300, 600, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat pitch_curve(300, 600, CV_8UC3, cv::Scalar(0, 0, 0));

  cv::line(curve, cv::Point(50, 250), cv::Point(550, 250),
           cv::Scalar(255, 255, 255), 1);
  cv::line(curve, cv::Point(50, 50), cv::Point(50, 250),
           cv::Scalar(255, 255, 255), 1);

  cv::line(yaw_curve, cv::Point(50, 250), cv::Point(550, 250),
           cv::Scalar(255, 255, 255), 1);
  cv::line(yaw_curve, cv::Point(50, 50), cv::Point(50, 250),
           cv::Scalar(255, 255, 255), 1);

  cv::line(pitch_curve, cv::Point(50, 250), cv::Point(550, 250),
           cv::Scalar(255, 255, 255), 1);
  cv::line(pitch_curve, cv::Point(50, 50), cv::Point(50, 250),
           cv::Scalar(255, 255, 255), 1);

  for (size_t i = 1; i < fai_history.size(); i++) {
    double x1 = 50 + (time_points[i - 1] - time_points[0]) /
                         (time_points.back() - time_points[0]) * 500;
    double y1 = 250 - fai_history[i - 1] * 200 / 360;
    double x2 = 50 + (time_points[i] - time_points[0]) /
                         (time_points.back() - time_points[0]) * 500;
    double y2 = 250 - fai_history[i] * 200 / 360;

    cv::line(curve, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),
             2);
  }

  for (size_t i = 1; i < yaw_history.size(); i++) {
    double x1 = 50 + (time_points[i - 1] - time_points[0]) /
                         (time_points.back() - time_points[0]) * 500;
    double y1 = 150 - (yaw_history[i - 1]) * 180 / 20; // 150是中心线(0度)的位置
    double x2 = 50 + (time_points[i] - time_points[0]) /
                         (time_points.back() - time_points[0]) * 500;
    double y2 = 150 - (yaw_history[i]) * 180 / 20;

    cv::line(yaw_curve, cv::Point(x1, y1), cv::Point(x2, y2),
             cv::Scalar(0, 0, 255), 2);
  }

  cv::line(yaw_curve, cv::Point(50, 150), cv::Point(550, 150),
           cv::Scalar(100, 100, 100), 1, cv::LINE_8);

  for (size_t i = 1; i < pitch_history.size(); i++) {
    double x1 = 50 + (time_points[i - 1] - time_points[0]) /
                         (time_points.back() - time_points[0]) * 500;
    double y1 = 250 - (pitch_history[i - 1]) * 180 / 25; // 直接显示绝对值
    double x2 = 50 + (time_points[i] - time_points[0]) /
                         (time_points.back() - time_points[0]) * 500;
    double y2 = 250 - (pitch_history[i]) * 180 / 25;

    cv::line(pitch_curve, cv::Point(x1, y1), cv::Point(x2, y2),
             cv::Scalar(255, 0, 0), 2);
  }

  cv::line(yaw_curve, cv::Point(50, 60), cv::Point(550, 60),
           cv::Scalar(50, 50, 50), 1, cv::LINE_8); // +10度
  cv::line(yaw_curve, cv::Point(50, 240), cv::Point(550, 240),
           cv::Scalar(50, 50, 50), 1, cv::LINE_8); // -10度
  cv::putText(yaw_curve, "+10°", cv::Point(20, 65), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(200, 200, 200), 1);
  cv::putText(yaw_curve, "0°", cv::Point(30, 155), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(200, 200, 200), 1);
  cv::putText(yaw_curve, "-10°", cv::Point(20, 245), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(200, 200, 200), 1);

  cv::line(pitch_curve, cv::Point(50, 130), cv::Point(550, 130),
           cv::Scalar(50, 50, 50), 1, cv::LINE_8); // 12.5度
  cv::putText(pitch_curve, "25°", cv::Point(25, 75), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(200, 200, 200), 1);
  cv::putText(pitch_curve, "12.5°", cv::Point(15, 135),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
  cv::putText(pitch_curve, "0°", cv::Point(30, 255), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(200, 200, 200), 1);

  cv::putText(curve, "Fai", cv::Point(250, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
              cv::Scalar(255, 255, 255), 1);
  cv::putText(yaw_curve, "Yaw", cv::Point(250, 30), cv::FONT_HERSHEY_SIMPLEX,
              0.8, cv::Scalar(255, 255, 255), 1);
  cv::putText(pitch_curve, "Pitch", cv::Point(250, 30),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);

  cv::imshow("Fai", curve);
  cv::imshow("Yaw", yaw_curve);
  cv::imshow("Pitch", pitch_curve);
  cv::imshow("debugImg", this->debugImg);
  cv::waitKey(1);

  translator.messageWM.predict_time =
      translator.messageWM.predict_time + 1000 * delta_t; // 发给电控的开火时间

  // LOG_IF(INFO, gp.switch_INFO) << " 对应的子弹飞行时间 " << fly_t0;
  // LOG_IF(INFO, gp.switch_INFO) << " " << delta_t << "s后要调整到的yaw: " <<
  // 180 / pi * yaw; LOG_IF(INFO, gp.switch_INFO) << " " << delta_t <<
  // "s后要调整的pitch: " << translator.message.pitch;

  // LOG_IF(INFO, gp.switch_INFO) << " 对应的子弹飞行时间: " << fly_t0;
}

/**
 * @description: 输入w后，进行P参数的最优估计
 * @param {double} w
 * @param {double} &p1
 * @param {double} &p2
 * @param {double} &p3
 * @param {deque<double>} x_data  时间队列
 * @param {deque<double>} y_data  角速度队列
 * @return {double} 返回残差
 */
double WMPredict::Estim(double w, double &p1, double &p2, double &p3,
                        std::deque<double> x_data, std::deque<double> y_data) {
  std::vector<double> x1;
  std::vector<double> x2;
  std::vector<double> temp1;
  std::vector<double> temp2;
  std::vector<double> temp3;
  for (auto x : x_data) {
    x2.push_back(cos(w * x));
    x1.push_back(sin(w * x));
  }

  // 对两个向量进行操作
  std::transform(x1.begin(), x1.end(), x2.begin(), std::back_inserter(temp1),
                 [](double a, double b) { return a * b; });
  std::transform(x1.begin(), x1.end(), y_data.begin(),
                 std::back_inserter(temp2),
                 [](double a, double b) { return a * b; });
  std::transform(x2.begin(), x2.end(), y_data.begin(),
                 std::back_inserter(temp3),
                 [](double a, double b) { return a * b; });

  // 最小二乘法的求和
  double sum_x1x2 = std::accumulate(temp1.begin(), temp1.end(), 0.0);
  double sum_x1y = std::accumulate(temp2.begin(), temp2.end(), 0.0);
  double sum_x2y = std::accumulate(temp3.begin(), temp3.end(), 0.0);
  double sum_y = std::accumulate(y_data.begin(), y_data.end(), 0.0);
  double sum_x1 = std::accumulate(x1.begin(), x1.end(), 0.0);
  double sum_x2 = std::accumulate(x2.begin(), x2.end(), 0.0);
  double sum_x1x1 = std::accumulate(
      x1.begin(), x1.end(), 0.0, [](double a, double b) { return a + b * b; });
  double sum_x2x2 = std::accumulate(
      x2.begin(), x2.end(), 0.0, [](double a, double b) { return a + b * b; });

  // 定义矩阵
  cv::Mat_<double> A(3, 3);
  cv::Mat_<double> b(3, 1);
  A << sum_x1x1, sum_x1x2, sum_x1, sum_x1x2, sum_x2x2, sum_x2, sum_x1, sum_x2,
      y_data.size();
  b << sum_x1y, sum_x2y, sum_y;
  // 求解 Ax=b
  cv::Mat_<double> x = A.inv() * b;
  p1 = x(0);
  p2 = x(1);
  p3 = x(2);
  // 求出残差和
  double err_sum = 0;
  for (int i = 0; i < x1.size(); i++) {
    err_sum += abs(x1[i] * p1 + x2[i] * p2 + p3 - y_data[i]);
  }

  return err_sum;
}
/**
 * @description: 拟合函数，遍历w选取残差最小参数组
 * @param {deque<double>} x_data
 * @param {deque<double>} y_data
 * @param {GlobalParam} &gp
 * @param {Translator} &tr
 * @return {*}
 */
void WMPredict::ConvexOptimization(std::deque<double> x_data,
                                   std::deque<double> y_data, GlobalParam &gp,
                                   Translator &tr) {
  for (int i = 0; i < y_data.size(); i++) {
    y_data[i] = abs(y_data[i]);
  }
  double w0 = 0;
  double p1 = 0;
  double p2 = 0;
  double p3 = 0;
  double err_min = 0;
  std::vector<double> err_list;

  std::once_flag once;
  double p1_temp;
  double p2_temp;
  double p3_temp;
  double err_temp;

  // 如果已经固定了w、A、b参数，只需拟合fai
  if (params_fixed) {
    // 使用固定的参数
    w0 = fixed_w_big;
    p3 = fixed_b;

    // 遍历可能的fai值寻找最小误差
    double fai_step = 0.01;
    for (double fai_temp = -M_PI; fai_temp < M_PI; fai_temp += fai_step) {
      // 用固定的w_big和A0计算p1和p2
      p1_temp = fixed_A0 * cos(fai_temp);
      p2_temp = fixed_A0 * sin(fai_temp);

      // 计算当前fai的误差
      double error_sum = 0.0;
      for (int i = 0; i < x_data.size(); i++) {
        double predicted =
            p1_temp * cos(w0 * x_data[i]) + p2_temp * sin(w0 * x_data[i]) + p3;
        double actual = y_data[i];
        error_sum += pow(predicted - actual, 2);
      }

      if (err_list.size() == 0 || error_sum < err_min) {
        err_min = error_sum;
        p1 = p1_temp;
        p2 = p2_temp;
      }

      err_list.push_back(error_sum);
    }

    this->w_big = fixed_w_big;
    this->fai = atan2(p2, p1);
    this->A0 = fixed_A0;
    this->b = fixed_b;
  }
  // 否则进行完整的拟合
  else {
    // 遍历所有 w 找出最小的残差对应的 w
    for (double w_temp = w_low; w_temp < w_up; w_temp += diff_w) {
      err_temp = Estim(w_temp, p1_temp, p2_temp, p3_temp, x_data, y_data);
      // std::cout << err_temp << std::endl;
      if (err_list.size() != 0) {

        std::call_once(once, [&]() {
          err_min = err_list[0];
          w0 = w_temp;
          p1 = p1_temp;
          p2 = p2_temp;
          p3 = p3_temp;
        });
        if (err_temp < err_min) {
          err_min = err_temp;
          w0 = w_temp;
          p1 = p1_temp;
          p2 = p2_temp;
          p3 = p3_temp;
        }
      }
      err_list.push_back(err_temp);
    }

    this->w_big = w0;
    this->fai = atan2(p2, p1);
    this->A0 = p1 / cos(fai);
    this->b = p3;
  }

  if (w_big < 2.1 && w_big > 1.9 && A0 < 1.01 && A0 > 0.99) {
    // got_angle_velocity = true;
  }
  // this->w_big = 2;
  // this->fai = 0;

  // this->A0 = 1;
  // this->b = 1.09;
  // this->w_big = 0;
  // this->fai = 0;

  // this->A0 = 0;
  // this->b = 0;

#ifdef DEBUGHIT
  cv::Mat word_show = cv::Mat::zeros(700, 800, CV_8UC3);
  cv::putText(
      word_show, "time:" + std::to_string(tr.messageWM.predict_time / 1000),
      cv::Point(30, 390), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 0));
  // cv::putText(word_show, "final_cost:" + std::to_string(sun), cv::Point(30,
  // 330), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 0));
  cv::putText(word_show, "A0:" + std::to_string(this->A0), cv::Point(30, 420),
              cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 0));
  cv::putText(word_show, "w0:" + std::to_string(this->w_big),
              cv::Point(30, 450), cv::FONT_HERSHEY_PLAIN, 1,
              cv::Scalar(255, 255, 0));
  cv::putText(word_show, "fai0:" + std::to_string(this->fai),
              cv::Point(30, 480), cv::FONT_HERSHEY_PLAIN, 1,
              cv::Scalar(255, 255, 0));
  cv::putText(word_show, "b:" + std::to_string(this->b), cv::Point(30, 510),
              cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 0));
  cv::putText(word_show, "p1:" + std::to_string(p1), cv::Point(30, 540),
              cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 0));
  cv::putText(word_show, "p2:" + std::to_string(p2), cv::Point(30, 570),
              cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 0));
  cv::putText(word_show, "p3:" + std::to_string(p3), cv::Point(30, 600),
              cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 0));
  cv::putText(word_show, "err_min:" + std::to_string(err_min),
              cv::Point(30, 630), cv::FONT_HERSHEY_PLAIN, 1,
              cv::Scalar(255, 255, 0));
  cv::putText(word_show, "err_judge:" + std::to_string(err_min / y_data.size()),
              cv::Point(30, 660), cv::FONT_HERSHEY_PLAIN, 1,
              cv::Scalar(255, 255, 0));
  cv::imshow("word_show", word_show);

  cv::Point2f first_point(x_data[0], abs(y_data[0]));
  cv::Mat data_img = cv::Mat::zeros(1080, 1440, CV_8UC3);
  for (int i = 0; i < x_data.size(); i++) {

    cv::Point2f now_point(x_data[i], abs(y_data[i]));
    cv::Point2f now_point_fit(x_data[i], abs(y_data[i]));

    now_point.x -= first_point.x;
    now_point.x *= 20;
    now_point.y = now_point.y * 100;

    now_point_fit.x -= first_point.x;
    now_point_fit.x *= 20;
    now_point_fit.y = (A0 * sin(w0 * x_data[i] + fai) + b) * 100;

    cv::circle(data_img, now_point, 1, cv::Scalar(0, 255, 255));
    // cv::circle(this->smoothData_img, now_point_fit, 1, cv::Scalar(0, 0,
    // 255));
    cv::circle(data_img, now_point_fit, 1, cv::Scalar(0, 0, 255));
  }
  cv::imshow("initial", data_img);
#endif // DEBUGHIT
  // LOG_IF(INFO, gp.switch_INFO) << "estimated A:" << this->A0;
  // LOG_IF(INFO, gp.switch_INFO) << "estimated w:" << this->w_big;
  // LOG_IF(INFO, gp.switch_INFO) << "estimated sketchy:fai:" << this->fai <<
  // std::endl;
}
/**
 * @description: 预测结果日志输出
 * @param {Translator} &translator
 * @param {GlobalParam} &gp
 * @param {double} R_yaw  当前云台相对于R标（能量机关中心）的yaw角
 * @return {*}
 */
void WMPredict::ResultLog(Translator &translator, GlobalParam &gp,
                          double R_yaw) {
#ifdef DEBUGHIT
  cv::putText(
      this->debugImg, "bullet_v:" + std::to_string(translator.messageWM.bullet_v),
      cv::Point(30, 60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 100, 0));

  cv::putText(this->debugImg, "delta_t:" + std::to_string(delta_t),
              cv::Point(30, 90), cv::FONT_HERSHEY_PLAIN, 1,
              cv::Scalar(255, 100, 0));
  cv::putText(
      this->debugImg, "pitch:" + std::to_string(translator.message.pitch),
      cv::Point(30, 120), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 100, 0));
  cv::putText(
      this->debugImg, "delta_yaw:" + std::to_string(translator.message.yaw),
      cv::Point(30, 150), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 100, 0));
  cv::putText(this->debugImg, "R_yaw:" + std::to_string(R_yaw * 180 / pi),
              cv::Point(30, 180), cv::FONT_HERSHEY_PLAIN, 1,
              cv::Scalar(255, 100, 0));
  cv::putText(this->debugImg, "w:" + std::to_string(w), cv::Point(30, 210),
              cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 100, 0));
#endif
  LOG_IF(INFO, gp.switch_INFO) << "当前状态: " << +translator.message.status;
  LOG_IF(INFO, gp.switch_INFO)
      << "拍照时(" << translator.messageWM.predict_time
      << ")云台相对于中心点R的yaw: " << 180 / pi * R_yaw;
  LOG_IF(INFO, gp.switch_INFO)
      << "预测时间戳: " << translator.messageWM.predict_time;
}

double WMPredict::ThetaToolForBig(double dt,
                                  double t0) // 计算t0->t0+dt的大符角度
{
  return -(this->b * dt + this->A0 / this->w_big *
                              (cos(this->w_big * t0 + this->fai) -
                               cos(this->w_big * (t0 + dt) + this->fai)));
}
cv::Point2f WMPredict::CalPointGuess(double theta) // 似乎没什么用
{
  std::vector<cv::Point3f> objectPoints = {
      cv::Point3f(r * cos(theta), -r * sin((theta)), 0)};

  // 投影二维点
  std::vector<cv::Point2f> imagePoints;
  cv::projectPoints(objectPoints, this->rvec, this->tvec, this->camera_matrix,
                    this->dist_coeffs, imagePoints);
  cv::Point2f point_guess = imagePoints[0];

  cv::circle(this->debugImg, point_guess, 5, cv::Scalar(0, 0, 255), -1);
  cv::imshow("CalPointGuess", this->debugImg);

  return point_guess;
}
double WMPredict::f1(double P0, double fly_t0, double theta_0, double v0) {
  return sqrt(pow(r * cos(theta_0 + w * fly_t0), 2) + s * s) -
         v0 * cos(P0) / k + v0 / k * cos(P0) * pow(Exp, -k * fly_t0);
}

double WMPredict::f2(double P0, double fly_t0, double theta_0, double v0) {
  return h0 + r * sin(theta_0 + w * fly_t0) -
         (k * v0 * sin(P0) + g -
          (k * v0 * sin(P0) + g) * pow(Exp, -k * fly_t0) - g * k * fly_t0) /
             (k * k);
}

double WMPredict::f1P(double P, double fly_t, double theta_0,
                      double v0) // f1关于p的导数
{
  return v0 * sin(P) / k * (1 - pow(Exp, -k * fly_t));
}
double WMPredict::f1t(double P, double fly_t, double theta_0,
                      double v0) // f1关于t的导数
{
  return (-r * r * w * cos(theta_0 + w * fly_t) * sin(theta_0 + w * fly_t)) /
             sqrt(pow(r * cos(theta_0 + w * fly_t), 2) + s * s) -
         v0 * cos(P) * pow(Exp, -k * fly_t);
}
double WMPredict::f2P(double P, double fly_t, double theta_0,
                      double v0) // f2关于p的导数
{
  return v0 * cos(P) / k * (pow(Exp, -k * fly_t) - 1);
}
double WMPredict::f2t(double P, double fly_t, double theta_0,
                      double v0) // f2关于t的导数
{
  return w * r * cos(theta_0 + w * fly_t) -
         (k * v0 * sin(P) + g) * pow(Exp, -k * fly_t) / k + g / k;
}

double WMPredict::f1A(double P0, double delta_theta_delay,
                      const cv::Mat &world2car, double fly_t0, double distance,
                      double v0) {
  cv::Mat world_point =
      (cv::Mat_<double>(4, 1) << r * cos(delta_theta_delay + w * fly_t0),
       -r * sin(delta_theta_delay + w * fly_t0), 0, 1.0);
  cv::Mat world_point_car = world2car * world_point;
  return sqrt(distance * distance - pow(world_point_car.at<double>(1, 0), 2)) -
         v0 * cos(P0) / k + v0 / k * cos(P0) * pow(Exp, -k * fly_t0);
}

double WMPredict::f2A(double P0, double delta_theta_delay,
                      const cv::Mat &world2car, double fly_t0, double v0) {
  cv::Mat world_point =
      (cv::Mat_<double>(4, 1) << r * cos(delta_theta_delay + w * fly_t0),
       -r * sin(delta_theta_delay + w * fly_t0), 0, 1.0);
  cv::Mat world_point_car = world2car * world_point;
  return -world_point_car.at<double>(1, 0) - 0.4 -
         (k * v0 * sin(P0) + g -
          (k * v0 * sin(P0) + g) * pow(Exp, -k * fly_t0) - g * k * fly_t0) /
             (k * k);
}

// 关于 P0 的偏导数：f1A 对 P0
double WMPredict::f1PA(double P, double x, double fly_t, double distance,
                       double v0) {
  return -(v0 / k) * sin(P) * (exp(-k * fly_t) - 1);
}

// 关于 fly_t0 的偏导数：f1A 对 fly_t0
double WMPredict::f1tA(double P0, double delta_theta_delay,
                       const cv::Mat &world2car, double fly_t0, double distance,
                       double v0) {
  double m00 = world2car.at<double>(1, 0);
  double m01 = world2car.at<double>(1, 1);
  double m03 = world2car.at<double>(1, 3);

  double A0 = m00 * r * cos(delta_theta_delay + w * fly_t0) -
              m01 * r * sin(delta_theta_delay + w * fly_t0) + m03;
  double dA0_dt = -r * w *
                  (m00 * sin(delta_theta_delay + w * fly_t0) +
                   m01 * cos(delta_theta_delay + w * fly_t0));
  double dPart = -(A0 * dA0_dt) / sqrt(distance * distance - A0 * A0);
  double dB_dt = -v0 * cos(P0) * exp(-k * fly_t0);
  return dPart + dB_dt;
}

double WMPredict::f2PA(double P, double z, double fly_t, double v0) {
  return -(v0 * cos(P) * (1 - exp(-k * fly_t))) / k;
}

// 关于 fly_t0 的偏导数：f2A 对 fly_t0
double WMPredict::f2tA(double P0, double delta_theta_delay,
                       const cv::Mat &world2car, double fly_t0, double v0) {
  double m10 = world2car.at<double>(1, 0);
  double m11 = world2car.at<double>(1, 1);
  double m13 = world2car.at<double>(1, 3);

  double A1 = m10 * r * cos(delta_theta_delay + w * fly_t0) -
              m11 * r * sin(delta_theta_delay + w * fly_t0) + m13;
  double dA1_dt = -r * w *
                  (m10 * sin(delta_theta_delay + w * fly_t0) +
                   m11 * cos(delta_theta_delay + w * fly_t0));
  // L = k*v0*sin(P0) + g
  double L = k * v0 * sin(P0) + g;
  double dQ_dt = (L * exp(-k * fly_t0) - g) / k;
  return -dA1_dt + dQ_dt;
}

double WMPredict::F1(double P0, double fly_t0, double theta_0, double v0) {
  return sqrt(pow(r * cos(theta_0 + ThetaToolForBig(fly_t0, this->Fire_time)),
                  2) +
              s * s) -
         v0 * cos(P0) / k + v0 / k * cos(P0) * pow(Exp, -k * fly_t0);
}
double WMPredict::F2(double P0, double fly_t0, double theta_0, double v0) {
  return h0 + r * sin(theta_0 + ThetaToolForBig(fly_t0, this->Fire_time)) -
         (k * v0 * sin(P0) + g -
          (k * v0 * sin(P0) + g) * pow(Exp, -k * fly_t0) - g * k * fly_t0) /
             (k * k);
}
double WMPredict::F1P(double P, double fly_t, double theta_0,
                      double v0) // F1关于p的导数
{
  return v0 * sin(P) / k * (1 - pow(Exp, -k * fly_t));
}
double WMPredict::F1t(double P, double fly_t, double theta_0,
                      double v0) // f1关于t的导数
{
  return (-r * r * w * cos(theta_0 + ThetaToolForBig(fly_t, this->Fire_time)) *
          sin(theta_0 + ThetaToolForBig(fly_t, this->Fire_time))) /
             sqrt(
                 pow(r * cos(theta_0 + ThetaToolForBig(fly_t, this->Fire_time)),
                     2) +
                 s * s) -
         v0 * cos(P) * pow(Exp, -k * fly_t);
}
double WMPredict::F2P(double P, double fly_t, double theta_0,
                      double v0) // f2关于p的导数
{
  return v0 * cos(P) / k * (pow(Exp, -k * fly_t) - 1);
}
double WMPredict::F2t(double P, double fly_t, double theta_0,
                      double v0) // f2关于t的导数
{
  return w * r * cos(theta_0 + ThetaToolForBig(fly_t, this->Fire_time)) -
         (k * v0 * sin(P) + g) * pow(Exp, -k * fly_t) / k + g / k;
}

double WMPredict::F1A(double P0, double delta_theta_delay,
                      const cv::Mat &world2car, double fly_t0, double distance,
                      double v0) {
  cv::Mat world_point =
      (cv::Mat_<double>(4, 1)
           << r * cos(delta_theta_delay +
                      ThetaToolForBig(fly_t0, this->Fire_time)),
       -r * sin(delta_theta_delay + ThetaToolForBig(fly_t0, this->Fire_time)),
       0, 1.0);
  cv::Mat world_point_car = world2car * world_point;

  return sqrt(distance * distance - pow(world_point_car.at<double>(1, 0), 2)) -
         v0 * cos(P0) / k + v0 * cos(P0) * pow(Exp, -k * fly_t0) / k;
}
double WMPredict::F2A(double P0, double delta_theta_delay,
                      const cv::Mat &world2car, double fly_t0, double v0) {
  cv::Mat world_point =
      (cv::Mat_<double>(4, 1)
           << r * cos(delta_theta_delay +
                      ThetaToolForBig(fly_t0, this->Fire_time)),
       -r * sin(delta_theta_delay + ThetaToolForBig(fly_t0, this->Fire_time)),
       0, 1.0);
  cv::Mat world_point_car = world2car * world_point;

  return -world_point_car.at<double>(1, 0) - 0.4 -
         (k * v0 * sin(P0) + g -
          (k * v0 * sin(P0) + g) * pow(Exp, -k * fly_t0) - g * k * fly_t0) /
             (k * k);
}
double WMPredict::F1PA(double P, double x, double fly_t, double distance,
                       double v0) {
  return v0 * sin(P) / k * (1 - pow(Exp, -k * fly_t));
}

double WMPredict::F1tA(double P0, double delta_theta_delay,
                       const cv::Mat &world2car, double fly_t0, double distance,
                       double v0) {
  double M00 = world2car.at<double>(1, 0);
  double M01 = world2car.at<double>(1, 1);
  double M03 = world2car.at<double>(1, 3);

  double angle = delta_theta_delay + ThetaToolForBig(fly_t0, this->Fire_time);

  double x = r * M00 * cos(angle) - r * M01 * sin(angle) + M03;

  double x_dot = -w * r * (M00 * sin(angle) + M01 * cos(angle));

  double d_sqrt = -(x * x_dot) / sqrt(distance * distance - x * x);

  double d_exp = -v0 * cos(P0) * exp(-k * fly_t0);

  return d_sqrt + d_exp;
}

double WMPredict::F2PA(double P, double z, double fly_t, double v0) {
  return v0 * cos(P) / k * (pow(Exp, -k * fly_t) - 1);
}

double WMPredict::F2tA(double P0, double delta_theta_delay,
                       const cv::Mat &world2car, double fly_t0, double v0) {
  double M10 = world2car.at<double>(1, 0);
  double M11 = world2car.at<double>(1, 1);
  double M13 = world2car.at<double>(1, 3);

  double angle = delta_theta_delay + ThetaToolForBig(fly_t0, this->Fire_time);

  double d_first_term = w * r * (M10 * sin(angle) + M11 * cos(angle));

  double B = k * v0 * sin(P0) + g;
  double dQ = (B * exp(-k * fly_t0) - g) / k;

  return d_first_term - dQ;
}

cv::Mat WMPredict::GetDebugImg() { return this->debugImg; }
void WMPredict::GiveDebugImg(cv::Mat debugImg) { this->debugImg = debugImg; }
/**
 * @description: 弹速单位转化为m/s,同时判断弹速能否激活能量机关
 * @param {Translator} &translator
 * @param {GlobalParam} &gp
 * @return {*}
 */
int WMPredict::BulletSpeedProcess(Translator &translator, GlobalParam &gp) {
  translator.messageWM.bullet_v /= 10;
  // 弹速过小时无法命中
  if (translator.messageWM.bullet_v < gp.min_bullet_v) {
    // std::cout << "弹速小" << std::endl;
    // LOG_IF(INFO, gp.switch_INFO) << "弹速小，为：" <<
    // translator.messageWM.bullet_v;
    return 0;
  }
  return 1;
}

void WMPredict::CeresFitting(std::deque<double> time_data_queue,
                             std::deque<double> angle_data_queue,
                             double start_time) {
  double params[4] = {1.5, 1.5, 1.5, 0.0}; // 初始数值

  ceres::Problem problem;
  for (size_t j = 2; j < time_data_queue.size(); ++j) {
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<WindmillResidual, 1, 4>(
            new WindmillResidual(start_time, time_data_queue[j],
                                 angle_data_queue[j]));
    problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), params);
  }

  // 设定参数边界
  for (int i = 0; i < 4; i++) {
    problem.SetParameterLowerBound(params, i, 0);
    problem.SetParameterUpperBound(params, i, 3);
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 10000;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;

  // CustomTerminationCallback callback(params, true_A, true_w, true_fai,
  // true_A0); options.callbacks.push_back(&callback);
  // options.update_state_every_iteration = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  this->b = params[3];
  this->w_big = params[1];
  this->fai = params[2];
  this->A0 = params[0];
  std::cout << "A(b): " << this->b << " w_big: " << this->w_big
            << " fai: " << this->fai << " A0: " << this->A0 << std::endl;
}
