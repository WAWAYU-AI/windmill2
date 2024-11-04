#ifndef ARMOR_PROCESSOR__TRACKER_HPP_
#define ARMOR_PROCESSOR__TRACKER_HPP_

// Eigen
#include <Eigen/Eigen>

// STD
#include "KalmanFilter.hpp"
#include "globalParam.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <cfloat>
#include <deque>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

enum class ArmorsNum { NORMAL_4 = 4, BALANCE_2 = 2, OUTPOST_3 = 3 };

class Tracker
{
public:
  Tracker(double max_match_distance, double max_match_yaw_diff);

  void init(Armors &armors_msg);

  void update(Armors &armors_msg,cv::Mat &src,int &);

  ExtendedKalmanFilter ekf;

  int tracking_thres;
  int lost_thres;

  enum State {
    LOST,
    DETECTING,
    TRACKING,
    TEMP_LOST,
  } tracker_state;

  int tracked_id;
  Armor tracked_armor;
  ArmorsNum tracked_armors_num;

  double info_position_diff;
  double info_yaw_diff;

  Eigen::VectorXd measurement;

  Eigen::VectorXd target_state;
  Eigen::VectorXd last_state;

  // To store another pair of armors message
  double dz, another_r;

private:
  void initEKF(Armor & a);
  void check_data(cv::Mat &src);
  void updateArmorsNum(Armor & a);

  void handleArmorJump(Armor & a);
  double orientationToYaw(double & yaw);

//   double orientationToYaw(const geometry_msgs::msg::Quaternion & q);

  Eigen::Vector3d getArmorPositionFromState(Eigen::VectorXd & x);

  double max_match_distance_;
  double max_match_yaw_diff_;

  int detect_count_;
  int lost_count_;

  double last_yaw_;
};

// class Tracker
// {
// public:
//     Tracker(
//         double max_match_distance, int tracking_threshold, int lost_threshold,
//         Eigen::DiagonalMatrix<double, 8> q, Eigen::DiagonalMatrix<double, 4> r);

//     void init(const Armors &armors_msg, double r);
//     void update(const Armors &armors_msg, cv::Mat &src, double r);
//     enum State
//     {
//         LOST,
//         DETECTING,
//         TRACKING,
//         TEMP_LOST,
//     } tracker_state;
//     ExtendedKalmanFilter ekf;
//     Armor tracked_armor;
//     int tracked_id;
//     Eigen::VectorXd target_state;

//     double last_z, last_r, r;

// private:
//     void initEKF(const Armor &a);

//     void handleArmorJump(const Armor &a);
//     // void tmp_draw(cv::Mat &src, std::string text);
//     Eigen::Vector3d getArmorPositionFromState(const Eigen::VectorXd &x);

//     double max_match_distance_;

//     int tracking_threshold_;
//     int lost_threshold_;

//     int detect_count_;
//     int lost_count_;

// #ifdef DEBUGMODE
//     cv::Mat tmp_mat;
// #endif
// };

#endif // ARMOR_PROCESSOR__TRACKER_HPP_
