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

class Robot{
    public:
};

class Tracker{
    public:
    Tracker(GlobalParam &gp);
    private:
    std::vector<ExtendedKalmanFilter> kf_list;
    

};
#endif // ARMOR_PROCESSOR__TRACKER_HPP_
