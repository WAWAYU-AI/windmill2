#include "tracker.hpp"
#include "globalParam.hpp"

// STD

static inline double normalize_angle(double angle)
{
    const double result = fmod(angle + M_PI, 2.0 * M_PI);
    if (result <= 0.0)
        return result + M_PI;
    return result - M_PI;
}
double shortest_angular_distance(double from, double to)
{
    return normalize_angle(to - from);
}
Tracker::Tracker(double max_match_distance, double max_match_yaw_diff)
    : tracker_state(LOST),
      tracked_id(0),
      measurement(Eigen::VectorXd::Zero(4)),
      target_state(Eigen::VectorXd::Zero(9)),
      max_match_distance_(max_match_distance),
      max_match_yaw_diff_(max_match_yaw_diff)
{
}
// Tracker::Tracker(
//     double max_match_distance, int tracking_threshold, int lost_threshold,
//     Eigen::DiagonalMatrix<double, 8> q, Eigen::DiagonalMatrix<double, 4> r)
//     : tracker_state(LOST),
//     tracked_id(-1),
//     target_state(Eigen::VectorXd::Zero(8)),
//     max_match_distance_(max_match_distance),
//     tracking_threshold_(tracking_threshold),
//     lost_threshold_(lost_threshold)
// {
// }

void Tracker::init(Armors &armors_msg)
{
    if (armors_msg.armors.empty())
    {
        return;
    }

    // Simply choose the armor that is closest to image center
    double min_distance = DBL_MAX;
    tracked_armor = armors_msg.armors[0];
    for (const auto &armor : armors_msg.armors)
    {
        if (armor.distance_to_image_center < min_distance)
        {
            min_distance = armor.distance_to_image_center;
            tracked_armor = armor;
        }
    }

    initEKF(tracked_armor);

    tracked_id = tracked_armor.type;
    tracker_state = DETECTING;

    updateArmorsNum(tracked_armor);
}

// void Tracker::init(const Armors &armors_msg, double r)
// {
//     this->r = r;
//     if (armors_msg.armors.empty())
//     {
//         return;
//     }
//     tracked_armor = armors_msg.armors[0];
//     initEKF(tracked_armor);
//     tracked_id = tracked_armor.type;
//     tracker_state = DETECTING;
// }
void Tracker::check_data(cv::Mat &src)
{
    Eigen::VectorXd ratios = (target_state.array() / last_state.array()).abs();
    Eigen::VectorXd diff = (target_state.array() - last_state.array()).abs();
    // Eigen::VectorXd abs_ = target_state.array().abs();
    //  state: xc, v_xc, yc, v_yc, za, v_za, yaw, v_yaw, r
    std::vector<double> ratios_error_max =
        {10, 3, 10, 3, 10, 100, 10, 3, 5};
    std::vector<double> diff_error_max =
        {400, 200, 400, 200, 200, 200, 10, 6, 0.2};
    std::vector<double> abs_error_max =
        {DBL_MAX, 5e3, DBL_MAX, 5e3, 500, 5e2, 10.0, 15, DBL_MAX};
    for (int i = 0; i < ratios.size(); ++i)
    {
        bool ratios_abnormal = (ratios(i) > ratios_error_max[i]) or (ratios(i) < 1 / ratios_error_max[i]);
        bool diff_abnormal = (diff(i) > diff_error_max[i]);
        bool abs_abnormal = abs(target_state(i)) > abs_error_max[i];
        if ((ratios_abnormal and diff_abnormal) or abs_abnormal)
        {
            // target_state(i) = last_state(i);
            if (i == 7)
            {
                // target_state(i) = pow(target_state(i), 0.5);
            }
#ifdef DEBUGMODE
            cv::putText(src, std::to_string(i) + " abnormal in data", cv::Point(900, 100), 2, 2, cv::Scalar(0, 0, 255));
#endif
        }
    }
}
void Tracker::update(Armors &armors_msg, cv::Mat &src, int &real_id)
{
    // KF predict
    Eigen::VectorXd ekf_prediction = ekf.predict();

    bool matched = false;
    // Use KF prediction as default target state if no matched armor is found
    target_state = ekf_prediction;
    if (tracked_id == 7)
    {
        // target_state(8) = 0.35;
    }
    else if (tracked_id == 4)
    {
        // target_state(8) = .25;
    }
    this->check_data(src);
    if (!armors_msg.armors.empty())
    {
        // Find the closest armor with the same id
        Armor same_id_armor;
        int same_id_armors_count = 0;
        auto predicted_position = getArmorPositionFromState(ekf_prediction);
        double min_position_diff = DBL_MAX;
        double yaw_diff = DBL_MAX;
        for (auto &armor : armors_msg.armors)
        {
            if (tracked_id != real_id)
            {
                same_id_armors_count = 0;
                tracked_id = real_id;
            } // Only consider armors with the same id
            if (armor.type == tracked_id)
            {
                // real_id = tracked_id;
                same_id_armor = armor;
                same_id_armors_count++;
                // Calculate the difference between the predicted position and the current armor position
                auto p = armor.position;
                Eigen::Vector3d position_vec(p[0], p[1], p[2]);
                double position_diff = (predicted_position - position_vec).norm();
                if (position_diff < min_position_diff)
                {
                    // Find the closest armor
                    min_position_diff = position_diff;
                    yaw_diff = abs(orientationToYaw(armor.yaw) - ekf_prediction(6));
                    tracked_armor = armor;
                }
            }
        }
        // Store tracker info
        info_position_diff = min_position_diff;
        info_yaw_diff = yaw_diff;

        // Check if the distance and yaw difference of closest armor are within the threshold
        if (min_position_diff < max_match_distance_ && yaw_diff < max_match_yaw_diff_)
        {
            // Matched armor found
            matched = true;
            auto p = tracked_armor.position;
            // Update EKF
            double measured_yaw = orientationToYaw(tracked_armor.yaw);
            measurement = Eigen::Vector4d(p[0], p[1], p[2], measured_yaw);
            target_state = ekf.update(measurement);
        }
        else if (same_id_armors_count == 1 && yaw_diff > max_match_yaw_diff_)
        {
            // Matched armor not found, but there is only one armor with the same id
            // and yaw has jumped, take this case as the target is spinning and armor jumped
            handleArmorJump(same_id_armor);
        }
        else
        {
            // No matched armor found
        }
    }

    // Prevent radius from spreading
    if (target_state(8) < 0.2)
    {
        // printf("\033[41m\033[1;37mun expected r:%f\033[0m\n",target_state(8));
        target_state(8) = 0.2;
    }
    else if (target_state(8) > 0.45)
    {
        // printf("\033[41m\033[1;37mun expected r:%f\033[0m\n",target_state(8));
        target_state(8) = 0.45;
    }
    ekf.setState(target_state);

    // Tracking state machine
    if (tracker_state == DETECTING)
    {
        if (matched)
        {
            detect_count_++;
            if (detect_count_ > tracking_thres)
            {
                detect_count_ = 0;
                tracker_state = TRACKING;
            }
        }
        else
        {
            detect_count_ = 0;
            tracker_state = LOST;
        }
    }
    else if (tracker_state == TRACKING)
    {
        if (!matched)
        {
            tracker_state = TEMP_LOST;
            lost_count_++;
        }
    }
    else if (tracker_state == TEMP_LOST)
    {
        if (!matched)
        {
            lost_count_++;
            if (lost_count_ > lost_thres)
            {
                lost_count_ = 0;
                tracker_state = LOST;
            }
        }
        else
        {
            tracker_state = TRACKING;
            lost_count_ = 0;
        }
    }
    last_state = target_state;
}

void Tracker::initEKF(Armor &a)
{
    double xa = a.position[0];
    double ya = a.position[1];
    double za = a.position[2];
    last_yaw_ = 0;
    double yaw = orientationToYaw(a.yaw);

    // Set initial position at 0.2m behind the target
    last_state = target_state = Eigen::VectorXd::Ones(9) * 1e-5;
    double r = 0.28;
    double xc = xa + r * cos(yaw);
    double yc = ya + r * sin(yaw);
    dz = 0, another_r = r;
    target_state << xc, 0, yc, 0, za, 0, yaw, 0, r;

    ekf.setState(target_state);
}

void Tracker::updateArmorsNum(Armor &armor)
{
    if (tracked_id == 9 || tracked_id == 0 || tracked_id == 10)
    {
        tracked_armors_num = ArmorsNum::BALANCE_2;
    }
    else if (tracked_id == 0)
    {
        tracked_armors_num = ArmorsNum::OUTPOST_3;
    }
    else
    {
        tracked_armors_num = ArmorsNum::NORMAL_4;
    }
}

void Tracker::handleArmorJump(Armor &current_armor)
{
    double yaw = orientationToYaw(current_armor.yaw);
    target_state(6) = yaw;
    updateArmorsNum(current_armor);
    // Only 4 armors has 2 radius and height
    if (tracked_armors_num == ArmorsNum::NORMAL_4)
    {
        // std::cout << "4 armors and solved" << std::endl;
        dz = target_state(4) - current_armor.position[2];
        target_state(4) = current_armor.position[2];
        // std::swap(target_state(8), another_r);
    }

    // If position difference is larger than max_match_distance_,
    // take this case as the ekf diverged, reset the state
    auto p = current_armor.position;
    Eigen::Vector3d current_p(p[0], p[1], p[2]);
    Eigen::Vector3d infer_p = getArmorPositionFromState(target_state);
    if ((current_p - infer_p).norm() > max_match_distance_)
    {
        double r = target_state(8);
        target_state(0) = p[0] + r * cos(yaw); // xc
        target_state(1) = 0;                   // vxc
        target_state(2) = p[1] + r * sin(yaw); // yc
        target_state(3) = 0;                   // vyc
        target_state(4) = p[2];                // za
        target_state(5) = 0;                   // vza
                                               // RCLCPP_ERROR(rclcpp::get_logger("armor_tracker"), "Reset State!");
    }

    ekf.setState(target_state);
}

Eigen::Vector3d Tracker::getArmorPositionFromState(Eigen::VectorXd &x)
{
    // Calculate predicted position of the current armor
    double xc = x(0), yc = x(2), za = x(4);
    double yaw = x(6), r = x(8);
    double xa = xc - r * cos(yaw);
    double ya = yc - r * sin(yaw);
    return Eigen::Vector3d(xa, ya, za);
}
double Tracker::orientationToYaw(double &yaw)
{
    // Get armor yaw
    //   tf2::Quaternion tf_q;
    //   tf2::fromMsg(q, tf_q);
    //   double roll, pitch, yaw;
    //   tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
    // Make yaw change continuous (-pi~pi to -inf~inf)
    yaw = last_yaw_ + shortest_angular_distance(last_yaw_, yaw);
    last_yaw_ = yaw;
    return yaw;
}