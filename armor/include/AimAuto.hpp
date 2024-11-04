#ifndef AIMAUTO
#define AIMAUTO
#include "PredictShow.hpp"
// #include "fftw_omega.hpp"
#include "gaoning.hpp"
#include "globalParam.hpp"
#include "globalText.hpp"
#include "opencv2/core/mat.hpp"
#include "tracker.hpp"
#include <camera.hpp>
#include <chrono>
#include <cstdint>
#include <detector.hpp>
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/video.hpp>
#include <vector>
class AimAuto
{
private:
    // ===Global Info Variables=== //
    GlobalParam *gp;         // Use for reading params from config file
    address addr;            // Use for reading params from config file
    gaoning gn;              // Some tweaks written by GaoNing
    bulletToGaoNing_t timer; // Use for cal `AimAuto::flt`

    // Use for pnp_solve
    cv::Mat tVec, rVec;
    std::vector<float> _dist;
    cv::Mat _K;
    cv::Mat empty;

    // Used for reprojection and image display
    Eigen::MatrixXd _K_;

    // Other
    cv::Size src_size;
    double dt;
    double r;
    double react_time_[4];
    int tracking_numb;

    // ===Predict and FireControl Variables=== //

    cv::Point3f position_save; // almost unused
    cv::Point pixel_center;    // almost unused

    double flt;         // bullet's flight time
    double vyaw_zoom;   // use for vyaw control
    double v_zoom;      // use for translational velocity control
    double rawOmega;    // Original angular velocity (Radian)
    double isClockwise; // -1.0 for not clockwise and +1.0 for clockwise
    int position_gap;   // use for choose the armor,almost unused

    int rotate_speed_level;        // 3-levels,larger for higher speed
    int translational_speed_level; // 2-levels,larger for higher speed
    int choose_ycr;                /*(Related to `AimAuto::y_camera_reserve`)
                                    -1 is use Kalmen data,0 is choose left,1 is choose right*/

    // ===Logic Control Variables=== //
    double last_time, time, restart_time;
    double first_see_time;
    bool first_see;
    std::vector<bool> isBigArmor; // only have one value

    // ===Kalmen Filter Variables=== //
    // double r_xyz_factor;
    // double r_yaw;
    // double s2qxyz_;
    // double s2qyaw_;
    // double s2qr_;
    struct TrackerConfigs
    {
        double max_match_distance;
        double max_match_yaw_diff;
        double r_xyz_factor;
        double r_yaw;
        double s2qxyz_;
        double s2qyaw_;
        double s2qr_;
        TrackerConfigs() {}
        TrackerConfigs(double t1, double t2, double t3, double t4, double t5, double t6, double t7) : max_match_distance(t1), max_match_yaw_diff(t2), r_xyz_factor(t3), r_yaw(t4), s2qxyz_(t5), s2qyaw_(t6), s2qr_(t7) {}
    };
    TrackerConfigs config0;
    TrackerConfigs config1;
    double time_add;

    // ===Global Data Containers=== //
    Tracker *tracker_0;
    Tracker *tracker_1;
    Armors armors_msg;
    std::deque<Armor> tar_list;
    std::unique_ptr<rm_auto_aim::Detector> det;

    // cal ArmorSpeed
    cv::Point3f current_world_state;
    cv::Point3f last_world_state;
    std::deque<double> ArmorSpeed_array; //

    // auto zone choose
    std::deque<int> PredPos_array;       //
    std::vector<int> PredPosIndex_array; //

    // cal speed level
    std::deque<int> rotateLevelCount; // related to `AimAuto::rotate_speed_level`
    std::deque<int> transLevelCount;  // related to `AimAuto::translational_speed_level`
    int level_count[3];               // use for show level_count

    // use for y override
    std::vector<std::pair<cv::Point3f, int>> y_camera_reserve; // reserve data from camera_axis,against the error from process of converting from the world_axis to the camera_axis
    std::vector<cv::Point3f> last_ycr;
    bool use_ycr;

    // ===The following are all unfinished contents=== //
    // use for high speed hitting
    std::vector<cv::Point3f> closest2Armors;
    // std::deque<double> shootCenterArray;
    std::deque<double> shootCenterLeftArray;
    std::deque<double> shootCenterRightArray;
    // std::vector<double> shootCenterLeftArray;
    // std::vector<double> shootCenterRightArray;
    std::deque<double> filteredVyawArray;
    std::deque<double> yReserveArray;
    // Use for Posterior Estimation
    uint8_t shootState;
    struct predictPoint
    {
        cv::Point3f point;
        float yaw_a;
        double time_second;
        predictPoint(cv::Point3f &p, double t, float yaw)
            : point(p), time_second(t), yaw_a(yaw){};
    };
    float yaw_a_now;
    double receive_second;
    double shootCenterBias = 0.0;
    std::deque<predictPoint> predictPoints;
    void posteriorEstimation(Translator &ts, cv::Mat &src, float y);
    int fireTimes;

    // use for shoot keeper
    int shootKeeper;
    Translator last_ts;

    // ===Special Reserved Variables=== //
    double raw_yaw;

    // ===Time Recorders=== //
    std::chrono::milliseconds last_time_;

    // hitting outposed
    double last_yaw;
    bool is_hitting_outpose;
    std::deque<double> rawYawArray;

    // ===Unused Variables=== //
    // double cnt;
    // int count;
    // double last_y;
    // double last_yaw;
    // double forward_save;
    // fftw_omega omega_filter;
    // PredictShow preshow;
    Eigen::Vector3d predict_center;
    double time_save;

    void pnp_solve(rm_auto_aim::Armor &armor, Translator &ts, cv::Mat &src, Armor &tar, int number);
    cv::Point3f findTarget(Translator &ts, double &time_add, cv::Mat &src);
    cv::Point3f findTarget(Translator &ts, double pz, double pyaw, double &real_time, cv::Mat &src);
    void storeMessage(cv::Point3f target, Translator &ts);
    void convertPoint(Translator &ts, cv::Point3f target, bool, cv::Mat &);
    void showDist(std::deque<Armor> &tar_list, cv::Mat &src);
    bool updateTracker(Translator &ts, cv::Mat &src);
    void put_state(Translator &ts, cv::Mat &src);

    void fireControl(Translator &ts, cv::Mat &src);
    void fireControl(Translator &ts0, Translator &ts1, cv::Mat &src);
    double updateArmorSpeed(cv::Mat &src, Translator &ts);
    double cal_ArmorSpeed(cv::Mat &src);
    void camera2plot(Eigen::Vector3d &point, cv::Mat &src);
    Eigen::Vector3d world2camera(std::vector<double> &input, Translator &ts);
    Eigen::Vector3d world2camera(cv::Point3f input, Translator &ts, bool to_shoot);
    cv::Point camera2pixel(Eigen::Vector3d &point);
    int autoZoneChoose(cv::Mat &src, std::vector<double> &zone_, float value, double center);
    void getParam();

public:
    AimAuto(GlobalParam *gp);
    ~AimAuto();
    void AimAutoYHY(cv::Mat &src, Translator &ts);
    void NewTracker(Translator &ts, cv::Mat &src);
    void setTime(double t);

    int z_len = 50;
    double *z_list;

    // ===Unused Variables=== //
    // int z_p = 0, z_q = 0;
    // bool z_isfull = false;
    // int count_z = 0;
    // double z_mean, z_total = 0;
    // cv::Mat empty;
    // std::vector<double> last_p_world;
};
std::unique_ptr<rm_auto_aim::Detector> initDetector(int color);
#endif // AIMAUTO