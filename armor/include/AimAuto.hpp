#ifndef AIMAUTO
#define AIMAUTO
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
    GlobalParam *gp;
    Detector *detector;
    Tracker *tracker;
    void pnp_solve(UnsolvedArmor &armor, Translator &ts, cv::Mat &src, Armor &tar, int number);
    void draw_armor_back(cv::Mat &src, Armor &armor, int number);
public:
    AimAuto(GlobalParam *gp);
    ~AimAuto();
    void auto_aim(cv::Mat &src, Translator &ts, double dt);
};
std::unique_ptr<Detector> initDetector(int color);
#endif // AIMAUTO