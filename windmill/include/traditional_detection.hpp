#ifndef TRADITIONAL_DETECTION_HPP
#define TRADITIONAL_DETECTION_HPP

#include "WMIdentify.hpp"
#include "globalParam.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

struct DetectionResult {
  std::vector<cv::Point> intersections;
  std::vector<cv::Point> circlePoints;
  cv::Mat processedImage;
  double processingTime;
};

struct KeyPoints {
  std::vector<std::vector<cv::Point>> circleContours;
  std::vector<double> circleAreas;
  std::vector<double> circularities;
  std::vector<cv::Point2f> rectCenters;
  std::vector<cv::Point> circlePoints;
  // 定义面积范围常量
  static constexpr double min_low = 150.0;
  static constexpr double min_high = 1800.0;
  static constexpr double max_low = 2500.0;
  static constexpr double max_high = 15000.0;

  bool isValid() const {
    // 我们只检查数量。面积的判断已经由更智能的 detect 函数完成。
    // 这可以避免因为面积的微小抖动导致 isValid() 意外失败。
    if (circleContours.size() == 2 && rectCenters.size() == 1) {
      return true;
    }
    return false;
  }
};

DetectionResult detect(const cv::Mat &inputImage, WMBlade &blade,
                       GlobalParam &gp, int is_blue, Translator &translator);

KeyPoints detect_key_points(const std::vector<std::vector<cv::Point>> &contours,
                            const std::vector<cv::Vec4i> &hierarchy,
                            cv::Mat &processedImage, WMBlade &blade,
                            GlobalParam &gp,
                            Translator &translator);

std::vector<cv::Point>
findIntersectionsByEquation(const cv::Point &center1, const cv::Point &center2,
                            double radius, const cv::RotatedRect &ellipse,
                            cv::Mat &pic, GlobalParam &gp, WMBlade &blade);

#endif // TRADITIONAL_DETECTION_HPP
