/**
 * @file traditional_detection.cpp
 * @author Clarence Stark (3038736583@qq.com)
 * @brief 用于传统算法检测
 * @version 0.1
 * @date 2025-01-04
 *
 * @copyright Copyright (c) 2025
 */

#include "opencv2/core/types.hpp"
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <traditional_detection.hpp>

using namespace cv;
using namespace std;
using namespace std::chrono;
// 计算两点点距
double computeDistance(Point p1, Point p2) {
  return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

/**
 * @brief 将图像进行偏航角度的透视变换
 * @param inputImage 输入图像
 * @param yawFactor 偏航角度因子
 * @return 变换后的图像
 */
cv::Mat applyYawPerspectiveTransform(const cv::Mat &inputImage,
                                     float yawFactor) {
  // 检查输入图像是否为空
  if (inputImage.empty()) {
    std::cerr << "输入图像为空！" << std::endl;
    return cv::Mat();
  }

  int rows = inputImage.rows;
  int cols = inputImage.cols;

  std::vector<cv::Point2f> pts1 = {cv::Point2f(0, 0), cv::Point2f(cols, 0),
                                   cv::Point2f(0, rows),
                                   cv::Point2f(cols, rows)};

  float horizontalOffset = cols * yawFactor; // 根据输入的因子计算水平偏移量
  std::vector<cv::Point2f> pts2 = {
      cv::Point2f(horizontalOffset, 0), cv::Point2f(cols - horizontalOffset, 0),
      cv::Point2f(horizontalOffset / 2, rows),
      cv::Point2f(cols - horizontalOffset / 2, rows)};

  cv::Mat M = cv::getPerspectiveTransform(pts1, pts2);

  cv::Mat warpedImage;
  cv::warpPerspective(inputImage, warpedImage, M, inputImage.size());

  return warpedImage;
}

const string WINDOW_NAME = "Parameter Controls";
int circularityThreshold = 60; // 圆度阈值
int medianBlurSize = 3;        // 中值滤波核大小

bool debug = false;        // debug模式
bool useTrackbars = debug; // 是否使用滑动条动态调参
int dilationSize = 9;      // 膨胀核大小
int erosionSize = 5;       // 腐蚀核大小
int thresholdValue = 110;  // 二值化阈值

int rect_area_threshold = 2000;  // 矩形面积阈值
int circle_area_threshold = 100; // 类圆轮廓面积阈值

int length_width_ratio_threshold = 3; // 长宽比阈值

int minContourArea = 400; // 最小轮廓面积

void createTrackbars() {
  namedWindow(WINDOW_NAME, WINDOW_NORMAL);
  moveWindow(WINDOW_NAME, 0, 0);
  createTrackbar("Circularity", WINDOW_NAME, &circularityThreshold, 100,
                 nullptr);
  createTrackbar("Dilation Size", WINDOW_NAME, &dilationSize, 21, nullptr);
  createTrackbar("Erosion Size", WINDOW_NAME, &erosionSize, 21, nullptr);
  createTrackbar("Blur Size", WINDOW_NAME, &medianBlurSize, 21, nullptr);

  createTrackbar("Rect Area Threshold", WINDOW_NAME, &rect_area_threshold, 1000,
                 nullptr);
  createTrackbar("Min Contour Area", WINDOW_NAME, &minContourArea, 1000,
                 nullptr);
  createTrackbar("Circle Area Threshold", WINDOW_NAME, &circle_area_threshold,
                 1000, nullptr);
  createTrackbar("Length Width Ratio Threshold", WINDOW_NAME,
                 &length_width_ratio_threshold, 10, nullptr);
  createTrackbar("Threshold", WINDOW_NAME, &thresholdValue, 255, nullptr);
}

/**
 * @brief 图像预处理函数
 * @param inputImage 输入图像
 * @param debug 是否显示中间步骤
 * @return 预处理后的掩码图像
 */
cv::Mat preprocess(const cv::Mat &inputImage, bool debug) {
  if (useTrackbars) {
    static bool trackbarsInitialized = false;
    if (!trackbarsInitialized) {
      createTrackbars();
      trackbarsInitialized = true;
    }
  }
  cv::Mat final_mask;

  std::vector<cv::Mat> channels;
  cv::split(inputImage, channels);

  cv::Mat blue = channels[0];
  cv::Mat red = channels[2];

  // 通道相减得到灰度图
  Mat temp;
  subtract(red, blue, temp);
  // 确保medianBlurSize是奇数
  int kernelSize = medianBlurSize;
  if (kernelSize % 2 == 0) {
    kernelSize++;
  }
  cv::medianBlur(temp, temp, kernelSize);
  //  cv::imshow("medianBlur", temp);

  // 对灰度图进行二值化
  threshold(temp, final_mask, thresholdValue, 255, THRESH_BINARY);

  Mat kernel1 =
      getStructuringElement(MORPH_RECT, Size(dilationSize, dilationSize));
  Mat kernel2 =
      getStructuringElement(MORPH_RECT, Size(erosionSize, erosionSize));

  dilate(final_mask, final_mask, kernel1);
  erode(final_mask, final_mask, kernel2);
  return final_mask;
}

/**
 * @brief 检测关键点（R标，扇叶中心点，流水灯条中心点）
 * @param contours 轮廓
 * @param hierarchy 轮廓层次
 * @param processedImage 处理后的图像
 * @return 关键点结果
 */
KeyPoints detect_key_points(const vector<vector<Point>> &contours,
                            const vector<Vec4i> &hierarchy, Mat &processedImage,
                            WMBlade &blade) {
  KeyPoints result;
  // 用于标记已被识别为矩形轮廓的索引
  vector<bool> isRectContour(contours.size(), false);

  // 处理每个轮廓
  for (int i = 0; i < contours.size(); i++) {
    const auto &contour = contours[i];
    if (contourArea(contour) < minContourArea) {
      //  cv::drawContours(processedImage, contours, i, Scalar(0, 0, 255), 2);
      //  cv::imshow("processedImage", processedImage);
      continue;
    }
    if (hierarchy[i][3] != -1) {
      continue;
    }
    double area = contourArea(contour);
    if (area > rect_area_threshold) {
      RotatedRect rect = minAreaRect(contour);
      float width = rect.size.width;
      float height = rect.size.height;

      float aspectRatio = width > height ? width / height : height / width;

      if (aspectRatio > length_width_ratio_threshold) {
        isRectContour[i] = true;

        Moments m = moments(contour);
        result.rectCenters.push_back(Point(m.m10 / m.m00, m.m01 / m.m00));
        circle(processedImage, result.rectCenters[0], 3, Scalar(0, 255, 0), -1);
      }
    }

    if (isRectContour[i]) {
      continue;
    }

    if (area > circle_area_threshold) {
      double circularity =
          4 * CV_PI * area / (pow(arcLength(contour, true), 2));

      if (circularity > (circularityThreshold / 100.0)) {
        // 计算子轮廓数量
        int childCount = 0;
        int firstChild = hierarchy[i][2];
        if (firstChild >= 0) {
          childCount = 1;                           // 至少有一个子轮廓
          int nextChild = hierarchy[firstChild][0]; // 下一个同级轮廓
          while (nextChild >= 0 && nextChild != firstChild) {
            childCount++;
            nextChild = hierarchy[nextChild][0];
          }
        }
        if (childCount > 1 || childCount == 0 && hierarchy[i][3] == -1) {
          result.circleContours.push_back(contour);
          result.circleAreas.push_back(area);
          result.circularities.push_back(circularity);

          Moments m = moments(contour);
          Point circleCenter(int(m.m10 / m.m00), int(m.m01 / m.m00));
          result.circlePoints.push_back(circleCenter);
        }
      }
    }
  }

  vector<bool> filteredRectContour = isRectContour;

  if (result.rectCenters.size() > 1) {
    size_t maxCircleIdx = 0;
    double maxArea = 0;
    for (size_t i = 0; i < result.circleAreas.size(); i++) {
      if (result.circleAreas[i] > maxArea) {
        maxArea = result.circleAreas[i];
        maxCircleIdx = i;
      }
    }

    if (!result.circlePoints.empty()) {
      cv::Point maxCircleCenter = result.circlePoints[maxCircleIdx];

      std::vector<double> distances;
      for (const auto &rectCenter : result.rectCenters) {
        double dist = computeDistance(rectCenter, maxCircleCenter);
        distances.push_back(dist);
      }

      size_t minDistIdx = std::min_element(distances.begin(), distances.end()) -
                          distances.begin();

      cv::Point2f closestRectCenter = result.rectCenters[minDistIdx];

      std::fill(filteredRectContour.begin(), filteredRectContour.end(), false);

      result.rectCenters.clear();
      result.rectCenters.push_back(closestRectCenter);

      for (int i = 0; i < contours.size(); i++) {
        if (isRectContour[i]) {
          Moments m = moments(contours[i]);
          Point center(m.m10 / m.m00, m.m01 / m.m00);

          if (computeDistance(center, closestRectCenter) < 5.0) {
            filteredRectContour[i] = true;
          }
        }
      }

      if (debug) {
        std::cout << "保留距离最近的矩形轮廓，距离为: " << distances[minDistIdx]
                  << std::endl;
        std::cout << "筛选后result.rectCenters.size() = "
                  << result.rectCenters.size() << std::endl;
      }
    }
  }

  if (result.circleContours.size() > 2) {
    size_t maxCircleIdx = 0;
    double maxArea = 0;
    for (size_t i = 0; i < result.circleAreas.size(); i++) {
      if (result.circleAreas[i] > maxArea) {
        maxArea = result.circleAreas[i];
        maxCircleIdx = i;
      }
    }

    if (!result.rectCenters.empty()) {
      cv::Point rectCenter(result.rectCenters[0].x, result.rectCenters[0].y);

      std::vector<std::pair<size_t, double>> idxDistPairs;
      for (size_t i = 0; i < result.circlePoints.size(); i++) {
        if (i != maxCircleIdx) {
          double dist = computeDistance(result.circlePoints[i], rectCenter);
          idxDistPairs.push_back({i, dist});
        }
      }

      // 按距离排序
      std::sort(
          idxDistPairs.begin(), idxDistPairs.end(),
          [](const auto &a, const auto &b) { return a.second < b.second; });

      size_t closestCircleIdx = idxDistPairs[0].first;

      std::vector<std::vector<cv::Point>> newCircleContours;
      std::vector<double> newCircleAreas;
      std::vector<double> newCircularities;
      std::vector<cv::Point> newCirclePoints;

      newCircleContours.push_back(result.circleContours[maxCircleIdx]);
      newCircleAreas.push_back(result.circleAreas[maxCircleIdx]);
      newCircularities.push_back(result.circularities[maxCircleIdx]);
      newCirclePoints.push_back(result.circlePoints[maxCircleIdx]);

      newCircleContours.push_back(result.circleContours[closestCircleIdx]);
      newCircleAreas.push_back(result.circleAreas[closestCircleIdx]);
      newCircularities.push_back(result.circularities[closestCircleIdx]);
      newCirclePoints.push_back(result.circlePoints[closestCircleIdx]);

      result.circleContours = newCircleContours;
      result.circleAreas = newCircleAreas;
      result.circularities = newCircularities;
      result.circlePoints = newCirclePoints;

      if (debug) {
        std::cout << "保留面积最大的圆轮廓和距离最近的圆轮廓，距离为: "
                  << idxDistPairs[0].second << std::endl;
      }
    }
  }

  // 可视化代码
  if (debug) {
    Mat visualImage = processedImage.clone();

    for (size_t i = 0; i < result.circleContours.size(); i++) {
      drawContours(visualImage, result.circleContours, i, Scalar(0, 255, 0), 2);

      circle(visualImage, result.circlePoints[i], 5, Scalar(0, 255, 0), -1);

      putText(visualImage,
              "Area: " + to_string(static_cast<int>(result.circleAreas[i])) +
                  " Circ: " + to_string(result.circularities[i]).substr(0, 4),
              result.circlePoints[i] + Point(10, 10), FONT_HERSHEY_SIMPLEX, 0.5,
              Scalar(255, 255, 0), 1);
    }

    for (size_t i = 0; i < contours.size(); i++) {
      if (filteredRectContour[i]) { // 使用过滤后的标记
        drawContours(visualImage, contours, i, Scalar(0, 255, 0), 10);

        RotatedRect rotRect = minAreaRect(contours[i]);
        float width = rotRect.size.width;
        float height = rotRect.size.height;
        float aspectRatio = width > height ? width / height : height / width;
        float area = contourArea(contours[i]);

        Point center;
        if (i < result.rectCenters.size()) {
          center = result.rectCenters[i];
        } else {
          Moments m = moments(contours[i]);
          center = Point(int(m.m10 / m.m00), int(m.m01 / m.m00));
        }
        circle(visualImage, center, 8, Scalar(0, 255, 255), -1);

        putText(visualImage, "Rect", center + Point(10, -10),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

        putText(visualImage,
                "Ratio: " + to_string(aspectRatio).substr(0, 4) +
                    " Area: " + to_string(static_cast<int>(area)),
                center + Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5,
                Scalar(0, 255, 255), 1);
      }
    }

    imshow("Detected Key Points", visualImage);
    cv::waitKey(1);
  }

  return result;
}

DetectionResult detect(const cv::Mat &inputImage, WMBlade &blade) {

  DetectionResult result;
  auto start_time = high_resolution_clock::now();

  // 预处理
  Mat final_mask = preprocess(inputImage, debug);
  if (debug) {
    imshow("final_mask", final_mask);
  }

  // 轮廓分析阶段,提取能量机关扇叶中心点，流水灯条中心点以及R标
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(final_mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

  // 创建输入图像的副本
  Mat processedImage = inputImage;

  KeyPoints keyPoints =
      detect_key_points(contours, hierarchy, processedImage, blade);
  if (!keyPoints.isValid()) {
    std::cout << "检测失败!" << std::endl;
    std::cout << "keyPoints.isValid() = " << keyPoints.circleContours.size()
              << std::endl;
    std::cout << "keyPoints.isValid() = " << keyPoints.rectCenters.size()
              << std::endl;

    // 处理检测失败的情况
    return DetectionResult();
  }

  // 按面积从大到小排序类圆轮廓
  vector<size_t> indices(keyPoints.circleContours.size());
  iota(indices.begin(), indices.end(), 0);
  sort(indices.begin(), indices.end(), [&keyPoints](size_t i1, size_t i2) {
    return keyPoints.circleAreas[i1] > keyPoints.circleAreas[i2];
  });

  // 交点计算

  blade.apex.push_back(keyPoints.circlePoints[indices[1]]);
  blade.apex.push_back(keyPoints.circlePoints[indices[0]]);
  // cv::circle(processedImage, blade.apex[0], 3, Scalar(0, 255, 0), -1);
  Moments m1 = moments(keyPoints.circleContours[indices[0]]);
  // 拟合椭圆轮廓
  cv::RotatedRect ellipse =
      cv::fitEllipse(keyPoints.circleContours[indices[0]]);
  // 使用椭圆中心代替矩计算的中心点
  Point center1(ellipse.center.x, ellipse.center.y);
  double radius = sqrt(keyPoints.circleAreas[indices[0]] / CV_PI);

  result.intersections =
      findIntersectionsByEquation(center1, keyPoints.rectCenters[0], radius,
                                  ellipse, processedImage, debug, blade);
  if (!result.intersections.empty()) {
    // ROI 2
    int x2 = result.intersections[result.intersections.size() - 1].x - 100;
    int y2 = result.intersections[result.intersections.size() - 1].y - 100;
    int width2 = 200;
    int height2 = 200;

    // 确保ROI不会超出图像边界
    x2 = std::max(0, std::min(x2, processedImage.cols - width2));
    y2 = std::max(0, std::min(y2, processedImage.rows - height2));

    // 调整width和height以确保不会超出图像边界
    width2 = std::min(width2, processedImage.cols - x2);
    height2 = std::min(height2, processedImage.rows - y2);

    if (width2 > 0 && height2 > 0) { // 确保ROI区域有效
      cv::Rect roi2(x2, y2, width2, height2);
      cv::Mat roi2_img = processedImage(roi2).clone();

      // cv::imshow("roi2", processedImage(roi2));
    }

    // ROI 3
    int x3 = result.intersections[result.intersections.size() - 2].x - 20;
    int y3 = result.intersections[result.intersections.size() - 2].y - 20;
    int width3 = 40;
    int height3 = 40;

    // 确保ROI不会超出图像边界
    x3 = std::max(0, std::min(x3, processedImage.cols - width3));
    y3 = std::max(0, std::min(y3, processedImage.rows - height3));

    // 调整width和height以确保不会超出图像边界
    width3 = std::min(width3, processedImage.cols - x3);
    height3 = std::min(height3, processedImage.rows - y3);
  }

  result.processedImage = processedImage; // 处理后的图像

  // 计算处理时间
  auto end_time = high_resolution_clock::now();
  result.processingTime =
      duration_cast<milliseconds>(end_time - start_time).count();
  blade.apex.push_back(keyPoints.rectCenters[0]);

  return result;
}

// 通过方程求解交点的方法
vector<Point> findIntersectionsByEquation(const Point &center1,
                                          const Point &center2, double radius,
                                          const RotatedRect &ellipse, Mat &pic,
                                          bool debug, WMBlade &blade) {
  vector<Point> intersections;

  // 获取椭圆参数
  Point2f ellipse_center = ellipse.center;
  Size2f size = ellipse.size;
  float angle_deg = ellipse.angle;              // 旋转角度（度）
  double angle_rad = angle_deg * CV_PI / 180.0; // 旋转角度（弧度）

  // 将椭圆缩放0.9倍 (考虑灯条粗细需要缩放让该椭圆方程能够拟合到灯条中心)
  double scale = 0.9;
  // 半长轴和半短轴缩放
  double a = (size.width / 2.0) * scale;
  double b = (size.height / 2.0) * scale;

  // 计算第一条直线的系数 A x + B y + C = 0
  double A = center2.y - center1.y;
  double B = center1.x - center2.x;
  double C = center2.x * center1.y - center1.x * center2.y;

  // 将直线方程旋转到椭圆的坐标系
  double cos_theta = cos(angle_rad);
  double sin_theta = sin(angle_rad);

  double A_rot = A * cos_theta + B * sin_theta;
  double B_rot = -A * sin_theta + B * cos_theta;
  double C_rot = C + A * ellipse_center.x + B * ellipse_center.y;

  // 避免除零的情况
  if (fabs(B_rot) < 1e-8) {
    cout << "直线几乎垂直" << endl;
    // 如果需要，可添加额外的逻辑来处理这种直线几乎垂直的情况
  }

  // 计算二次方程系数
  double M = (1.0 / (a * a)) + (A_rot * A_rot) / (B_rot * B_rot * b * b);
  double N = (2.0 * A_rot * C_rot) / (B_rot * B_rot * b * b);
  double P = (C_rot * C_rot) / (B_rot * B_rot * b * b) - 1.0;

  // delta
  double discriminant = N * N - 4.0 * M * P;

  if (discriminant >= 0) {
    double sqrt_discriminant = sqrt(discriminant);
    double x1_rot = (-N + sqrt_discriminant) / (2.0 * M);
    double x2_rot = (-N - sqrt_discriminant) / (2.0 * M);

    double y1_rot = (-A_rot * x1_rot - C_rot) / B_rot;
    double y2_rot = (-A_rot * x2_rot - C_rot) / B_rot;

    double x1 = x1_rot * cos_theta - y1_rot * sin_theta + ellipse_center.x;
    double y1 = x1_rot * sin_theta + y1_rot * cos_theta + ellipse_center.y;

    double x2 = x2_rot * cos_theta - y2_rot * sin_theta + ellipse_center.x;
    double y2 = x2_rot * sin_theta + y2_rot * cos_theta + ellipse_center.y;

    Point pt1(cvRound(x1), cvRound(y1));
    Point pt2(cvRound(x2), cvRound(y2));

    // 如果debug模式，绘制矩形框、椭圆、直径和交点
    if (debug) {
      // 绘制矩形框
      Point2f rect_points[4];
      ellipse.points(rect_points);
      //  for (int i = 0; i < 4; i++) {
      //    line(pic, rect_points[i], rect_points[(i+1)%4], Scalar(0, 255, 255),
      //    2);
      //  }

      // 绘制扇叶椭圆
      cv::ellipse(pic, ellipse.center, Size(a / scale, b / scale), angle_deg, 0,
                  360, Scalar(255, 0, 255), 2);

      // 绘制两条直径
      line(pic, center1, center2, Scalar(255, 255, 0), 2);
      circle(pic, center1, 5, Scalar(0, 0, 255), -1);
      circle(pic, center2, 5, Scalar(0, 255, 0), -1);

      // 绘制交点
      circle(pic, pt1, 5, Scalar(255, 0, 0), -1);
      circle(pic, pt2, 5, Scalar(255, 0, 0), -1);
    }

    if (computeDistance(pt1, center2) > computeDistance(pt2, center2)) {
      intersections.emplace_back(pt1);
      blade.apex.push_back(pt1);
    } else {
      intersections.emplace_back(pt2);
      blade.apex.push_back(pt2);
    }
    circle(pic, intersections[0], 3, Scalar(0, 255, 0), -1);
  }

  // 共轭直径
  if (A != 0) { // 确保A不为零以避免除以零
    double new_slope = (b * b * B) / (a * a * A);
    // double new_slope = B / A;

    double A2 = new_slope;
    double B2 = -1.0;
    double C2 = center1.y - new_slope * center1.x;

    double A2_rot = A2 * cos_theta + B2 * sin_theta;
    double B2_rot = -A2 * sin_theta + B2 * cos_theta;
    double C2_rot = C2 + A2 * ellipse_center.x + B2 * ellipse_center.y;

    // 特殊处理B2_rot接近0的情况
    if (fabs(B2_rot) < 1e-8) {
      // 直接计算与旋转后的椭圆方程的交点
      double x_rot = -C2_rot / A2_rot;
      double discriminant2 = (b * b) * (1 - (x_rot * x_rot) / (a * a));

      if (discriminant2 >= 0) {
        double sqrt_discriminant2 = sqrt(discriminant2);
        double y1_rot = sqrt_discriminant2;
        double y2_rot = -sqrt_discriminant2;

        double x1_2 = x_rot * cos_theta - y1_rot * sin_theta + ellipse_center.x;
        double y1_2 = x_rot * sin_theta + y1_rot * cos_theta + ellipse_center.y;

        double x2_2 = x_rot * cos_theta - y2_rot * sin_theta + ellipse_center.x;
        double y2_2 = x_rot * sin_theta + y2_rot * cos_theta + ellipse_center.y;

        Point pt3(cvRound(x1_2), cvRound(y1_2));
        Point pt4(cvRound(x2_2), cvRound(y2_2));

        circle(pic, pt3, 3, Scalar(0, 255, 0), -1);
        circle(pic, pt4, 3, Scalar(0, 255, 0), -1);

        // 排序切向两个关键点
        Point O = center2;   // 获取矩形灯条中心点
        Point OP3 = pt3 - O; // OP3向量
        Point OP4 = pt4 - O; // OP4向量

        // 计算OP3N = (-OP3_y, OP3_x)
        Point OP3N(-OP3.y, OP3.x);

        // 计算OP3N和OP4的点积
        double dotProduct = OP3N.x * OP4.x + OP3N.y * OP4.y;

        // 根据点积符号决定push顺序
        if (dotProduct < 0) {
          intersections.emplace_back(pt3);
          blade.apex.push_back(pt3);
          intersections.emplace_back(pt4);
          blade.apex.push_back(pt4);
        } else {
          intersections.emplace_back(pt4);
          blade.apex.push_back(pt4);
          intersections.emplace_back(pt3);
          blade.apex.push_back(pt3);
        }
      }
    } else {
      // 正常处理B2_rot不接近0的情况
      double M2 =
          (1.0 / (a * a)) + (A2_rot * A2_rot) / (B2_rot * B2_rot * b * b);
      double N2 = (2.0 * A2_rot * C2_rot) / (B2_rot * B2_rot * b * b);
      double P2 = (C2_rot * C2_rot) / (B2_rot * B2_rot * b * b) - 1.0;

      double discriminant2 = N2 * N2 - 4.0 * M2 * P2;

      if (discriminant2 >= 0) {
        double sqrt_discriminant2 = sqrt(discriminant2);
        double x1_rot2 = (-N2 + sqrt_discriminant2) / (2.0 * M2);
        double x2_rot2 = (-N2 - sqrt_discriminant2) / (2.0 * M2);

        double y1_rot2 = (-A2_rot * x1_rot2 - C2_rot) / B2_rot;
        double y2_rot2 = (-A2_rot * x2_rot2 - C2_rot) / B2_rot;

        double x1_2 =
            x1_rot2 * cos_theta - y1_rot2 * sin_theta + ellipse_center.x;
        double y1_2 =
            x1_rot2 * sin_theta + y1_rot2 * cos_theta + ellipse_center.y;

        double x2_2 =
            x2_rot2 * cos_theta - y2_rot2 * sin_theta + ellipse_center.x;
        double y2_2 =
            x2_rot2 * sin_theta + y2_rot2 * cos_theta + ellipse_center.y;

        Point pt3(cvRound(x1_2), cvRound(y1_2));
        Point pt4(cvRound(x2_2), cvRound(y2_2));

        intersections.emplace_back(pt3);
        intersections.emplace_back(pt4);
        circle(pic, pt3, 3, Scalar(0, 255, 0), -1);
        circle(pic, pt4, 3, Scalar(0, 255, 0), -1);
        // 排序切向两个关键点的算法
        // Point O = blade.apex[0]; // 获取O点
        Point O = center2;   // 获取O点（矩形灯条中心点）
        Point OP3 = pt3 - O; // OP3向量
        Point OP4 = pt4 - O; // OP4向量

        // 计算OP3N = (-OP3_y, OP3_x)
        Point OP3N(-OP3.y, OP3.x);

        // 计算OP3N和OP4的点积
        double dotProduct = OP3N.x * OP4.x + OP3N.y * OP4.y;

        // 根据点积符号决定push顺序
        if (dotProduct < 0) {
          intersections.emplace_back(pt3);
          blade.apex.push_back(pt3);
          intersections.emplace_back(pt4);
          blade.apex.push_back(pt4);
        } else {
          intersections.emplace_back(pt4);
          blade.apex.push_back(pt4);
          intersections.emplace_back(pt3);
          blade.apex.push_back(pt3);
        }
      }
    }
  } else {
    // 当A == 0时，直线是水平的，共轭直径应该是垂直的
    double A2 = 1.0; // 垂直线的斜率是无穷大，方程为 x = constant
    double B2 = 0.0;
    double C2 = -center1.x; // 垂直线通过center1点

    double A2_rot = A2 * cos_theta + B2 * sin_theta;
    double B2_rot = -A2 * sin_theta + B2 * cos_theta;
    double C2_rot = C2 + A2 * ellipse_center.x + B2 * ellipse_center.y;

    // 特殊处理B2_rot接近0的情况
    if (fabs(B2_rot) < 1e-8) {
      // 直接计算与旋转后的椭圆方程的交点
      double x_rot = -C2_rot / A2_rot;
      double discriminant2 = (b * b) * (1 - (x_rot * x_rot) / (a * a));

      if (discriminant2 >= 0) {
        double sqrt_discriminant2 = sqrt(discriminant2);
        double y1_rot = sqrt_discriminant2;
        double y2_rot = -sqrt_discriminant2;

        double x1_2 = x_rot * cos_theta - y1_rot * sin_theta + ellipse_center.x;
        double y1_2 = x_rot * sin_theta + y1_rot * cos_theta + ellipse_center.y;

        double x2_2 = x_rot * cos_theta - y2_rot * sin_theta + ellipse_center.x;
        double y2_2 = x_rot * sin_theta + y2_rot * cos_theta + ellipse_center.y;

        Point pt3(cvRound(x1_2), cvRound(y1_2));
        Point pt4(cvRound(x2_2), cvRound(y2_2));

        circle(pic, pt3, 3, Scalar(0, 255, 0), -1);
        circle(pic, pt4, 3, Scalar(0, 255, 0), -1);

        // 排序切向两个关键点
        Point O = center2;   // 获取矩形灯条中心点
        Point OP3 = pt3 - O; // OP3向量
        Point OP4 = pt4 - O; // OP4向量

        // 计算OP3N = (-OP3_y, OP3_x)
        Point OP3N(-OP3.y, OP3.x);

        // 计算OP3N和OP4的点积
        double dotProduct = OP3N.x * OP4.x + OP3N.y * OP4.y;

        // 根据点积符号决定push顺序
        if (dotProduct < 0) {
          intersections.emplace_back(pt3);
          blade.apex.push_back(pt3);
          intersections.emplace_back(pt4);
          blade.apex.push_back(pt4);
        } else {
          intersections.emplace_back(pt4);
          blade.apex.push_back(pt4);
          intersections.emplace_back(pt3);
          blade.apex.push_back(pt3);
        }
      }
    } else {
      // 正常处理B2_rot不接近0的情况
      double M2 =
          (1.0 / (a * a)) + (A2_rot * A2_rot) / (B2_rot * B2_rot * b * b);
      double N2 = (2.0 * A2_rot * C2_rot) / (B2_rot * B2_rot * b * b);
      double P2 = (C2_rot * C2_rot) / (B2_rot * B2_rot * b * b) - 1.0;

      double discriminant2 = N2 * N2 - 4.0 * M2 * P2;

      if (discriminant2 >= 0) {
        double sqrt_discriminant2 = sqrt(discriminant2);
        double x1_rot2 = (-N2 + sqrt_discriminant2) / (2.0 * M2);
        double x2_rot2 = (-N2 - sqrt_discriminant2) / (2.0 * M2);

        double y1_rot2 = (-A2_rot * x1_rot2 - C2_rot) / B2_rot;
        double y2_rot2 = (-A2_rot * x2_rot2 - C2_rot) / B2_rot;

        double x1_2 =
            x1_rot2 * cos_theta - y1_rot2 * sin_theta + ellipse_center.x;
        double y1_2 =
            x1_rot2 * sin_theta + y1_rot2 * cos_theta + ellipse_center.y;

        double x2_2 =
            x2_rot2 * cos_theta - y2_rot2 * sin_theta + ellipse_center.x;
        double y2_2 =
            x2_rot2 * sin_theta + y2_rot2 * cos_theta + ellipse_center.y;

        Point pt3(cvRound(x1_2), cvRound(y1_2));
        Point pt4(cvRound(x2_2), cvRound(y2_2));

        intersections.emplace_back(pt3);
        intersections.emplace_back(pt4);
        circle(pic, pt3, 3, Scalar(0, 255, 0), -1);
        circle(pic, pt4, 3, Scalar(0, 255, 0), -1);
        // 排序切向两个关键点
        Point O = center2;   // 获取矩形灯条中心点
        Point OP3 = pt3 - O; // OP3向量
        Point OP4 = pt4 - O; // OP4向量

        // 计算OP3N = (-OP3_y, OP3_x)
        Point OP3N(-OP3.y, OP3.x);

        // 计算OP3N和OP4的点积
        double dotProduct = OP3N.x * OP4.x + OP3N.y * OP4.y;

        // 根据点积符号决定push顺序
        if (dotProduct < 0) {
          intersections.emplace_back(pt3);
          blade.apex.push_back(pt3);
          intersections.emplace_back(pt4);
          blade.apex.push_back(pt4);
        } else {
          intersections.emplace_back(pt4);
          blade.apex.push_back(pt4);
          intersections.emplace_back(pt3);
          blade.apex.push_back(pt3);
        }
      }
    }
  }
  return intersections;
}

// int main() {
//   cout << "传统算法检测" << endl;

//   bool useOffcialWindmill = true;
//   bool perspective = false;
//   bool useVideo = true;

//   if (useVideo) {
//     VideoCapture cap;
//     if (!useOffcialWindmill) {
//       cap = VideoCapture("/Users/clarencestark/RoboMaster/第四次任务/"
//                          "nanodet_rm/camera/build/output.avi");
//     } else {

//       // cap = VideoCapture(
//       // "/Users/clarencestark/RoboMaster/步兵打符-视觉组/"
//       // "传统识别算法/XJTU2025WindMill/imgs_and_videos/output.mp4");
//       // cap = VideoCapture("/Users/clarencestark/RoboMaster/第四次任务/"
//       //                    "nanodet_rm/camera/build/output222.mp4");
//       cap = VideoCapture("/Users/clarencestark/RoboMaster/第四次任务/"
//                          "nanodet_rm/camera/build/2-5-正对大符有R标(红色).avi");
//     }

//     if (!cap.isOpened()) {
//       cout << "无法打开视频文件" << endl;
//       return -1;
//     }

//     Mat frame;
//     bool paused = false;

//     // 添加帧率计算相关变量
//     double fps = 0;
//     auto last_time = high_resolution_clock::now();
//     int frame_count = 0;

//     while (true) {
//       if (!paused) {
//         if (!cap.read(frame)) {
//           cout << "视频结束或帧获取失败" << endl;
//           break;
//         }

//         // 计算帧率
//         frame_count++;
//         auto current_time = high_resolution_clock::now();
//         auto time_diff =
//             duration_cast<milliseconds>(current_time - last_time).count();

//         if (time_diff >= 1000) { // 每秒更新一次帧率
//           fps = frame_count * 1000.0 / time_diff;
//           frame_count = 0;
//           last_time = current_time;
//         }
//       }

//       DetectionResult result;
//       if (perspective) {
//         Mat transformedFrame = applyYawPerspectiveTransform(frame, 0.18);
//         WMBlade temp_blade;
//         result = detect(transformedFrame, temp_blade);
//         imshow("Original Image", frame);
//         imshow("Transformed Image", transformedFrame);
//       } else {
//         WMBlade temp_blade;
//         result = detect(frame, temp_blade);
//       }

//       // 帧率和处理时间
//       string fps_text = "FPS: " + to_string(static_cast<int>(fps));
//       string time_text =
//           "Process Time: " + to_string(result.processingTime) + "ms";
//       putText(result.processedImage, fps_text, Point(10, 30),
//               FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
//       putText(result.processedImage, time_text, Point(10, 70),
//               FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

//       // 显示结果图像
//       imshow("Processed Image", result.processedImage);

//       // 等待按键
//       char key = (char)waitKey(70);

//       // 按键控制
//       if (key == 'q' || key == 'Q') {
//         break; // 退出
//       } else if (key == ' ') {
//         paused = !paused; // 空格键切换暂停/继续
//       }
//     }

//     cap.release();
//   } else {
//     // 原有的图像处理逻辑
//     Mat frame;
//     if (!useOffcialWindmill) {
//       frame = imread("/Users/clarencestark/RoboMaster/第四次任务/nanodet_rm/"
//                      "camera/build/imgs/image52.jpg");
//     } else {
//       frame = imread("/Users/clarencestark/RoboMaster/步兵打符-视觉组/"
//                      "local_Indentify_Develop/src/test3.jpg");
//     }

//     if (frame.empty()) {
//       cout << "无法获取图像" << endl;
//       return -1;
//     }

//     WMBlade temp_blade;
//     DetectionResult result;
//     if (perspective) {
//       Mat transformedFrame = applyYawPerspectiveTransform(frame, 0.20);

//       // 显示原始图像和变换后的图像
//       imshow("Original Image", frame);
//       imshow("Transformed Image", transformedFrame);

//       // 处理变换后的图像
//       result = detect(transformedFrame, temp_blade);
//     } else {
//       result = detect(frame, temp_blade);
//     }

//     // 显示结果
//     cout << "处理时间: " << result.processingTime << " ms" << endl;
//     cout << "检测到 " << result.circlePoints.size() << " 个圆心" << endl;
//     cout << "检测到 " << result.intersections.size() << " 个交点" << endl;

//     // 显示结果图像
//     imshow("Processed Image", result.processedImage);
//     waitKey(0);
//   }

//   return 0;
// }
