/**
 * @file traditional_detection.cpp
 * @author Clarence Stark (3038736583@qq.com)
 * @brief 用于传统算法检测
 * @version 0.1
 * @date 2025-01-04
 *
 * @copyright Copyright (c) 2025
 */

#include "globalParam.hpp"
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
// int circularityThreshold = 45; // 圆度阈值
// int medianBlurSize = 3;        // 中值滤波核大小

// bool gp.debug = false;        // gp.debug模式
// bool useTrackbars = gp.debug; // 是否使用滑动条动态调参
// int dilationSize = 7;      // 膨胀核大小
// int erosionSize = 3;       // 腐蚀核大小
// int thresholdValue = 108;  // 二值化阈值

// int rect_area_threshold = 2000; // 矩形面积阈值
// int circle_area_threshold = 50; // 类圆轮廓面积阈值

// int length_width_ratio_threshold = 3; // 长宽比阈值

// int minContourArea = 200; // 最小轮廓面积

void createTrackbars(GlobalParam &gp) {
  namedWindow(WINDOW_NAME, WINDOW_NORMAL);
  moveWindow(WINDOW_NAME, 0, 0);
  createTrackbar("Circularity", WINDOW_NAME, &gp.circularityThreshold, 100,
                 nullptr);
  createTrackbar("Dilation Size", WINDOW_NAME, &gp.dilationSize, 21, nullptr);
  createTrackbar("Erosion Size", WINDOW_NAME, &gp.erosionSize, 21, nullptr);
  createTrackbar("Blur Size", WINDOW_NAME, &gp.medianBlurSize, 21, nullptr);

  createTrackbar("Rect Area Threshold", WINDOW_NAME, &gp.rect_area_threshold,
                 1000, nullptr);
  createTrackbar("Min Contour Area", WINDOW_NAME, &gp.minContourArea, 1000,
                 nullptr);
  createTrackbar("Circle Area Threshold", WINDOW_NAME,
                 &gp.circle_area_threshold, 1000, nullptr);
  //createTrackbar("Length Width Ratio Threshold", WINDOW_NAME,
  //               &gp.length_width_ratio_threshold, 10.0, nullptr);
  createTrackbar("Threshold", WINDOW_NAME, &gp.thresholdValue, 255, nullptr);
  createTrackbar("Threshold for ROI", WINDOW_NAME, &gp.thresholdValue_for_roi, 255, nullptr);
}

/**
 * @brief 图像预处理函数
 * @param inputImage 输入图像
 * @param gp.debug 是否显示中间步骤
 * @return 预处理后的掩码图像
 */
cv::Mat preprocess(const cv::Mat &inputImage, GlobalParam &gp, int is_blue, Translator &translator) {
  if (gp.debug) {
    static bool trackbarsInitialized = false;
    if (!trackbarsInitialized) {
      createTrackbars(gp);
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
  if (!is_blue) {
    subtract(red, blue, temp);
    if (!translator.message.is_far) {
      threshold(temp, final_mask, gp.thresholdValue, 255, THRESH_BINARY);
    } else {
      threshold(temp, final_mask, gp.thresholdValue_1, 255, THRESH_BINARY);
    }
  } else {
    subtract(blue, red, temp);
    if (!translator.message.is_far) {
      threshold(temp, final_mask, gp.thresholdValueBlue, 255, THRESH_BINARY);
    } else {
      threshold(temp, final_mask, gp.thresholdValueBlue_1, 255, THRESH_BINARY);
    }
  }
  int kernelSize;
  // 确保medianBlurSize是奇数
  if (!translator.message.is_far) {
    kernelSize = gp.medianBlurSize;
  } else {
    kernelSize = gp.medianBlurSize_1;
  }

  if (kernelSize % 2 == 0) {
    kernelSize++;
  }
  cv::medianBlur(final_mask, final_mask, kernelSize);
  //  cv::imshow("medianBlur", temp);

  // 对灰度图进行二值化

  Mat kernel1;
  Mat kernel2;

  if (!translator.message.is_far) {
    kernel1 = getStructuringElement(MORPH_RECT,
                                    Size(gp.dilationSize, gp.dilationSize));
    kernel2 =
        getStructuringElement(MORPH_RECT, Size(gp.erosionSize, gp.erosionSize));
  } else {
    kernel1 = getStructuringElement(MORPH_RECT,
                                    Size(gp.dilationSize_1, gp.dilationSize_1));
    kernel2 = getStructuringElement(MORPH_RECT,
                                    Size(gp.erosionSize_1, gp.erosionSize_1));
  }

  dilate(final_mask, final_mask, kernel1);
  erode(final_mask, final_mask, kernel2);
  return final_mask;
}

/**
 * @brief 识别初始的圆形和矩形候选轮廓
 * @param contours 输入的轮廓
 * @param hierarchy 轮廓的层级结构
 * @param is_potential_rect_contour_flags 输出参数，标记哪些轮廓是潜在的矩形
 * (大小与contours相同)
 * @return KeyPoints 结构，包含初步识别的圆形轮廓及其属性 (面积、圆度、中心点)
 */
KeyPoints identify_initial_shapes(const std::vector<std::vector<cv::Point>> &contours,
                        const std::vector<cv::Vec4i> &hierarchy,
                        std::vector<bool> &is_potential_rect_contour_flags,
                        GlobalParam &gp,
                        std::vector<int> &child_counts_for_circles) {

  KeyPoints initial_circles_result;
  is_potential_rect_contour_flags.assign(contours.size(), false);
  child_counts_for_circles.clear();

  for (int i = 0; i < (int)contours.size(); ++i) {
    const auto &contour = contours[i];
    double area = cv::contourArea(contour);

    // 过滤掉面积过小的轮廓
    if (area < gp.minContourArea) {
      continue;
    }

    // --- 矩形检测 (流水灯条) ---
    // 逻辑：外层轮廓，面积达标，长宽比达标，且形状必须接近矩形（非梯形）
    if (hierarchy[i][3] == -1) { 
      if (area > gp.rect_area_threshold) {
        cv::RotatedRect rect = cv::minAreaRect(contour);
        float width = rect.size.width;
        float height = rect.size.height;

        if (height == 0 || width == 0) continue;

        float aspectRatio = (width > height) ? (width / height) : (height / width);

        // 1. 长宽比检查
        if (aspectRatio > gp.length_width_ratio_threshold) {
          
          // 2. 核心改进：矩形度检查 (Extent)
          // 计算轮廓面积与外接矩形面积的比值
          double rect_area = width * height;
          double extent = area / rect_area;

          // 3. 核心改进：顶点数检查
          std::vector<cv::Point> approx;
          double epsilon = 0.04 * cv::arcLength(contour, true); 
          cv::approxPolyDP(contour, approx, epsilon, true);

          // 准则：
          // - 矩形度 extent 应该在 0.70 以上
          // - 逼近后的顶点数应该在 3 到 6 之间
          if (extent > 0.45 && approx.size() >= 3 && approx.size() <= 6) {
              is_potential_rect_contour_flags[i] = true;
          } else {
              // --- 修正后的打印逻辑 ---
          //if (gp.debug && area > 500) {
          //        printf("[矩形初步过滤] ID:%d 面积:%.0f 比例:%.2f(需>%f) 矩形度:%.2f(需>0.7) 顶点:%zu(需3-6)\n", 
          //               i, area, aspectRatio, gp.length_width_ratio_threshold, extent, approx.size());
          //  }
          }
        }
        //else{
        //  if (gp.debug && area > 500) {
        //      printf("[矩形初步过滤]长宽比 %.2f 太小(需>%f)\n", aspectRatio, gp.length_width_ratio_threshold);
        //  }
        //}
      //} else if (gp.debug && area > 500) {
      //    printf("[矩形初步过滤] 面积 %.0f 太小(需>%d)\n", area, gp.rect_area_threshold);
      }
    }

    // 如果已标记为潜在矩形，则跳过圆形检测
    if (is_potential_rect_contour_flags[i]) {
      continue;
    }

    // --- 圆形检测 (目标扇叶和R标) ---
    if (area > gp.circle_area_threshold) {
      double perimeter = cv::arcLength(contour, true);
      if (perimeter == 0) continue;

      double circularity = 4 * CV_PI * area / (perimeter * perimeter);

      if (circularity > (gp.circularityThreshold / 100.0)) {
        int childCount = 0;
        if (hierarchy[i][2] != -1) { 
          int current_child_idx = hierarchy[i][2];
          while (current_child_idx != -1) {
            childCount++;
            current_child_idx = hierarchy[current_child_idx][0];
          }
        }

        bool is_target_candidate = childCount > 1;
        bool is_r_logo_candidate = (childCount == 0 && hierarchy[i][3] == -1);

        if (is_target_candidate || is_r_logo_candidate) {
          initial_circles_result.circleContours.push_back(contour);
          initial_circles_result.circleAreas.push_back(area);
          initial_circles_result.circularities.push_back(circularity);
          child_counts_for_circles.push_back(childCount);

          cv::Moments m = cv::moments(contour);
          if (m.m00 == 0) continue;
          cv::Point circleCenter(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
          initial_circles_result.circlePoints.push_back(circleCenter);
        }
      }
    }
  }
  return initial_circles_result;
}
/**
 * @brief 对初步识别的矩形进行筛选，包括ROI处理和距离筛选
 * @param all_contours 原始图像中的所有轮廓
 * @param is_potential_rect_contour_flags 标记了哪些轮廓是初步认定的矩形
 * @param processed_image 用于提取ROI的图像的非const引用 (因为会从中提取ROI)
 * @param initial_circle_centers 初步识别的圆心 (用于距离筛选)
 * @param initial_circle_areas 初步识别的圆面积 (用于找到最大圆作参考)
 * @param final_selected_rect_flags 输出参数，标记最终选定的矩形轮廓
 * (大小与all_contours相同)
 * @param debug_flag 是否开启调试模式
 * @return 最终确定的矩形中心点列表
 */
std::vector<cv::Point2f> refine_rectangles_roi(
    const std::vector<std::vector<cv::Point>> &all_contours,
    const std::vector<bool> &is_potential_rect_contour_flags,
    cv::Mat &processed_image,
    const std::vector<cv::Point> &initial_circle_centers,
    const std::vector<double> &initial_circle_areas,
    const std::vector<int> &initial_circle_child_counts,
    std::vector<bool> &final_selected_rect_flags, bool debug_flag,
    GlobalParam &gp) {

  std::vector<cv::Point2f> final_rect_centers_list;
  final_selected_rect_flags.assign(all_contours.size(), false);
  std::vector<int> candidate_indices;
  std::vector<cv::Point2f> candidate_centers_coords;

  // 1. 初步筛选
  for (int i = 0; i < (int)all_contours.size(); ++i) {
    if (is_potential_rect_contour_flags[i]) {
      cv::Moments m = cv::moments(all_contours[i]);
      if (m.m00 == 0) continue;
      candidate_centers_coords.push_back(cv::Point2f(m.m10 / m.m00, m.m01 / m.m00));
      candidate_indices.push_back(i);
    }
  }

  if (candidate_indices.empty()) return final_rect_centers_list;

  for (size_t k = 0; k < candidate_indices.size(); ++k) {
    int idx = candidate_indices[k];
    
    // ================= 方法一：填充率判断 (Solidity) =================
    // 不需要扣图，直接算几何属性
    cv::RotatedRect r_rect = cv::minAreaRect(all_contours[idx]);
    double contour_area = cv::contourArea(all_contours[idx]);
    double rect_area = r_rect.size.width * r_rect.size.height;
    
    // 计算填充率 (轮廓面积 / 旋转矩形面积)
    // 流水灯条因为有空隙，填充率通常较低 (0.3 ~ 0.6)
    // 实心物体填充率通常很高 (> 0.8)
    double solidity = 0.0;
    if (rect_area > 0) solidity = contour_area / rect_area;

    // ================= 方法二：强腐蚀分块法 =================
    // 依然需要一个小 ROI 来做图像处理，但不需要太精确
    cv::Rect roi_rect = cv::boundingRect(all_contours[idx]);
    // 稍微扩大一点 ROI，防止边缘被切断，保证处理完整性
    roi_rect.x = std::max(0, roi_rect.x - 5);
    roi_rect.y = std::max(0, roi_rect.y - 5);
    roi_rect.width = std::min(roi_rect.width + 10, processed_image.cols - roi_rect.x);
    roi_rect.height = std::min(roi_rect.height + 10, processed_image.rows - roi_rect.y);

    if (roi_rect.width <= 0 || roi_rect.height <= 0) continue;

    // 提取 ROI (直接用单通道，不需要转灰度再二值化，直接用原图二值化后的结果更准)
    // 这里假设 processed_image 是原图。我们需要重新在局部做一次二值化。
    cv::Mat roi_gray;
    if (processed_image(roi_rect).channels() == 3) 
        cv::cvtColor(processed_image(roi_rect), roi_gray, cv::COLOR_BGR2GRAY);
    else 
        roi_gray = processed_image(roi_rect).clone();
        
    cv::Mat roi_bin;
    cv::threshold(roi_gray, roi_bin, gp.thresholdValue_for_roi, 255, cv::THRESH_BINARY);

    // 【关键步骤】强力腐蚀
    // 使用较大的核，把粘连的箭头强行断开
    // 3x3 可能不够，针对大目标可能需要 5x5 或更大，这里先用 3x3 迭代 2 次
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));
    cv::erode(roi_bin, roi_bin, kernel, cv::Point(-1,-1), 1); // iterations = 2

    // 找轮廓计数
    std::vector<std::vector<cv::Point>> sub_contours;
    cv::findContours(roi_bin, sub_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    int valid_blobs = 0;
    for(const auto& cnt : sub_contours) {
        if(cv::contourArea(cnt) > 0) valid_blobs++; // 过滤噪点
    }

    // ================= 综合判断 =================
    // 条件1：腐蚀后分裂成多个块 (说明是断续的) -> 肯定是流水灯条
    // 条件2：填充率适中 (说明中间有空隙) -> 也就是 solidity < 0.75
    
    bool is_flowing_light = false;

    // 逻辑：如果分裂成了超过2个块，或者填充率比较低(说明空心)，认为是流水灯条
    if (valid_blobs >= 2 || (solidity < 0.75 && valid_blobs > 0)) {
        is_flowing_light = true;
    }
    std::cout << "ID:" << idx 
              << " 填充率:" << solidity 
              << " 分块数:" << valid_blobs 
              << " 判定:" << (is_flowing_light ? "流水灯条" : "非流水灯条") 
              << std::endl;

    if (is_flowing_light) {
      final_rect_centers_list.push_back(candidate_centers_coords[k]);
      final_selected_rect_flags[idx] = true;
    } else {
      if (debug_flag) {
          printf("[矩形过滤失败] ID:%d 填充率:%.2f(需<0.75) 分块数:%d(需>=2)\n", 
                 idx, solidity, valid_blobs);
          
          // 可视化调试
          //if (valid_blobs > 0) { // 画一下腐蚀后的样子
          //    std::string win_name = "Eroded Check ID:" + std::to_string(idx);
          //    cv::imshow(win_name, roi_bin);
          }
      }
    }
  }
  return final_rect_centers_list;
}
/**
 * @brief 根据矩形中心筛选最终的两个圆形轮廓 (目标扇叶和R标)
 * @param current_keypoints
 * KeyPoints结构，包含初步识别的圆，此函数会就地修改它以包含最终选择的圆
 * @param detected_rect_centers 检测到的矩形中心点列表 (通常只使用第一个)
 * @param circle_child_counts 每个初步识别圆的子轮廓数量
 * @param debug_flag 是否开启调试模式
 */
void select_final_circles(KeyPoints &current_keypoints,
                          const std::vector<cv::Point2f> &detected_rect_centers,
                          const std::vector<int> &circle_child_counts,
                          bool debug_flag, GlobalParam &gp,
                          Translator &translator) {

  // 1. 基本条件检查
  if (current_keypoints.circleContours.size() < 2 || detected_rect_centers.empty()) {
    return;
  }

  // 2. 状态与模式判断变量
  bool is_big_windmill = (translator.message.status % 5 == 3);
  static bool is_target_locked = false;
  static cv::Point2f last_blade_pos; // 持久化：记录上一帧锁定的【扇叶】绝对坐标
  std::cout << "现在矩形个数:" << detected_rect_centers.size() << std::endl;
  std::cout << "现在圆形个数:" << current_keypoints.circleContours.size() << std::endl;

  // 3. 第一步：筛选出所有候选扇叶 (必须满足面积和子轮廓条件)
  std::vector<int> blade_candidate_indices;
  for (int i = 0; i < (int)current_keypoints.circleContours.size(); ++i) {
      double area = current_keypoints.circleAreas[i];
      int child = circle_child_counts[i];
      if (child >= 2 && area > gp.target_circle_area_min && area < gp.target_circle_area_max) {
          blade_candidate_indices.push_back(i);
      }
  }

  if (blade_candidate_indices.empty()) {
      is_target_locked = false;
      return;
  }

  // 4. 第二步：在候选扇叶中锁定一个目标 (selected_blade_idx)
  int selected_blade_idx = -1;

  if (!is_target_locked) {
      // 首次锁定：选择 Y 坐标最小的（最上方的扇叶）
      double min_y = DBL_MAX;
      for (int idx : blade_candidate_indices) {
          if (current_keypoints.circlePoints[idx].y < min_y) {
              min_y = current_keypoints.circlePoints[idx].y;
              selected_blade_idx = idx;
          }
          else{
            std::cout << "扇叶候选 ID:" << idx << " Y坐标:" << current_keypoints.circlePoints[idx].y << std::endl;
          }
      }
      if (selected_blade_idx != -1) {
          last_blade_pos = cv::Point2f(current_keypoints.circlePoints[selected_blade_idx]);
          is_target_locked = true;
      }
  } else {
      // 持续跟踪：选择离上一帧扇叶位置最近的候选
      double min_dist = DBL_MAX;
      for (int idx : blade_candidate_indices) {
          double dist = cv::norm(cv::Point2f(current_keypoints.circlePoints[idx]) - last_blade_pos);
          if (dist < min_dist) {
              min_dist = dist;
              selected_blade_idx = idx;
          }
      }
      // 距离保护：如果跨度超过 150 像素，认为丢了
      if (selected_blade_idx != -1 && min_dist < 150.0) {
          last_blade_pos = cv::Point2f(current_keypoints.circlePoints[selected_blade_idx]);
      } else {
          is_target_locked = false;
          if(gp.debug) std::cout << "[扇叶丢失] 距离过大，重新锁定" << std::endl; 
          return;
      }
  }
  //std::cout << "选中扇叶 ID:" << selected_blade_idx << std::endl;
  // 5. 第三步：找到离这个扇叶最近的矩形 (selected_rect_center)
  cv::Point2f b_pos = cv::Point2f(current_keypoints.circlePoints[selected_blade_idx]);
  cv::Point2f selected_rect_center;
  double min_rect_dist = DBL_MAX;
  for (const auto& rect_p : detected_rect_centers) {
      double d = cv::norm(rect_p - b_pos);
      if (d < min_rect_dist && d < 150.0) { // 距离保护：矩形不能太远
          min_rect_dist = d;
          selected_rect_center = rect_p;
      }
      else if(d >= 50.0){
        //std::cout << "矩形离扇叶太远 距离:" << d << std::endl;
        //std::cout <<"当前最小距离:" << min_rect_dist << std::endl;
      }
  }
  std::cout << "最小距离:" << min_rect_dist << std::endl;

  // 6. 第四步：找到离这个矩形最近的 R 标 (selected_r_idx)
  int selected_r_idx = -1;
  double min_r_dist = DBL_MAX;
  for (int i = 0; i < (int)current_keypoints.circleContours.size(); ++i) {
      if (i == selected_blade_idx) {
        //std:;cout << "跳过扇叶自己 ID:" << i << std::endl;
        continue;
      }; // 不能是扇叶自己
      
      double area = current_keypoints.circleAreas[i];
      // R标条件：符合面积范围
      if (area > gp.R_area_min && area < gp.R_area_max) {
          double d = cv::norm(cv::Point2f(current_keypoints.circlePoints[i]) - selected_rect_center);
          if (d < min_r_dist) {
              min_r_dist = d;
              selected_r_idx = i;
          }
      }
      else{
        std::cout << "R标面积不符 ID:" << i << " 面积:" << area << std::endl;
      }
  }
  std::cout << "选中R标 面积：" 
            << (selected_r_idx != -1 ? std::to_string(current_keypoints.circleAreas[selected_r_idx]) : "无") 
            << std::endl;

  // 7. 最终提纯数据
  if (selected_blade_idx != -1 && selected_r_idx != -1) {
      KeyPoints pure;
      pure.rectCenters.push_back(selected_rect_center);
      
      // 存入扇叶 (Index 0)
      pure.circleContours.push_back(current_keypoints.circleContours[selected_blade_idx]);
      pure.circleAreas.push_back(current_keypoints.circleAreas[selected_blade_idx]);
      pure.circlePoints.push_back(current_keypoints.circlePoints[selected_blade_idx]);
      pure.circularities.push_back(current_keypoints.circularities[selected_blade_idx]);

      // 存入 R 标 (Index 1)
      pure.circleContours.push_back(current_keypoints.circleContours[selected_r_idx]);
      pure.circleAreas.push_back(current_keypoints.circleAreas[selected_r_idx]);
      pure.circlePoints.push_back(current_keypoints.circlePoints[selected_r_idx]);
      pure.circularities.push_back(current_keypoints.circularities[selected_r_idx]);

      current_keypoints = pure;
  } else {
      if (debug_flag) printf("[提纯失败] 扇叶:%d R标:%d\n", selected_blade_idx, selected_r_idx);
  }
}
/**
 * @brief 可视化检测到的关键点 (如果debug模式开启)
 * @param image_to_draw_on 用于绘制的图像 (通常是 processedImage 的克隆)
 * @param final_result 包含最终检测结果的KeyPoints结构
 * @param all_contours 原始图像中的所有轮廓
 * @param initial_is_rect_flags 标记了哪些轮廓是初步认定的矩形
 * @param final_selected_rect_flags 标记了最终选定的矩形轮廓
 */
void visualize_keypoints(
    cv::Mat &image_to_draw_on, // Mat会被修改用于显示
    const KeyPoints &final_result,
    const std::vector<std::vector<cv::Point>> &all_contours,
    const std::vector<bool> &initial_is_rect_flags, // 初始矩形候选标记
    const std::vector<bool> &final_selected_rect_flags) { // 最终选择的矩形标记

  // --- 新增：寻找 R 标索引 (最终选出的圆中面积最小的通常为 R 标) ---
  int r_idx_in_final = -1;
  if (!final_result.circleAreas.empty()) {
      auto min_it = std::min_element(final_result.circleAreas.begin(), final_result.circleAreas.end());
      r_idx_in_final = std::distance(final_result.circleAreas.begin(), min_it);
  }

  // 绘制最终选择的圆形轮廓 (绿色)
  for (size_t i = 0; i < final_result.circleContours.size(); ++i) {
    cv::drawContours(image_to_draw_on, final_result.circleContours,
                     static_cast<int>(i), cv::Scalar(0, 255, 0), 2); // 绿色轮廓
    cv::circle(image_to_draw_on, final_result.circlePoints[i], 5,
               cv::Scalar(0, 255, 0), -1); // 绿色中心点

    // --- 新增：如果是 R 标，使用加粗蓝色矩形框标记 ---
    if ((int)i == r_idx_in_final) {
        cv::Rect r_rect = cv::boundingRect(final_result.circleContours[i]);
        cv::rectangle(image_to_draw_on, r_rect, cv::Scalar(255, 0, 0), 5); // 蓝色加粗框
        cv::putText(image_to_draw_on, "R-LOGO", r_rect.tl() + cv::Point(0, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
    }

    std::string circle_info =
        "Area: " +
        std::to_string(static_cast<int>(final_result.circleAreas[i])) +
        " Circ: " +
        (final_result.circularities[i] >= 0 &&
                 final_result.circularities[i] <= 1
             ? std::to_string(final_result.circularities[i]).substr(0, 4)
             : "N/A");
    cv::putText(image_to_draw_on, circle_info,
                final_result.circlePoints[i] + cv::Point(10, 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0),
                1); // 青色文字
  }

  // 绘制所有初步识别但未被最终选中的矩形轮廓 (白色)
  for (size_t i = 0; i < all_contours.size(); ++i) {
    if (initial_is_rect_flags[i] && !final_selected_rect_flags[i]) {
      cv::drawContours(image_to_draw_on, all_contours, static_cast<int>(i),
                       cv::Scalar(255, 255, 255), 1); // 白色细线条
    }
  }

  // 绘制最终筛选完毕的目标矩形轮廓 (紫色)
  for (size_t i = 0; i < all_contours.size(); ++i) {
    if (final_selected_rect_flags[i]) { // 如果这个轮廓是最终选定的矩形之一
      cv::drawContours(image_to_draw_on, all_contours, static_cast<int>(i),
                       cv::Scalar(255, 0, 255), 3); // 紫色粗线条

      cv::Point2f rect_center_to_draw;
      bool center_for_drawing_found = false;

      // 尝试从 final_result.rectCenters 中匹配正确的中心点
      // 如果只有一个最终矩形，其中心点应该就是 final_result.rectCenters[0]
      if (final_result.rectCenters.size() == 1) {
        rect_center_to_draw = final_result.rectCenters[0];
        center_for_drawing_found = true;
      } else { // 如果有多个最终矩形，需要匹配
        cv::Moments m_contour = cv::moments(all_contours[i]);
        if (m_contour.m00 != 0) {
          cv::Point2f current_contour_center(
              static_cast<float>(m_contour.m10 / m_contour.m00),
              static_cast<float>(m_contour.m01 / m_contour.m00));
          for (const auto &stored_center : final_result.rectCenters) {
            if (cv::norm(stored_center - current_contour_center) <
                1.0) { // 允许微小误差
              rect_center_to_draw = stored_center;
              center_for_drawing_found = true;
              break;
            }
          }
        }
      }
      // 如果通过上述方法未找到，作为后备，直接计算当前轮廓的中心
      if (!center_for_drawing_found) {
        cv::Moments m = cv::moments(all_contours[i]);
        if (m.m00 != 0) {
          rect_center_to_draw = cv::Point2f(static_cast<float>(m.m10 / m.m00),
                                            static_cast<float>(m.m01 / m.m00));
          center_for_drawing_found = true; // 至少我们有一个中心点来绘制
        }
      }

      if (center_for_drawing_found) {
        cv::circle(image_to_draw_on,
                   cv::Point(static_cast<int>(rect_center_to_draw.x),
                             static_cast<int>(rect_center_to_draw.y)),
                   8, cv::Scalar(255, 0, 255), -1); // 紫色中心点

        cv::RotatedRect rot_rect = cv::minAreaRect(all_contours[i]);
        float width = rot_rect.size.width;
        float height = rot_rect.size.height;
        float aspectRatio =
            (width == 0 || height == 0)
                ? 0.0f
                : ((width > height) ? (width / height) : (height / width));
        float area = static_cast<float>(cv::contourArea(all_contours[i]));

        cv::putText(image_to_draw_on, "Final Rect",
                    cv::Point(static_cast<int>(rect_center_to_draw.x),
                              static_cast<int>(rect_center_to_draw.y)) +
                        cv::Point(10, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 255), 2);
        std::string rect_info =
            "Ratio: " +
            (aspectRatio > 0 ? std::to_string(aspectRatio).substr(0, 4)
                             : "N/A") +
            " Area: " + std::to_string(static_cast<int>(area));
        cv::putText(image_to_draw_on, rect_info,
                    cv::Point(static_cast<int>(rect_center_to_draw.x),
                              static_cast<int>(rect_center_to_draw.y)) +
                        cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 1);
      }
    }
  }

  cv::Mat resized_image;
  cv::Size newSize(840, 620); // 与原代码一致的示例尺寸
  if (image_to_draw_on.cols > 0 && image_to_draw_on.rows > 0) { // 确保图像有效
    cv::resize(image_to_draw_on, resized_image, newSize, 0, 0,
               cv::INTER_LINEAR);
    cv::imshow("Detected Key Points (Refactored)", resized_image);
  } else {
    cv::imshow("Detected Key Points (Refactored) - Original Size",
               image_to_draw_on);
  }
  cv::waitKey(1); // 保持窗口更新
}

/**
 * @brief 检测关键点（R标，扇叶中心点，流水灯条中心点）
 * @param contours 轮廓
 * @param hierarchy 轮廓层次
 * @param processedImage 处理后的图像
 * (用于ROI和可视化，会被克隆使用，本身不修改)
 * @param blade WMBlade引用 (当前在此重构代码中未使用，但保留以匹配原始函数签名)
 * @return KeyPoints 关键点结果
 */
KeyPoints detect_key_points(
    const std::vector<std::vector<cv::Point>> &contours,
    const std::vector<cv::Vec4i> &hierarchy,
    cv::Mat &processedImage, 
    WMBlade &blade, 
    GlobalParam &gp,
    Translator &translator) { // 新增了参数

  KeyPoints final_result;                  // 最终返回的结果
  std::vector<bool> initial_is_rect_flags; // 标记初步识别的矩形轮廓
  std::vector<bool> final_selected_rect_flags; // 标记最终选定的矩形轮廓
  std::vector<int> initial_circle_child_counts;

  // 1. 识别初始的圆形候选和标记潜在的矩形轮廓
  KeyPoints initial_potential_circles =
      identify_initial_shapes(contours, hierarchy, initial_is_rect_flags, gp,
                              initial_circle_child_counts);
  
  if (gp.debug) {
    std::cout << "初始识别到 " << initial_potential_circles.circleContours.size() << " 个候选圆。" << std::endl;
  }

  // 2. 筛选矩形轮廓 (此时返回 1 或 2 个矩形)
  final_result.rectCenters = refine_rectangles_roi(
      contours,
      initial_is_rect_flags,                  
      processedImage,                         
      initial_potential_circles.circlePoints, 
      initial_potential_circles.circleAreas, 
      initial_circle_child_counts,
      final_selected_rect_flags, 
      gp.debug, gp               
  );

  // 3. 将初始圆信息复制到结果中，准备进入最后的 select_final_circles
  final_result.circleContours = initial_potential_circles.circleContours;
  final_result.circleAreas = initial_potential_circles.circleAreas;
  final_result.circularities = initial_potential_circles.circularities;
  final_result.circlePoints = initial_potential_circles.circlePoints;

  // 4. 调用通用的筛选函数 (处理锁定、大小符逻辑、2圆1矩形提纯)
  // 这里传入了 translator
  select_final_circles(
      final_result, 
      final_result.rectCenters, 
      initial_circle_child_counts, 
      gp.debug, 
      gp,
      translator 
  );

  if (gp.debug) {
    std::cout << "最终筛选后，保留 " << final_result.circleContours.size() << " 个圆。" << std::endl;
  }

  // 5. 可视化最终结果
  if (gp.debug) {
    cv::Mat visual_image = processedImage.clone(); 
    visualize_keypoints(visual_image, final_result, contours, initial_is_rect_flags, final_selected_rect_flags);
  }

  return final_result;
}


DetectionResult detect(const cv::Mat &inputImage, WMBlade &blade,
                       GlobalParam &gp, int is_blue, Translator &translator) {

  DetectionResult result;
  auto start_time = high_resolution_clock::now();

  Mat final_mask = preprocess(inputImage, gp, is_blue, translator);
  cv::imshow("Final Mask", final_mask);  // 显示二值化结果

  Mat processedImage = inputImage.clone(); // 克隆一份用于绘图

  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(final_mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

  // 1. 基础形状识别
  std::vector<bool> initial_is_rect_flags;
  std::vector<int> initial_circle_child_counts;
  KeyPoints final_result = identify_initial_shapes(contours, hierarchy, initial_is_rect_flags, gp, initial_circle_child_counts);

  // 2. 矩形精筛
  std::vector<bool> final_selected_rect_flags;
  final_result.rectCenters = refine_rectangles_roi(contours, initial_is_rect_flags, processedImage, 
                                                final_result.circlePoints, final_result.circleAreas, 
                                                initial_circle_child_counts, final_selected_rect_flags, gp.debug, gp); 

  // 3. 执行核心锁定逻辑 (扇叶 -> 矩形 -> R标)
  select_final_circles(final_result, final_result.rectCenters, initial_circle_child_counts, gp.debug, gp, translator);

  // 4. 有效性检查
  if (!final_result.isValid()) {
    result.processedImage = processedImage;
    return result;
  }

  // 5. 数据排序
  vector<size_t> indices(final_result.circleContours.size());
  iota(indices.begin(), indices.end(), 0);
  sort(indices.begin(), indices.end(), [&final_result](size_t i1, size_t i2) {
    return final_result.circleAreas[i1] > final_result.circleAreas[i2];
  });

  if (indices.size() < 2) return result;

  // 6. 填充 PNP 2D 点
  blade.apex.push_back(final_result.circlePoints[indices[1]]); // R 标
  blade.apex.push_back(final_result.circlePoints[indices[0]]); // 扇叶
  
  cv::RotatedRect ellipse = cv::fitEllipse(final_result.circleContours[indices[0]]);
  cv::Point2f center1(ellipse.center.x, ellipse.center.y);
  double radius = sqrt(final_result.circleAreas[indices[0]] / CV_PI);

  // 7. 交点解算
  result.intersections = findIntersectionsByEquation(center1, final_result.rectCenters[0], radius, ellipse, processedImage, gp, blade);

  // ==========================================
  //  可视化检测到的关键点 (如果debug模式开启)
  //  注意：之前名为 visualize_keypoints 的函数内容被复制到这里
  // ==========================================
  if (gp.debug) {
    cv::Mat &image_to_draw_on = processedImage; // 使用引用，避免不必要的拷贝

    // 绘制最终选择的圆形轮廓 (绿色)
    for (size_t i = 0; i < final_result.circleContours.size(); ++i) {
      cv::drawContours(image_to_draw_on, final_result.circleContours,
                       static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
      cv::circle(image_to_draw_on, final_result.circlePoints[i], 5,
                 cv::Scalar(0, 255, 0), -1);
      std::string circle_info =
          "Area: " +
          std::to_string(static_cast<int>(final_result.circleAreas[i])) +
          " Circ: " +
          (final_result.circularities[i] >= 0 &&
                 final_result.circularities[i] <= 1
             ? std::to_string(final_result.circularities[i]).substr(0, 4)
             : "N/A");
      cv::putText(image_to_draw_on, circle_info,
                  final_result.circlePoints[i] + cv::Point(10, 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0),
                  1);
    }

    // 绘制所有初步识别但未被最终选中的矩形轮廓 (白色)
    for (size_t i = 0; i < contours.size(); ++i) {
      if (initial_is_rect_flags[i] && !final_selected_rect_flags[i]) {
        cv::drawContours(image_to_draw_on, contours, static_cast<int>(i),
                         cv::Scalar(255, 255, 255), 1);
      }
    }

    // 绘制最终筛选完毕的目标矩形轮廓 (紫色)
    for (size_t i = 0; i < contours.size(); ++i) {
      if (final_selected_rect_flags[i]) { // 如果这个轮廓是最终选定的矩形之一
        cv::drawContours(image_to_draw_on, contours, static_cast<int>(i),
                         cv::Scalar(255, 0, 255), 3); // 紫色粗线条

        cv::Point2f rect_center_to_draw;
        bool center_for_drawing_found = false;

        // 尝试从 final_result.rectCenters 中匹配正确的中心点
        // 如果只有一个最终矩形，其中心点应该就是 final_result.rectCenters[0]
        if (final_result.rectCenters.size() == 1) {
          rect_center_to_draw = final_result.rectCenters[0];
          center_for_drawing_found = true;
        } else { // 如果有多个最终矩形，需要匹配
          cv::Moments m_contour = cv::moments(contours[i]);
          if (m_contour.m00 != 0) {
            cv::Point2f current_contour_center(
                static_cast<float>(m_contour.m10 / m_contour.m00),
                static_cast<float>(m_contour.m01 / m_contour.m00));
            for (const auto &stored_center : final_result.rectCenters) {
              if (cv::norm(stored_center - current_contour_center) < 1.0) { // 允许微小误差
                rect_center_to_draw = stored_center;
                center_for_drawing_found = true;
                break;
              }
            }
          }
        }
        // 如果通过上述方法未找到，作为后备，直接计算当前轮廓的中心
        if (!center_for_drawing_found) {
          cv::Moments m = cv::moments(contours[i]);
          if (m.m00 != 0) {
            rect_center_to_draw = cv::Point2f(static_cast<float>(m.m10 / m.m00),
                                              static_cast<float>(m.m01 / m.m00));
            center_for_drawing_found = true; // 至少我们有一个中心点来绘制
          }
        }

        if (center_for_drawing_found) {
          cv::circle(image_to_draw_on,
                     cv::Point(static_cast<int>(rect_center_to_draw.x),
                               static_cast<int>(rect_center_to_draw.y)),
                     8, cv::Scalar(255, 0, 255), -1); // 紫色中心点

          cv::RotatedRect rot_rect = cv::minAreaRect(contours[i]);
          float width = rot_rect.size.width;
          float height = rot_rect.size.height;
          float aspectRatio =
              (width == 0 || height == 0)
                  ? 0.0f
                  : ((width > height) ? (width / height) : (height / width));
          float area = static_cast<float>(cv::contourArea(contours[i]));

          cv::putText(image_to_draw_on, "Final Rect",
                    cv::Point(static_cast<int>(rect_center_to_draw.x),
                              static_cast<int>(rect_center_to_draw.y)) +
                        cv::Point(10, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 255), 2);
          std::string rect_info =
              "Ratio: " +
              (aspectRatio > 0 ? std::to_string(aspectRatio).substr(0, 4)
                             : "N/A") +
              " Area: " + std::to_string(static_cast<int>(area));
          cv::putText(image_to_draw_on, rect_info,
                      cv::Point(static_cast<int>(rect_center_to_draw.x),
                                static_cast<int>(rect_center_to_draw.y)) +
                          cv::Point(10, 20),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 1);
        }
      }
    }

    cv::Mat resized_image;
    cv::Size newSize(840, 620); 
    if (image_to_draw_on.cols > 0 && image_to_draw_on.rows > 0) { // 确保图像有效
      cv::resize(image_to_draw_on, resized_image, newSize, 0, 0,
               cv::INTER_LINEAR);
      cv::imshow("Detected Key Points (Refactored)", resized_image);
    } else {
      cv::imshow("Detected Key Points (Refactored) - Original Size",
               image_to_draw_on);
    }
    cv::waitKey(1); // 保持窗口更新
    
    // 确保 detect 函数返回的是画完图的图像！
    result.processedImage = image_to_draw_on;
  }

  auto end_time = high_resolution_clock::now();
  result.processingTime = duration_cast<milliseconds>(end_time - start_time).count();
  blade.apex.push_back(final_result.rectCenters[0]);

  return result;
}
// 通过方程求解交点的方法
vector<Point> findIntersectionsByEquation(const Point &center1,
                                          const Point &center2, double radius,
                                          const RotatedRect &ellipse, Mat &pic,
                                          GlobalParam &gp, WMBlade &blade) {
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

  // 在图片右侧显示方程
  if (gp.debug) {
    // 计算显示位置
    int text_x = pic.cols - 500; // 距离右边界600像素
    int text_y = pic.rows / 2;   // 垂直居中
    int line_height = 40;        // 行间距

    // 显示坐标系信息
    string coord_info = "Coordinate System:";
    string coord_info1 = "x: major axis, rotated " +
                         to_string(angle_deg).substr(0, 4) + " degrees";
    string coord_info2 = "y: minor axis, perpendicular to x";
    putText(pic, coord_info, Point(text_x, text_y - 2 * line_height),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    putText(pic, coord_info1, Point(text_x, text_y - line_height),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    putText(pic, coord_info2, Point(text_x, text_y), FONT_HERSHEY_SIMPLEX, 0.8,
            Scalar(255, 255, 255), 2);

    // 显示椭圆方程
    string ellipse_eq = "Ellipse Equation:";
    string ellipse_eq1 = "x^2/" + to_string(a * a).substr(0, 6) + " + y^2/" +
                         to_string(b * b).substr(0, 6) + " = 1";
    putText(pic, ellipse_eq, Point(text_x, text_y + line_height),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    putText(pic, ellipse_eq1, Point(text_x, text_y + 2 * line_height),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

    // 显示直径方程
    string diameter_eq = "Diameter Equation:";
    string diameter_eq1 = to_string(A).substr(0, 6) + "x + " +
                          to_string(B).substr(0, 6) + "y + " +
                          to_string(C).substr(0, 6) + " = 0";
    putText(pic, diameter_eq, Point(text_x, text_y + 3 * line_height),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    putText(pic, diameter_eq1, Point(text_x, text_y + 4 * line_height),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

    // 显示共轭直径方程
    string conjugate_eq = "Conjugate Diameter:";
    string conjugate_eq1;
    if (A != 0) {
      double new_slope = (b * b * B) / (a * a * A);
      conjugate_eq1 =
          to_string(new_slope).substr(0, 6) + "x - y + " +
          to_string(center1.y - new_slope * center1.x).substr(0, 6) + " = 0";
    } else {
      conjugate_eq1 = "x - " + to_string(center1.x).substr(0, 6) + " = 0";
    }
    putText(pic, conjugate_eq, Point(text_x, text_y + 5 * line_height),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    putText(pic, conjugate_eq1, Point(text_x, text_y + 6 * line_height),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
  }

  // 避免除零的情况
  if (fabs(B_rot) < 1e-8) {
    cout << "直线几乎垂直" << endl;
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

    if (gp.debug) {
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

  double A2_rot_conj,
      B2_rot_conj; // 共轭直径在旋转坐标系下的系数 A_conj*x' + B_conj*y' = 0

  const double epsilon = 1e-9; // 用于比较浮点数是否接近0 (原用1e-8，可按需调整)

  bool conjugate_calculation_possible = true;

  if (fabs(A_rot) < epsilon && fabs(B_rot) < epsilon) {
    // A_rot 和 B_rot 都接近0 (比如 center1 和 center2 是同一点)
    // 无法定义第一条直径，因此无法计算共轭直径
    if (gp.debug) {
      cout << "警告: 计算共轭直径时，A_rot 和 B_rot "
              "同时接近0。跳过共轭直径计算。"
           << endl;
    }
    conjugate_calculation_possible = false;
  } else if (fabs(A_rot) < epsilon) {
    // 第一条直径在旋转坐标系下是水平的 (y' approx 0, 因为 B_rot*y' approx 0,
    // B_rot不为0) 其共轭直径在旋转坐标系下是垂直的 (x' = 0)
    A2_rot_conj = 1.0;
    B2_rot_conj = 0.0;
  } else if (fabs(B_rot) < epsilon) {
    // 第一条直径在旋转坐标系下是垂直的 (x' approx 0, 因为 A_rot*x' approx 0,
    // A_rot不为0) 其共轭直径在旋转坐标系下是水平的 (y' = 0)
    A2_rot_conj = 0.0;
    B2_rot_conj = 1.0; //  方向不影响直线本身, 可用 -1.0
  } else {
    // 一般情况: 第一条直径斜率 m'_1 = -A_rot / B_rot
    // 共轭直径斜率 m'_2 = -(b^2/a^2) / m'_1 = (b^2 * B_rot) / (a^2 * A_rot)
    // 方程为: y' = m'_2 * x'  =>  (b^2 * B_rot) * x' - (a^2 * A_rot) * y' = 0
    // 注意避免a或b为0的情况，尽管对于有效椭圆它们应为正
    if (fabs(a) < epsilon || fabs(b) < epsilon) {
      if (gp.debug) {
        cout << "警告: 椭圆半轴长a或b过小，无法安全计算共轭直径。" << endl;
      }
      conjugate_calculation_possible = false;
    } else {
      A2_rot_conj = b * b * B_rot;
      B2_rot_conj = -a * a * A_rot;
    }
  }

  if (conjugate_calculation_possible) {
    Point pt3, pt4; // 共轭直径的两个交点
    bool found_conjugate_intersections = false;

    if (fabs(B2_rot_conj) < epsilon) {
      // 共轭直径在旋转坐标系下是垂直的 (x' = 0), A2_rot_conj 不应为0
      // (除非A_rot,B_rot都为0)
      if (fabs(A2_rot_conj) > epsilon) { // 确保是 x'=0 而不是 0=0
        double x_rot_c = 0.0;
        // 代入椭圆方程: y'^2/b^2 = 1 => y' = +/- b
        if (b > epsilon) { // b 必须为正
          double y1_rot_c = b;
          double y2_rot_c = -b;

          pt3 = Point(cvRound(x_rot_c * cos_theta - y1_rot_c * sin_theta +
                              ellipse_center.x),
                      cvRound(x_rot_c * sin_theta + y1_rot_c * cos_theta +
                              ellipse_center.y));
          pt4 = Point(cvRound(x_rot_c * cos_theta - y2_rot_c * sin_theta +
                              ellipse_center.x),
                      cvRound(x_rot_c * sin_theta + y2_rot_c * cos_theta +
                              ellipse_center.y));
          found_conjugate_intersections = true;
        }
      }
    } else { // 一般情况，B2_rot_conj 不为0
      // 共轭直径方程 y' = -(A2_rot_conj / B2_rot_conj) * x'
      // 代入椭圆: x'^2/a^2 + (-(A2_rot_conj / B2_rot_conj) * x')^2 / b^2 = 1
      // x'^2 * [1/a^2 + A2_rot_conj^2 / (B2_rot_conj^2 * b^2)] = 1
      // 注意避免a, b, B2_rot_conj为0的情况
      if (fabs(a) < epsilon || fabs(b) < epsilon) {
        if (gp.debug)
          cout << "警告: 椭圆半轴a或b为0，无法计算共轭直径交点。" << endl;
      } else {
        double term_A_sq = A2_rot_conj * A2_rot_conj;
        double term_B_sq_b_sq = B2_rot_conj * B2_rot_conj * b * b;

        if (fabs(term_B_sq_b_sq) < epsilon) { // 防止除以0
          if (gp.debug)
            cout << "警告: 计算共轭直径交点时出现分母为0的情况 "
                    "(term_B_sq_b_sq)。"
                 << endl;
        } else {
          double M2_val = (1.0 / (a * a)) + term_A_sq / term_B_sq_b_sq;
          if (M2_val > epsilon) { // M2_val 必须为正才能开方
            double x_val_sq = 1.0 / M2_val;
            double x1_rot_c = sqrt(x_val_sq);
            double x2_rot_c = -sqrt(x_val_sq);

            double y1_rot_c = (-A2_rot_conj * x1_rot_c) / B2_rot_conj;
            double y2_rot_c = (-A2_rot_conj * x2_rot_c) / B2_rot_conj;

            pt3 = Point(cvRound(x1_rot_c * cos_theta - y1_rot_c * sin_theta +
                                ellipse_center.x),
                        cvRound(x1_rot_c * sin_theta + y1_rot_c * cos_theta +
                                ellipse_center.y));
            pt4 = Point(cvRound(x2_rot_c * cos_theta - y2_rot_c * sin_theta +
                                ellipse_center.x),
                        cvRound(x2_rot_c * sin_theta + y2_rot_c * cos_theta +
                                ellipse_center.y));
            found_conjugate_intersections = true;
          } else {
            if (gp.debug)
              cout << "警告: 计算共轭直径交点时 M2_val 非正。" << endl;
          }
        }
      }
    }

    if (found_conjugate_intersections) {
      if (gp.debug) {
        // 绘制共轭直径的两个交点
        circle(pic, pt3, 3, Scalar(255, 255, 0), -1); // 绿色表示共轭直径交点
        circle(pic, pt4, 3, Scalar(0, 255, 255), -1);
        // 可以选择绘制共轭直径本身 (pt3到pt4的连线，如果它们确实是直径的端点)
        // line(pic, pt3, pt4, Scalar(0, 0, 255), 3); // 示例：深青色线
      }

      // 使用原始代码中的排序逻辑，确保对 intersections 和 blade.apex
      // 的添加顺序一致
      Point O_sort = center2; // 参考点进行排序
      Point OP3_vec = pt3 - O_sort;
      Point OP4_vec = pt4 - O_sort;
      Point OP3N_vec(-OP3_vec.y, OP3_vec.x); // OP3_vec 旋转90度
      double dot_product_sort = OP3N_vec.x * OP4_vec.x + OP3N_vec.y * OP4_vec.y;

      if (dot_product_sort < 0) { // pt3 "在先"
        intersections.emplace_back(pt3);
        blade.apex.push_back(pt3);
        intersections.emplace_back(pt4);
        blade.apex.push_back(pt4);
      } else { // pt4 "在先"
        intersections.emplace_back(pt4);
        blade.apex.push_back(pt4);
        intersections.emplace_back(pt3);
        blade.apex.push_back(pt3);
      }
    }
  }
  //cv::imshow("Intersections Visualization", pic);

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
