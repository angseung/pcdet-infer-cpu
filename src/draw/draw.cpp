#include "params.h"
#include "type.h"
#include <cmath>
#include <draw/draw.h>
#include <vector>

cv::Point2f rotatePoint(const cv::Point2f &point, float angle) {
    float x = point.x * std::cos(angle) - point.y * std::sin(angle);
    float y = point.x * std::sin(angle) + point.y * std::cos(angle);
    return cv::Point2f(x, y);
}

cv::Mat drawBirdsEyeView(size_t points_size, const float *points_data,
                         const std::vector<vueron::BndBox> &boxes,
                         const std::vector<float> &scores,
                         const std::vector<size_t> &labels) {
    float scale = 8.0; // 1m를 몇 픽셀로 표현할지 결정하는 스케일

    int width = static_cast<int>((MAX_X_RANGE - MIN_X_RANGE) * scale);
    int height = static_cast<int>((MAX_Y_RANGE - MIN_Y_RANGE) * scale);

    cv::Mat image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    // 포인트 클라우드 그리기
    for (size_t i = 0; i < points_size; ++i) {
        int x = static_cast<int>(
            (points_data[i * NUM_POINT_VALUES] - MIN_X_RANGE) * scale);
        int y = static_cast<int>(
            (MAX_Y_RANGE - points_data[i * NUM_POINT_VALUES + 1]) * scale);
        cv::circle(image, cv::Point(x, y), 0, cv::Scalar(255, 255, 255), 1);
    }

    // 바운딩 박스 그리기
    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i] < 0.5)
            continue;
        cv::Scalar color;
        switch (labels[i]) {
        case 1:
            color = cv::Scalar(0, 0, 255); // 빨간색
            break;
        case 2:
            color = cv::Scalar(0, 255, 0); // 초록색
            break;
        case 3:
            color = cv::Scalar(255, 0, 0); // 파랑색
            break;
        default:
            color = cv::Scalar(0, 255, 255); // 기본 색상 (노란색)
            break;
        }
        vueron::BndBox box = boxes[i];
        cv::Point2f center((box.x - MIN_X_RANGE) * scale,
                           (MAX_Y_RANGE - box.y) * scale);
        cv::Point2f vertices[4];
        vertices[0] = cv::Point2f(box.dx / 2, box.dy / 2);
        vertices[1] = cv::Point2f(box.dx / 2, -box.dy / 2);
        vertices[2] = cv::Point2f(-box.dx / 2, -box.dy / 2);
        vertices[3] = cv::Point2f(-box.dx / 2, box.dy / 2);

        for (int i = 0; i < 4; ++i) {
            vertices[i] = rotatePoint(vertices[i], -box.heading);
            vertices[i].x *= scale;
            vertices[i].y *= scale;
            vertices[i] += center;
        }

        for (int i = 0; i < 4; ++i) {
            cv::line(image, vertices[i], vertices[(i + 1) % 4], color, 1);
        }
    }

    return image;
}

// 사용 예:
// drawBirdsEyeView(points_data, points_size, nms_pred_from_wrapper);
