"""
任务三（上）：简单几何图形识别
实现功能：
1. 识别图片中的矩形、圆形等基础几何图形
2. 在图形上标注识别结果
3. 输出原图与识别结果对比图
"""

import cv2
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import read_image, save_image, create_comparison_image


def preprocess_for_contours(image):
    """
    对图像进行预处理，便于后续轮廓检测
    :param image: BGR彩色图像
    :return: 边缘检测后的二值图
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # 膨胀连接断裂边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def detect_shapes(image, min_area=300):
    """
    基于轮廓近似识别矩形、三角形、圆形、多边形
    :param image: BGR彩色图像
    :param min_area: 最小有效轮廓面积
    :return: 标注后的图像和图形信息列表
    """
    edges = preprocess_for_contours(image)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()
    shapes = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # 计算轮廓周长
        peri = cv2.arcLength(cnt, True)
        # 多边形近似
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        vertices = len(approx)

        # 计算外接矩形、长宽比和最小外接圆
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h if h > 0 else 0
        center_x, center_y = x + w // 2, y + h // 2

        # 圆度判断：面积与周长关系
        circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

        shape_name = "unknown"
        if vertices == 3:
            shape_name = "triangle"
        elif vertices == 4:
            # 根据长宽比判断矩形还是正方形
            shape_name = "square" if 0.9 <= aspect_ratio <= 1.1 else "rectangle"
        elif vertices == 5:
            shape_name = "pentagon"
        elif vertices >= 6:
            # 边数>=6时，结合圆度判断是否为圆
            if circularity > 0.7:
                shape_name = "circle"
            else:
                shape_name = "polygon"

        shapes.append({
            'id': i + 1,
            'shape': shape_name,
            'vertices': vertices,
            'area': int(area),
            'center': (center_x, center_y),
            'circularity': round(circularity, 3)
        })

        # 绘制识别结果
        color = (0, 255, 0)
        cv2.drawContours(result, [approx], -1, color, 2)
        cv2.circle(result, (center_x, center_y), 5, (0, 0, 255), -1)
        label = f"{shape_name}#{i+1}"
        cv2.putText(result, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return result, shapes


def shape_recognition_pipeline(image_path, output_dir, min_area=300):
    """
    几何图形识别主流程
    :param image_path: 输入图片路径
    :param output_dir: 结果输出目录
    :param min_area: 最小有效面积
    """
    original = read_image(image_path)
    if original is None:
        return

    result, shapes = detect_shapes(original, min_area)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    save_image(os.path.join(output_dir, f"{base_name}_edges.jpg"), preprocess_for_contours(original))
    save_image(os.path.join(output_dir, f"{base_name}_shapes.jpg"), result)

    comparison = create_comparison_image(
        [original, result],
        ["Original", "Shape Recognition"],
        cols=2
    )
    save_image(os.path.join(output_dir, f"{base_name}_shape_comparison.jpg"), comparison)

    print(f"\n===== {base_name} 几何图形识别结果 =====")
    print(f"共识别到 {len(shapes)} 个图形：")
    for s in shapes:
        print(f"  #{s['id']} {s['shape']} 顶点数={s['vertices']} 面积={s['area']} 中心={s['center']}")


if __name__ == "__main__":
    input_image = os.path.join("test_images", "original", "images", "shape_number_test.jpg")
    output_dir = os.path.join("test_images", "results", "task3")
    shape_recognition_pipeline(input_image, output_dir)
