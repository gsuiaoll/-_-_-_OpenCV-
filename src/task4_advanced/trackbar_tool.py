"""
进阶任务三：交互调参工具
基于OpenCV的Trackbar组件创建可视化调参界面
支持实时调节HSV阈值、模糊核大小等关键参数
"""

import cv2
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _get_screen_size():
    """获取屏幕分辨率，Windows下使用ctypes，失败则返回常见默认值"""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080
from src.utils import read_image, save_image


class TrackbarTool:
    """Trackbar可视化调参工具类"""

    CONTROL_WINDOW = "HSV Controls"
    DISPLAY_WINDOW = "HSV Result"

    def __init__(self, image_path):
        self.image = read_image(image_path)
        if self.image is None:
            raise FileNotFoundError(f"无法加载图片: {image_path}")

        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # 获取屏幕分辨率，动态计算两个窗口的尺寸和位置
        screen_w, screen_h = _get_screen_size()
        # 控制窗口宽度固定约500px，显示窗口尽量占满剩余横向空间
        self.control_w = 500
        self.display_max_w = max(800, screen_w - self.control_w - 80)
        self.display_max_h = max(500, screen_h - 120)

        # 创建两个窗口：控制窗口专门放滑块，显示窗口专门放图片
        # 这是OpenCV下调参工具的标准做法，可避免单窗口内滑块和图片互相挤压
        cv2.namedWindow(self.CONTROL_WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.DISPLAY_WINDOW, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.CONTROL_WINDOW, 20, 20)
        cv2.moveWindow(self.DISPLAY_WINDOW, self.control_w + 40, 20)

        # 在控制窗口创建Trackbar
        cv2.createTrackbar("H Lower", self.CONTROL_WINDOW, 0, 180, self.nothing)
        cv2.createTrackbar("H Upper", self.CONTROL_WINDOW, 180, 180, self.nothing)
        cv2.createTrackbar("H Lower2", self.CONTROL_WINDOW, 0, 180, self.nothing)
        cv2.createTrackbar("H Upper2", self.CONTROL_WINDOW, 0, 180, self.nothing)
        cv2.createTrackbar("S Lower", self.CONTROL_WINDOW, 0, 255, self.nothing)
        cv2.createTrackbar("S Upper", self.CONTROL_WINDOW, 255, 255, self.nothing)
        cv2.createTrackbar("V Lower", self.CONTROL_WINDOW, 0, 255, self.nothing)
        cv2.createTrackbar("V Upper", self.CONTROL_WINDOW, 255, 255, self.nothing)
        cv2.createTrackbar("BlurK", self.CONTROL_WINDOW, 1, 21, self.nothing)
        cv2.createTrackbar("MorphK", self.CONTROL_WINDOW, 1, 21, self.nothing)
        cv2.createTrackbar("MinArea", self.CONTROL_WINDOW, 100, 2000, self.nothing)

        # 设置初始值（蓝色示例，H2 范围默认关闭）
        cv2.setTrackbarPos("H Lower", self.CONTROL_WINDOW, 100)
        cv2.setTrackbarPos("H Upper", self.CONTROL_WINDOW, 130)
        cv2.setTrackbarPos("S Lower", self.CONTROL_WINDOW, 120)
        cv2.setTrackbarPos("V Lower", self.CONTROL_WINDOW, 100)

        # 显示一个小提示，确保控制窗口能被系统正确渲染
        control_panel = np.zeros((60, 500, 3), dtype=np.uint8)
        cv2.putText(control_panel, "Drag sliders to tune", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(self.CONTROL_WINDOW, control_panel)

    def nothing(self, x):
        """Trackbar回调函数，不需要额外操作"""
        pass

    def _resize_for_display(self, image):
        """
        根据屏幕分辨率缩放显示图像，使显示窗口尽可能大且不超出屏幕
        :param image: 原始图像
        :return: 缩放后的图像
        """
        h, w = image.shape[:2]
        max_width = getattr(self, 'display_max_w', 1600)
        max_height = getattr(self, 'display_max_h', 900)
        scale = min(1.0, max_width / w, max_height / h)
        if scale >= 1.0:
            return image
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h))

    def get_trackbar_values(self):
        """读取当前Trackbar的值"""
        h_lower = cv2.getTrackbarPos("H Lower", self.CONTROL_WINDOW)
        h_upper = cv2.getTrackbarPos("H Upper", self.CONTROL_WINDOW)
        h_lower2 = cv2.getTrackbarPos("H Lower2", self.CONTROL_WINDOW)
        h_upper2 = cv2.getTrackbarPos("H Upper2", self.CONTROL_WINDOW)
        s_lower = cv2.getTrackbarPos("S Lower", self.CONTROL_WINDOW)
        s_upper = cv2.getTrackbarPos("S Upper", self.CONTROL_WINDOW)
        v_lower = cv2.getTrackbarPos("V Lower", self.CONTROL_WINDOW)
        v_upper = cv2.getTrackbarPos("V Upper", self.CONTROL_WINDOW)
        blur_k = cv2.getTrackbarPos("BlurK", self.CONTROL_WINDOW)
        morph_k = cv2.getTrackbarPos("MorphK", self.CONTROL_WINDOW)
        min_area = cv2.getTrackbarPos("MinArea", self.CONTROL_WINDOW)

        # 模糊核和形态学核必须为奇数
        blur_k = max(1, blur_k)
        if blur_k % 2 == 0:
            blur_k += 1
        morph_k = max(1, morph_k)
        if morph_k % 2 == 0:
            morph_k += 1

        return {
            'h_lower': h_lower, 'h_upper': h_upper,
            'h_lower2': h_lower2, 'h_upper2': h_upper2,
            's_lower': s_lower, 's_upper': s_upper,
            'v_lower': v_lower, 'v_upper': v_upper,
            'blur_k': blur_k, 'morph_k': morph_k,
            'min_area': min_area
        }

    def process(self, params):
        """
        根据当前参数处理图像
        :param params: Trackbar参数字典
        :return: 处理结果图、掩码图
        """
        # 高斯模糊
        if params['blur_k'] > 1:
            blurred_hsv = cv2.GaussianBlur(self.hsv, (params['blur_k'], params['blur_k']), 0)
        else:
            blurred_hsv = self.hsv

        # HSV阈值分割（支持双段H范围，如红色跨越0°/180°）
        lower1 = np.array([params['h_lower'], params['s_lower'], params['v_lower']])
        upper1 = np.array([params['h_upper'], params['s_upper'], params['v_upper']])
        mask = cv2.inRange(blurred_hsv, lower1, upper1)
        if params['h_upper2'] > params['h_lower2']:
            lower2 = np.array([params['h_lower2'], params['s_lower'], params['v_lower']])
            upper2 = np.array([params['h_upper2'], params['s_upper'], params['v_upper']])
            mask = cv2.bitwise_or(mask, cv2.inRange(blurred_hsv, lower2, upper2))

        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (params['morph_k'], params['morph_k']))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 查找轮廓并过滤
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = self.image.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < params['min_area']:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(result, f"A:{int(area)}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 合并原图、掩码、结果
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([self.image, mask_color, result])

        # 显示当前参数
        h2_info = f" H2:[{params['h_lower2']},{params['h_upper2']}]" if params['h_upper2'] > params['h_lower2'] else ""
        info = (f"H:[{params['h_lower']},{params['h_upper']}]{h2_info} "
                f"S:[{params['s_lower']},{params['s_upper']}] "
                f"V:[{params['v_lower']},{params['v_upper']}] "
                f"Blur:{params['blur_k']} Morph:{params['morph_k']} MinArea:{params['min_area']}")
        cv2.putText(combined, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return combined, mask

    def run(self):
        """启动调参界面主循环"""
        print("=" * 50)
        print("OpenCV HSV 交互调参工具")
        print("控制窗口:", self.CONTROL_WINDOW)
        print("显示窗口:", self.DISPLAY_WINDOW)
        print("按 'q' 退出")
        print("按 's' 保存当前参数和结果到 test_images/results/task4/trackbar/")
        print("=" * 50)
        while True:
            params = self.get_trackbar_values()
            combined, mask = self.process(params)
            # 控制窗口和图片窗口分离，图片可以独立放大显示
            display = self._resize_for_display(combined)
            cv2.imshow(self.DISPLAY_WINDOW, display)
            # 强制显示窗口尺寸与图片一致，避免WINDOW_NORMAL下图片被拉伸变形
            cv2.resizeWindow(self.DISPLAY_WINDOW, display.shape[1], display.shape[0])

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                self.save_current_state(params, combined, mask)

            # 检测任一窗口被关闭时退出，避免程序在后台空转
            try:
                if (cv2.getWindowProperty(self.CONTROL_WINDOW, cv2.WND_PROP_VISIBLE) < 1 or
                    cv2.getWindowProperty(self.DISPLAY_WINDOW, cv2.WND_PROP_VISIBLE) < 1):
                    break
            except cv2.error:
                break

        cv2.destroyAllWindows()
        print("调参工具已关闭")

    def save_current_state(self, params, combined, mask):
        """保存当前参数和结果图"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(project_root, "test_images", "results", "task4", "trackbar")
        os.makedirs(output_dir, exist_ok=True)

        # 保存结果图（使用save_image支持中文路径）
        save_image(os.path.join(output_dir, "trackbar_result.jpg"), combined)
        save_image(os.path.join(output_dir, "trackbar_mask.jpg"), mask)

        # 保存参数
        with open(os.path.join(output_dir, "trackbar_params.txt"), 'w', encoding='utf-8') as f:
            f.write("# Trackbar 调参结果\n")
            for k, v in params.items():
                f.write(f"{k}: {v}\n")
        print(f"[保存] 参数和结果已保存到 {output_dir}")


def trackbar_tool_pipeline(image_path):
    """
    Trackbar调参工具入口
    :param image_path: 输入图片路径
    """
    tool = TrackbarTool(image_path)
    tool.run()


if __name__ == "__main__":
    # 基于当前文件位置计算项目根目录，支持从任意目录运行
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_image = os.path.join(project_root, "test_images", "original", "images", "color_test.jpg")
    trackbar_tool_pipeline(input_image)
