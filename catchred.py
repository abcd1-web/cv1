import cv2
import numpy as np

def main():
    # 打开默认摄像头（0是第一个摄像头）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # 标定激光点在相机坐标系下的坐标
    laser_pixel_x = 345.0
    laser_pixel_y = 210.0

    deadzone = 5
    # 相机主点（如果需要更精确的坐标可以修改）
    principal_point_x = 320.0
    principal_point_y = 240.0

    MIN_AREA = 500    # 最小面积（排除噪点，根据你的目标大小调整）
    MAX_AREA = 200000 # 最大面积（排除背景大物体）

    # 红色激光点的HSV颜色范围（根据实际环境调整）
    lower_red = np.array([0, 90, 40])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 90, 40])
    upper_red2 = np.array([180, 255, 255])

    # 形态学操作核
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    print("激光点跟踪程序已启动")
    print("按 ESC 键退出程序")
    print("-" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Could not read frame.")
            continue

        img = frame.copy()

        # 图像预处理
        frame_blur = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        frame_enhanced = cv2.convertScaleAbs(frame_blur, alpha=1.2, beta=30)
        
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 自适应直方图均衡化，提升暗处细节
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v_eq = clahe.apply(v)
        hsv_eq = cv2.merge((h, s, v_eq))

        # 红色颜色分割
        mask1 = cv2.inRange(hsv_eq, lower_red, upper_red)
        mask2 = cv2.inRange(hsv_eq, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 形态学操作去噪
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.GaussianBlur(mask, (5,5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=1)
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 面积过滤
            if not (MIN_AREA < area < MAX_AREA):
                continue
            
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * (radius ** 2)
            if circle_area == 0:
                continue
            
            # 圆形度过滤（激光点近似圆形）
            circularity_ratio = area / circle_area
            if circularity_ratio < 0.5:
                continue
            
            valid_contours.append(cnt)
        error_x = 0
        error_y = 0
        if len(valid_contours) > 0:
            # 取面积最大的轮廓作为目标
            largest_cnt = max(valid_contours, key=cv2.contourArea)
            (cx, cy), radius = cv2.minEnclosingCircle(largest_cnt)
            center = (int(cx), int(cy))
            radius = int(radius)

            # 计算误差（误差的符号和相机安装有关，正值物在右下，负值物在左上）
            error_x = cx - laser_pixel_x
            error_y = cy - laser_pixel_y

            # 死区处理
            if abs(error_x) < deadzone:
                error_x = 0
            if abs(error_y) < deadzone:
                error_y = 0

            # 在图像上画出目标轮廓和中心
            cv2.circle(img, center, radius, (0, 255, 0), 2)
            cv2.circle(img, center, 3, (0, 255, 0), -1)
            cv2.putText(img, f"Target: ({int(cx)}, {int(cy)})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 在图像上画出标定的激光点位置
        cv2.drawMarker(img, (int(laser_pixel_x), int(laser_pixel_y)), 
                      (255, 0, 0), markerType=cv2.MARKER_CROSS, 
                      markerSize=20, thickness=2)

        # 显示误差信息
        cv2.putText(img, f"Error X: {error_x:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"Error Y: {error_y:.1f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 控制台输出误差
        print(f"\r误差 X: {error_x:6.1f} | 误差 Y: {error_y:6.1f}", end="")

        # 显示图像
        cv2.imshow("Processed Image", img)
        cv2.imshow("Mask", mask)

        # 按ESC键退出
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("\n程序已退出")

if __name__ == '__main__':
    main()