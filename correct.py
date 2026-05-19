import cv2
import numpy as np

# ===================== 配置参数（和你的相机分辨率一致）=====================
CAMERA_WIDTH = 640   # 相机画面宽度
CAMERA_HEIGHT = 480  # 相机画面高度
# =========================================================================

# 计算相机画面中心点（固定）
center_x = CAMERA_WIDTH // 2
center_y = CAMERA_HEIGHT // 2

# 存储激光点坐标
laser_x = None
laser_y = None

# 鼠标回调函数：左键点击获取激光点坐标
def mouse_click(event, x, y, flags, param):
    global laser_x, laser_y
    if event == cv2.EVENT_LBUTTONDOWN:
        laser_x = x
        laser_y = y

# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# 创建窗口并绑定鼠标事件
cv2.namedWindow("Laser-Camera Calibration")
cv2.setMouseCallback("Laser-Camera Calibration", mouse_click)

print("===== 校准操作说明 =====")
print("1. 将激光笔对准地面红色圆点")
print("2. 在画面中找到激光红点，鼠标左键点击它")
print("3. 程序自动计算 激光 ↔ 相机中心 的像素距离")
print("4. 按 q 退出程序")
print("========================\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("摄像头读取失败！")
        break

    # ---------------------- 绘制相机中心十字线（绿色）----------------------
    cv2.line(frame, (center_x, 0), (center_x, CAMERA_HEIGHT), (0, 255, 0), 2)
    cv2.line(frame, (0, center_y), (CAMERA_WIDTH, center_y), (0, 255, 0), 2)
    cv2.putText(frame, "Camera Center", (center_x+10, center_y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ---------------------- 如果点击了激光点，绘制并计算 ----------------------
    if laser_x is not None and laser_y is not None:
        # 绘制激光点（红色圆圈标记）
        cv2.circle(frame, (laser_x, laser_y), 5, (0, 0, 255), -1)
        cv2.putText(frame, "Laser Point", (laser_x+10, laser_y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 核心：计算 激光点 - 相机中心 的像素偏移量
        dx = laser_x - center_x  # 水平像素距离（右正左负）
        dy = laser_y - center_y  # 垂直像素距离（下正上负）

        # 显示计算结果
        cv2.putText(frame, f"DX = {dx} px", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"DY = {dy} px", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Distance: {np.sqrt(dx**2 + dy**2):.1f} px", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 控制台打印最终校准值（直接复制用！）
        print(f"\n✅ 校准完成！")
        print(f"相机中心：({center_x}, {center_y})")
        print(f"激光点：({laser_x}, {laser_y})")
        print(f"激光-相机中心 像素偏移：dx = {dx} , dy = {dy}")
        print("⚠️  请记录这两个值，用于无人机视觉定位校准！\n")

    # 显示画面
    cv2.imshow("Laser-Camera Calibration", frame)

    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()