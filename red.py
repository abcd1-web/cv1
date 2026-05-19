import numpy as np
import cv2
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
MIN_AREA = 500    # 最小面积（排除噪点，根据你的目标大小调整）
MAX_AREA = 200000  # 最大面积（排除背景大物体）
MIN_CIRCULARITY = 0.7  # 圆形度，过滤非椭圆形状
MAX_CIRCULARITY = 1

lower_= np.array([0, 90, 40])
lower_re = np.array([0, 90, 40])
lower_red = np.array([0, 90, 40])
upper_red = np.array([10, 255, 255])
lower_red2 = np.array([170, 90, 40])
upper_red2 = np.array([180, 255, 255])
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img =frame.copy()
#    frame_enhanced = cv2.convertScaleAbs(frame, alpha=1.3, beta=40)  
#    img = frame_enhanced.copy()  # 后续用增强后的图像处理
    frame_blur = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
    frame_enhanced = cv2.convertScaleAbs(frame_blur, alpha=1.2, beta=30)
#    frame_enhanced = cv2.fastNlMeansDenoisingColored(
#    frame_enhanced, h=5, hColor=5, templateWindowSize=3, searchWindowSize=11
#      )
    hsv = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 自适应直方图均衡化，提升暗处细节，避免过曝
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v_eq = clahe.apply(v)

    hsv_eq = cv2.merge((h, s, v_eq))  # 合并增强后的亮度通道
    mask1 = cv2.inRange(hsv_eq, lower_red, upper_red)
    mask2 = cv2.inRange(hsv_eq, lower_red2, upper_red2)

    mask = cv2.bitwise_or(mask1, mask2)
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
        circularity_ratio = area / circle_area  # 轮廓面积/外接圆面积，越接近1越圆
        if circularity_ratio < 0.5:  # 调低阈值，0.5-0.6都可，根据效果调整
            continue
#        perimeter = cv2.arcLength(cnt, True)
#       if perimeter == 0:
#            continue
#        circularity = 4 * np.pi * area / (perimeter ** 2)
#        if circularity < MIN_CIRCULARITY:
#            continue
#        if circularity > MAX_CIRCULARITY:
#            continue
        valid_contours.append(cnt)     
    if len(valid_contours) > 0:
        largest_cnt = max(valid_contours, key=cv2.contourArea)
#        x, y, w, h = cv2.boundingRect(largest_cnt)
#        cx = x + w//2
#        cy = y + h//2
        # 画轮廓 + 圆心
        (cx, cy), radius = cv2.minEnclosingCircle(largest_cnt)
        center = (int(cx), int(cy))
        radius = int(radius)
#        cv2.drawContours(img, [largest_cnt], -1, (0,255,0), 2)
        cv2.circle(img, center, radius, (0, 255, 0), 2)
        cv2.circle(img, center, 5, (0, 0, 255), -1)
#        cv2.circle(img, (cx, cy), 5, (0,0,255), -1)
#        cv2.putText(img, "Target", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imshow("Threshold (Debug)",mask)  # 显示二值化结果，方便调参
    cv2.imshow("Original Frame", img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()