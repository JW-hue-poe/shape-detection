# =========================================================
# 2020 电赛 G题  K230/CanMV 性能优化版
# 识别：圆形 / 正方形 / 正三角形
# 输出数字编号：0无目标 1圆 2正方形 3三角形
# 串口：shape,cx,cy,size\r\n  方便STM32解析
# =========================================================
from media.sensor import *
from media.display import *
from media.media import *
from machine import UART
import image, time, math

# ==================== 硬件初始化 ====================
sensor = Sensor()
sensor.reset()
sensor.set_framesize(Sensor.QVGA)    # 320x240
sensor.set_pixformat(Sensor.RGB565)
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)
sensor.run()

Display.init(Display.ST7701)
uart = UART(UART.UART1, 115200)
clock = time.clock()

# ==================== 全局参数 ====================
ROI = (40, 30, 240, 180)  # 限制检测区域
AREA_MIN = 1200            # 最小面积过滤杂波

# 形状编号
SHAPE_NONE   = 0
SHAPE_CIRCLE = 1
SHAPE_SQUARE = 2
SHAPE_TRIANGLE = 3

# 稳定性参数
STABLE_FRAMES = 3          # 连续帧数确认
SEND_INTERVAL = 5          # 发送间隔帧数

# ==================== 串口发送 ====================
def send_data(shape, cx, cy, size):
    """发送形状数据到STM32"""
    try:
        buf = f"{shape},{cx},{cy},{size}\r\n".encode()
        uart.write(buf)
    except:
        pass

# ==================== 精准形状识别（核心优化版） ====================
def detect_shape(img, blob):
    """
    优化的形状识别算法
    返回：SHAPE_CIRCLE / SHAPE_SQUARE / SHAPE_TRIANGLE / SHAPE_NONE
    """
    perimeter = blob.perimeter()
    if perimeter < 5:
        return SHAPE_NONE
    
    area = blob.pixels()
    if area == 0:
        return SHAPE_NONE
    
    # ========== 1. 圆度计算 (0~1, 越接近1越圆) ==========
    circularity = (4 * math.pi * area) / (perimeter ** 2)
    
    # ========== 2. 轮廓多边形逼近 提取角点 ==========
    contour = blob.contour()
    if not contour:
        return SHAPE_NONE
    
    # 降低容差以获得更精确的角点
    approx = contour.approx(0.02 * perimeter)
    corner_cnt = len(approx)
    
    # ========== 3. 尺寸比例与填充率 ==========
    w = blob.w()
    h = blob.h()
    max_side = max(w, h)
    min_side = min(w, h)
    
    if max_side == 0:
        return SHAPE_NONE
    
    ratio = min_side / max_side  # 宽高比 (0~1)
    fill = area / (w * h) if (w * h) != 0 else 0  # 填充率
    
    # ========== 4. 优化的判定逻辑 ==========
    # 优先级：圆 > 三角 > 正方形
    
    # 圆形：圆度高 + 填充率高 + 宽高比接近1
    if circularity > 0.8 and fill > 0.75 and ratio > 0.75:
        return SHAPE_CIRCLE
    
    # 三角形：3个角点 + 合理的填充率
    if corner_cnt == 3 and fill > 0.6:
        return SHAPE_TRIANGLE
    
    # 正方形：4个角点 + 宽高比接近1 + 填充率高
    if corner_cnt == 4 and ratio > 0.8 and fill > 0.7:
        return SHAPE_SQUARE
    
    # 兜底：根据宽高比判定
    if ratio > 0.8:
        return SHAPE_SQUARE
    else:
        return SHAPE_TRIANGLE
    
    return SHAPE_NONE

# ==================== 稳定性管理 ====================
class ShapeDetector:
    def __init__(self):
        self.last_shape = SHAPE_NONE
        self.shape_count = 0
        self.send_count = 0
        self.last_cx = 0
        self.last_cy = 0
        self.last_size = 0
    
    def update(self, shape, cx, cy, size):
        """更新检测结果，返回是否需要发送"""
        # 形状稳定判定
        if shape == self.last_shape:
            self.shape_count += 1
        else:
            self.shape_count = 0
            self.last_shape = shape
        
        # 发送间隔控制
        self.send_count += 1
        
        # 只在形状稳定且达到发送间隔时才发送
        if self.shape_count >= STABLE_FRAMES and self.send_count >= SEND_INTERVAL:
            self.last_cx = cx
            self.last_cy = cy
            self.last_size = size
            self.send_count = 0
            return True
        
        return False
    
    def get_last(self):
        return self.last_shape, self.last_cx, self.last_cy, self.last_size

detector = ShapeDetector()

# ==================== 主循环 ====================
while True:
    clock.tick()
    img = sensor.snapshot()
    
    # ========== 预处理：灰度+滤波+二值化 ==========
    # 复制ROI区域进行处理
    roi_x, roi_y, roi_w, roi_h = ROI
    gray = img.copy(roi=(roi_x, roi_y, roi_w, roi_h)).to_grayscale()
    
    # 高斯滤波（小核更快）
    gray.gaussian(1)
    
    # 自适应二值化
    hist = gray.get_histogram()
    thres = hist.get_threshold().value()
    gray.binary([(thres - 15, 255)])  # 扩大范围捕获更多目标
    
    # 形态学处理：去除噪点
    gray.erode(1)   # 腐蚀（去除小噪点）
    gray.dilate(1)  # 膨胀（恢复目标）
    
    # ========== 查找白色目标物体 ==========
    blobs = gray.find_blobs(
        [(200, 255)],
        area_threshold=AREA_MIN,
        pixels_threshold=100,  # 最小像素数
        merge=True,
        margin=10  # 合并边距
    )
    
    shape_flag = SHAPE_NONE
    cx, cy, obj_size = 0, 0, 0
    
    if blobs:
        # 选择面积最大的blob
        blob = max(blobs, key=lambda b: b.pixels())
        x, y, w, h = blob.rect()
        
        # 坐标转换（从ROI坐标转为图像坐标）
        cx = x + roi_x + w // 2
        cy = y + roi_y + h // 2
        obj_size = max(w, h)
        
        # 识别形状
        shape_flag = detect_shape(img, blob)
        
        # 绘制检测框和十字
        img.draw_rectangle(x + roi_x, y + roi_y, w, h, color=(255, 0, 0), thickness=1)
        img.draw_cross(cx, cy, color=(255, 255, 0), size=6)
    
    # ========== 稳定性判定与串口发送 ==========
    should_send = detector.update(shape_flag, cx, cy, obj_size)
    
    if should_send:
        send_data(shape_flag, cx, cy, obj_size)
    
    # 获取最后发送的数据用于显示
    display_shape, display_cx, display_cy, display_size = detector.get_last()
    
    # ========== 界面绘制 ==========
    # 画ROI区域
    img.draw_rectangle(ROI, color=(0, 255, 0), thickness=1)
    
    # 文字显示
    shape_names = ["None", "Circle", "Square", "Triangle"]
    shape_name = shape_names[display_shape]
    
    img.draw_string(5, 5, f"Shape:{shape_name}", color=(255, 0, 0), scale=1)
    img.draw_string(5, 20, f"X:{display_cx} Y:{display_cy}", color=(0, 255, 0), scale=1)
    img.draw_string(5, 35, f"Size:{display_size}", color=(0, 0, 255), scale=1)
    img.draw_string(5, 50, f"FPS:{clock.fps():.1f}", color=(255, 255, 255), scale=1)
    
    # 显示到屏幕
    Display.show_image(img)

# =========================================================
# 性能优化说明：
# 1. ROI处理：只处理感兴趣区域，减少50%计算量
# 2. 高斯核优化：从2改为1，保留更多细节，更快
# 3. 防抖稳定：连续STABLE_FRAMES帧确认才发送
# 4. 串口间隔：每SEND_INTERVAL帧发送一次，降低I/O开销
# 5. 形态学处理：腐蚀+膨胀，有效去除噪点
# 6. 改进判定逻辑：更精确的圆度、角点、填充率判定
# 7. 错误处理：串口异常不会卡死程序
# =========================================================
