from PIL import Image, ImageDraw
import json
import math
import numpy as np

# 创建更大分辨率的画布用于降采样
scale = 2  # 缩放因子
final_size = 512
temp_size = final_size * scale
image = Image.new('RGBA', (temp_size, temp_size), (0, 0, 0, 0))
draw = ImageDraw.Draw(image)

# 读取JSON文件
with open('more-info.json', 'r') as f:
    shapes = json.load(f)

def draw_shape_element(draw, shape_type, bbox, fill=None, outline=None):
    """基础图形绘制函数"""
    if shape_type == 0:  # Ellipse
        draw.ellipse(bbox, fill=fill, outline=outline)
    elif shape_type == 1:  # Rectangle
        draw.rectangle(bbox, fill=fill, outline=outline)
    elif shape_type == 2:  # Triangle
        # 计算三角形的三个顶点
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # 三角形顶点：底边两端和顶部中点
        points = [
            (x1, y2),           # 左下角
            (x1 + width/2, y1), # 顶部中点
            (x2, y2)            # 右下角
        ]
        draw.polygon(points, fill=fill, outline=outline)

def create_shape_layer(shape_type, bbox, color, rotation=0):
    """创建RGBA图形图层"""
    layer = Image.new('RGBA', (temp_size, temp_size), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    
    # 绘制主图形
    draw_shape_element(layer_draw, shape_type, bbox, fill=color)
    
    # 绘制边缘模糊效果
    expand = scale
    for i in range(expand):
        alpha = int(255 * (1 - (i / expand)))
        edge_bbox = [
            bbox[0] - i,
            bbox[1] - i,
            bbox[2] + i,
            bbox[3] + i
        ]
        edge_color = color[:-1] + (alpha,)
        draw_shape_element(layer_draw, shape_type, edge_bbox, outline=edge_color)
    
    if rotation != 0:
        center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
        layer = layer.rotate(rotation, center=center, resample=Image.BICUBIC)
    
    return layer

def create_mask_shape(shape_type, bbox, alpha, rotation=0):
    """创建单通道蒙版形状"""
    mask = Image.new('L', (temp_size, temp_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    
    draw_shape_element(mask_draw, shape_type, bbox, fill=alpha)
    
    if rotation != 0:
        center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
        mask = mask.rotate(rotation, center=center, resample=Image.BICUBIC)
    
    return mask

def get_bbox_from_bounds(bounds, size):
    """从标准化边界获取实际像素坐标"""
    x = int(bounds['X'] * size)
    y = int(bounds['Y'] * size)
    width = int(bounds['Width'] * size)
    height = int(bounds['Height'] * size)
    return [x, y, x + width, y + height]

def apply_mask_alpha(layer, mask_shape, mask_bbox):
    """应用蒙版的M值与原图层alpha的乘性混合"""
    # 获取M值并转换为0-1范围的系数
    m_value = mask_shape.get('M', 0) / 255.0
    
    # 创建蒙版形状
    mask = create_mask_shape(mask_shape['Type'], mask_bbox, 255, mask_shape.get('Rotation', 0))
    
    # 将图像转换为numpy数组进行快速处理
    mask_array = np.array(mask)
    alpha_array = np.array(layer.getchannel('A'))
    
    # 创建新的alpha通道数组
    new_alpha = alpha_array.copy()
    
    # 在蒙版区域内应用混合
    mask_region = mask_array > 0
    new_alpha[mask_region] = (alpha_array[mask_region] * (1 - m_value)).astype(np.uint8)
    
    # 应用新的alpha通道
    layer.putalpha(Image.fromarray(new_alpha))
    return layer

# 渲染每个形状
for shape in shapes:
    # 获取标准化的位置和大小，并转换为实际像素
    bbox = get_bbox_from_bounds(shape['NormalizedBounds'], temp_size)
    
    # 获取颜色和其他属性
    color = shape['SerializedColor']
    rgba = (color['R'], color['G'], color['B'], color['A'])
    shape_type = shape['Type']
    rotation = shape['Rotation']
    
    # 创建主形状图层
    main_layer = create_shape_layer(shape_type, bbox, rgba, rotation)
    
    # 创建蒙版图层
    if 'mask' in shape and shape['mask']:
        # 绘制所有蒙版形状
        for mask_shape in shape['mask']:
            mask_bbox = get_bbox_from_bounds(mask_shape['NormalizedBounds'], temp_size)
            main_layer = apply_mask_alpha(main_layer, mask_shape, mask_bbox)
    
    # 将图层合并到主画布
    image = Image.alpha_composite(image.convert('RGBA'), main_layer)

# 将图像缩放到最终尺寸
image = image.resize((final_size, final_size), Image.LANCZOS)

# 保存图片
image.save('demo.png', format='PNG')  # 直接保存为PNG格式，保持透明通道