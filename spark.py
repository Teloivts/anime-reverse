from line_render import LineRenderer
import math
from PIL import Image, ImageDraw  # 导入PIL库

def generate_sinewave_geometry(canvas_size=512, segment_count=200):
    max_radius = canvas_size / 24  # 最大半径设为画布宽度的1/6
    gr = (1 + math.sqrt(5)) / 2   # 黄金比例
    pi = math.pi

    positions = []
    radii = []

    for i in range(segment_count + 1):
        # 生成原始数学坐标
        a = i / segment_count
        x_math = -pi + (2 * pi * a)
        y_math = math.sin(x_math) / gr
        
        # 转换为画布坐标（原点在中心）
        x = (x_math + pi) / (2 * pi) * canvas_size  # X轴映射到 [0, canvas_size]
        y = (y_math * gr + 1) / 3 * canvas_size     # Y轴映射到 [0, canvas_size]
        
        # 计算半径（带动态变化）
        r = math.cos(x_math / 2.0) * max_radius
        positions.append((x,y,r))
    diff = canvas_size/2 - positions[0][1]
    positions = [(p[0],p[1]+diff,p[2])for p in positions]
    return positions

# 创建一个空白透明图像
def create_blank_image(size=(512, 512)):
    return Image.new("RGBA", size, (0, 0, 0, 0))  # 创建透明图像

# 初始化渲染器
renderer = LineRenderer()

from line_render import LineRenderer
from data import *
import math
def generate_sinewave_geometry(canvas_size=512, segment_count=200):
    max_radius = canvas_size / 24  # 最大半径设为画布宽度的1/6
    gr = (1 + math.sqrt(5)) / 2   # 黄金比例
    pi = math.pi

    positions = []
    radii = []

    for i in range(segment_count + 1):
        # 生成原始数学坐标
        a = i / segment_count
        x_math = -pi + (2 * pi * a)
        y_math = math.sin(x_math) / gr
        
        # 转换为画布坐标（原点在中心）
        x = (x_math + pi) / (2 * pi) * canvas_size  # X轴映射到 [0, canvas_size]
        y = (y_math * gr + 1) / 3 * canvas_size     # Y轴映射到 [0, canvas_size]
        
        # 计算半径（带动态变化）
        r = math.cos(x_math / 2.0) * max_radius
        positions.append((x,y,r))
    diff = canvas_size/2 - positions[0][1]
    positions = [(p[0],p[1]+diff,p[2])for p in positions]
    return positions


# 初始化渲染器
renderer = LineRenderer()

curves = []
points = []
raw_data = raw_bear_data
for i in range(0,len(raw_data),3):
    if raw_data[i] != 'NaN':
        points.append([raw_data[i+0]*128+256,raw_data[i+1]*128+256,raw_data[i+2]*2])
    else:
        curves.append(points)
        # print([point[:2] for point in points])
        points=[]
if points:
    curves.append(points)


# 创建透明图像
blank_image = create_blank_image()
frames = []

# # 渲染图像
for curve in curves:
    image = renderer.render(
        canvas_size=(512, 512),
        points=curve
    )
    blank_image.alpha_composite(image)
        # 创建带白色背景的帧并添加到列表
    frame = Image.new("RGB", blank_image.size, (255, 255, 255))
    frame.paste(blank_image, mask=blank_image.split()[3])  # 保持透明度
    frames.append(frame)

# 保存为GIF（带白色背景）
if frames:
    frames[0].save(
        "animation.gif",
        save_all=True,
        append_images=frames[1:],
        duration=1000,    # 每帧持续时间（毫秒）
        loop=0,          # 无限循环
        disposal=2       # 恢复背景设置
    )


# fit_points = [[169.03680419921875, 203.40879440307617], [170.3213312128981, 200.76390794068635], [173.15191433402882, 195.65284377893252], [176.33033480470226, 190.77763465567034], [179.86212524631458, 186.13161496494294], [183.7504672072501, 181.7182565010204], [188.00464027267276, 177.53930437992096], [192.59856701574896, 173.63053968543332], [197.5489981635812, 169.98798366091324], [202.82785398223876, 166.64350910388782], [208.42817260879727, 163.61059925903854], [214.31791197435894, 160.91457479142872], [220.4738822636515, 158.57200329561115], [226.86611676477384, 156.59969408496144], [233.4576725863145, 155.01334015158375], [240.21195767229153, 153.82512561718286], [247.10180887282178, 153.04390962389076], [254.07972549165538, 152.6793471060716], [261.11315909678063, 152.73752600361126], [268.15824230352587, 153.2232734147442], [275.19732931668057, 154.1430174046299], [282.1786477526936, 155.4996657809073], [289.0741364743237, 157.2988160337439], [295.8741815754906, 159.5551272108153], [302.53292962839635, 162.27586897804093], [309.02352388805696, 165.47727290941708], [315.3238662301728, 169.18521820933722], [321.4256189555535, 173.4481094540712], [327.25821905831555, 178.28475371965425], [332.8009657727109, 183.76731099030536], [337.99211082910114, 189.96095590518928], [342.75804445227527, 196.95137793543296], [346.9951934814453, 204.82959365844727]]
# for i in range(len(fit_points)):
#     fit_points[i] = [fit_points[i][0],fit_points[i][1],curves[0][i][2]]
# image = renderer.render(
#     canvas_size=(512, 512),
#     points=fit_points,
#     color=(0,1,0,0.4)
# )
# blank_image.alpha_composite(image)
    
# curve = [[256, 256, 30],
#  [256+35, 256, 5]]
# image = renderer.render(
#     canvas_size=(512, 512),
#     points=curve
# )
# blank_image.alpha_composite(image)
# 保存或处理图像
blank_image.save('output.png')
