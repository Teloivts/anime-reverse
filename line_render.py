# linerender.py
import moderngl
import numpy as np
from PIL import Image

class LineRenderer:
    def __init__(self):
        pass
        # self.ctx = moderngl.create_standalone_context()
        # self._init_shaders()
        # self._setup_rendering()

    def _init_shaders(self):
        # 顶点着色器
        self.vertex_shader = """
precision mediump float;
precision mediump int;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

// Radius values are given by geometry data
in float radius0;
in float radius1;
in vec2 position0;
in vec2 position1;

out vec2 p;
flat out vec2 p0;
flat out vec2 p1;
// Output radius
flat out float r0;
flat out float r1;

void main(){
    p0 = position0;
    p1 = position1;
    r0 = radius0;
    r1 = radius1;

    float cosTheta = (r0 - r1)/distance(p0, p1);
    // Coner case: One circle is entirely inside the another, discard the edge.
    if(abs(cosTheta) >= 1.0) return;

    vec2 tangent = normalize(p1 - p0);
    vec2 normal = vec2(-tangent.y, tangent.x);

    vec2 offsetSign = vec2[](
        vec2(-1.0,-1.0),
        vec2(-1.0, 1.0),
        vec2( 1.0, 1.0),
        vec2( 1.0,-1.0)
    )[gl_VertexID];
    vec2 position = vec2[](position0, position0, position1, position1)[gl_VertexID];
    float radius = vec4(radius0, radius0, radius1, radius1)[gl_VertexID];

    // Apply the half angle formula from cos(theta) to tan(theta/2)
    float tanHalfTheta = sqrt((1.0+cosTheta) / (1.0-cosTheta));
    float cotHalfTheta = 1.0 / tanHalfTheta;
    float normalTanValue = vec4(tanHalfTheta, tanHalfTheta, cotHalfTheta, cotHalfTheta)[gl_VertexID];
    // Corner case: The small circle is very close to the big one, casuing large offset in the normal direction, discard the edge
    if(normalTanValue > 10.0 || normalTanValue < 0.1) return;

    vec2 trapzoidVertexPosition = position +
        offsetSign.x * radius * tangent +
        offsetSign.y * radius * normal * normalTanValue;
    p = trapzoidVertexPosition;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 0.0, 1.0);
}
            """

        # 片段着色器
        self.fragment_shader = """
            precision mediump float;
            precision mediump int;

            //const vec4 color = vec4(0.0, 0.0, 0.0, 0.8);
            uniform vec4 color;  // 改为uniform变量以便控制
            
            out vec4 outColor;

            in vec2 p;
            flat in vec2 p0;
            flat in vec2 p1;
            flat in float r0;
            flat in float r1;

            void main() {
                vec2 tangent = normalize(p1 - p0);
                vec2 normal = vec2(-tangent.y, tangent.x);
                float len = distance(p1, p0);

                vec2 pLocal = vec2(dot(p-p0, tangent), dot(p-p0, normal));

                float d0 = distance(p, p0);
                float d1 = distance(p, p1);
                float d0cos = pLocal.x / d0;
                float d1cos = (pLocal.x - len) / d1;

                float cosTheta = (r0 - r1)/distance(p0, p1);

                if(d0cos < cosTheta && d0 > r0) discard;
                if(d1cos > cosTheta && d1 > r1) discard;

                float A = color.a;
                //if (d0 < r0 && d1 < r1) discard;
                if (d0 < r0 || d1 < r1) A = 1.0 - sqrt(1.0 - A);

                outColor = vec4(color.rgb, A);
            }
            """

    def _setup_rendering(self):
        # 创建着色器程序
        self.prog = self.ctx.program(
            vertex_shader=self.vertex_shader,
            fragment_shader=self.fragment_shader
        )

    def render(self, canvas_size, points, color=(1.0, 0, 0, 0.4)):
        """
        渲染入口函数
        :param canvas_size: 画布尺寸 (width, height)
        :param points: 点序列，每个元素为 (x, y, radius)
        :return: PIL.Image 对象
        """
        self.ctx = moderngl.create_standalone_context()
        self._init_shaders()
        self._setup_rendering()
        # 验证输入
        if len(points) < 1:
            raise ValueError("至少需要1个点来构成线段")
        if any(len(p) != 3 for p in points):
            raise ValueError("每个点必须包含(x, y, radius)三个值")
        if len(points) == 1:
            point2 = points[0]
            point2[0] += 0.0000001
            points.append(point2)
        # 设置颜色（包括透明度）
        self.prog['color'].value = color
        points += reversed(points)
        # 生成线段数据
        segments = self._generate_segments(points)
        # 创建顶点缓冲
        try:
            vbo = self.ctx.buffer(np.array(segments, dtype='f4'))
        except Exception as e:
            raise RuntimeError(f"创建缓冲失败: {e}，数据长度: {len(segments)}")
        
        # 配置顶点数组
        vao = self.ctx.vertex_array(
            self.prog,
            [(vbo, '2f 2f f f/i', 'position0', 'position1', 'radius0', 'radius1')]
        )

        # 设置投影矩阵
        width, height = canvas_size
        projection = self._create_projection_matrix(width, height)
        self.prog['projectionMatrix'].write(projection.tobytes())
        self.prog['modelViewMatrix'].write(np.eye(4, dtype='f4').tobytes())

        # 创建帧缓冲
        color_tex = self.ctx.texture((width, height), 4)
        fbo = self.ctx.framebuffer(color_attachments=[color_tex])

        # 执行渲染（修正上下文管理方式）
        fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA,
            moderngl.ONE,
            moderngl.ONE_MINUS_SRC_ALPHA
        )
        self.ctx.clear(1.0, 1.0, 1.0, 0.0)
        vao.render(mode=moderngl.TRIANGLE_STRIP, vertices=4, instances=len(points)-1)

        # 读取并处理图像
        image = Image.frombytes('RGBA', (width, height), color_tex.read(), 'raw', 'RGBA', 0, -1)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

        # 清理资源
        vbo.release()
        vao.release()
        color_tex.release()
        fbo.release()

        return image

    def _generate_segments(self, points):
        """将点序列转换为线段实例数据"""
        segments = []
        for i in range(len(points) - 1):
            x0, y0, r0 = points[i]
            x1, y1, r1 = points[i+1]
            segments.extend([x0, y0, x1, y1, r0, r1])
        return segments

    def _create_projection_matrix(self, width, height):
        """创建正交投影矩阵"""
        return np.array([
            [2.0/width, 0.0, 0.0, -1.0],
            [0.0, 2.0/height, 0.0, -1.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype='f4').T

    def __del__(self):
        """析构函数释放OpenGL资源"""
        if hasattr(self, 'ctx'):
            self.ctx.release()