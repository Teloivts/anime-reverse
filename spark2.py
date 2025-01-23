import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import label
from scipy.signal import convolve2d

import numpy as np
from skimage.graph import route_through_array

def shortest_path(start, end, shape, obstacles=None):
    """
    使用 skimage 查找绕过障碍物的最短路径。
    :param start: 起点坐标 (y, x)
    :param end: 终点坐标 (y, x)
    :param shape: 图像形状 (H, W)
    :param obstacles: 障碍物二值掩膜（可选，1表示障碍）
    :return: 路径点列表 [[y0, x0], [y1, x1], ...]
    """
    # 若未提供障碍物，假设全图可通行
    if obstacles is None:
        cost = np.ones(shape, dtype=np.float32)
    else:
        cost = np.where(obstacles, np.inf, 1.0)  # 障碍物位置代价为无穷大

    # 查找路径（使用欧氏距离作为启发式）
    path, _ = route_through_array(
        cost, 
        start=(int(start[0]), int(start[1])), 
        end=(int(end[0]), int(end[1])), 
        fully_connected=True,  # 8邻域连通
        geometric=True         # 使用A*算法
    )
    return np.array(path)

def debug(img):
    plt.imsave('output/guo_hall_thinning_image.png', img, cmap='gray')  # 保存细化图像
    

def constrained_guo_hall(binary_image, max_iter=100):
    skeleton = binary_image.copy()
    original_labels, num_features = label(binary_image, structure=np.ones((3,3)))  # 原始连通域标记
    
    for _ in range(max_iter):
        changed = False
        for step in [1, 2]:  # Guo-Hall的两个子迭代
            markers = np.zeros_like(skeleton)
            for y in range(1, skeleton.shape[0]-1):
                for x in range(1, skeleton.shape[1]-1):
                    if skeleton[y, x] == 0:
                        continue
                    # Guo-Hall删除条件判断
                    p2, p3, p4 = skeleton[y-1, x], skeleton[y-1, x+1], skeleton[y, x+1]
                    p5, p6, p7 = skeleton[y+1, x+1], skeleton[y+1, x], skeleton[y+1, x-1]
                    p8, p9, p2_ = skeleton[y, x-1], skeleton[y-1, x-1], skeleton[y-1, x]
                    C = (int(p2 == 0 and p3 == 1) + int(p3 == 0 and p4 == 1) +
                         int(p4 == 0 and p5 == 1) + int(p5 == 0 and p6 == 1) +
                         int(p6 == 0 and p7 == 1) + int(p7 == 0 and p8 == 1) +
                         int(p8 == 0 and p9 == 1) + int(p9 == 0 and p2_ == 1))
                    N = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                    m = min(p2, p4, p6) if step == 1 else min(p2, p4, p8)
                    
                    if C == 1 and 2 <= N <= 6 and m == 0:
                        # 检查是否为简单点（不破坏连通性）
                        temp = skeleton.copy()
                        temp[y, x] = 0
                        modified_labels, _ = label(temp, structure=np.ones((3,3)))
                        # 如果删除后导致原连通域分裂，则保留
                        if np.unique(original_labels[temp == 1]).size == np.unique(original_labels[skeleton == 1]).size:
                            markers[y, x] = 1
            skeleton = np.where(markers, 0, skeleton)
            changed = np.any(markers)
        if not changed:
            break
    return skeleton

def find_endpoints(region):
    kernel = np.array([[1,1,1], [1,0,1], [1,1,1]], dtype=np.uint8)
    neighbors = convolve2d(region, kernel, mode='same')
    endpoints = np.argwhere((region) & (neighbors == 1))
    return endpoints

def repair_disconnections(skeleton, original_labels):
    repaired = skeleton.copy()
    skel_labels, skel_num = label(skeleton, structure=np.ones((3,3)))
    plt.imsave('output/skel_labels_img.png',get_label_img(skel_labels, skel_num))
    # 遍历每个骨架连通域，检查对应的原始标签是否唯一
    for skel_id in range(1, skel_num+1):
        original_ids = original_labels[skel_labels == skel_id]
        print(original_ids)
        unique_original = np.unique(original_ids[original_ids != 0])
        if len(unique_original) > 1:
            print('reapir')
            # 断裂处理：找到断裂端点并连接
            endpoints = find_endpoints(skel_labels == skel_id)
            if len(endpoints) >= 2:
                # 计算端点间最短路径并绘制线条
                path = shortest_path(endpoints[0], endpoints[1], skeleton.shape)
                repaired[path[:,0], path[:,1]] = 1
    return repaired

def get_label_img(img,num):
    # # 4. 创建一个RGB空白图像用于显示结果 (初始化为白色背景)
    output_image = np.ones((img.shape[0], img.shape[1], 3), dtype=float)  # 3个通道

    # 5. 给每个连通域分配不同的颜色（并使用透明度）
    colors = plt.cm.get_cmap('hsv', num + 1)  # 使用HSV色图分配颜色

    # 6. 生成结果图像
    for i in range(1, num + 1):
        # 获取当前连通域的颜色
        color = colors(i)[:3]  # 只取RGB部分，忽略Alpha通道
        output_image[img == i] = color  # 分配颜色
    return output_image

from scipy.ndimage import find_objects

def detect_fractures(original_labels, num_original, skeleton_labels):
    fractures = []

    for orig_id in range(1, num_original + 1):
        # 提取当前原始连通域的掩膜
        mask = (original_labels == orig_id)
        # 获取该区域内所有骨架标签
        skel_ids_in_orig = np.unique(skeleton_labels[mask])

        skel_ids_in_orig = skel_ids_in_orig[skel_ids_in_orig != 0]  # 过滤背景
        
        # 如果骨架标签数量>1，说明存在断裂
        if len(skel_ids_in_orig) > 1:
            fractures.append({
                "original_id": orig_id,
                "skeleton_ids": skel_ids_in_orig,
                "mask": mask
            })
    return fractures

def repair_fractures(skeleton,fractures):
    repaired = skeleton.copy()
    for fracture in fractures:
        orig_mask = fracture["mask"]  # 原始连通域区域
        skel_ids = fracture["skeleton_ids"]  # 初始骨架连通域列表
        skeleton_labels, num_skeleton = label(skeleton, structure=np.ones((3,3)))
        # 迭代合并，直到只剩一个连通域
        while len(skel_ids) > 1:
            # 提取所有骨架分支的端点
            endpoints = []
            for skel_id in skel_ids:
                branch = (skeleton_labels == skel_id)
                eps = find_endpoints(branch)
                endpoints.extend(eps)
            
            # 计算所有端点对的距离，选择最近的一对
            closest_pair = None
            min_dist = np.inf
            for i in range(len(endpoints)):
                for j in range(i+1, len(endpoints)):
                    y1, x1 = endpoints[i]
                    y2, x2 = endpoints[j]
                    dist = np.sqrt((y1-y2)**2 + (x1-x2)**2)
                    if dist < min_dist and skeleton_labels[y1, x1] != skeleton_labels[y2, x2]:
                        min_dist = dist
                        closest_pair = (endpoints[i], endpoints[j])
            
            if closest_pair is None:
                break  # 无法进一步合并
            
            # 连接端点对（路径限制在原始区域内）
            start, end = closest_pair
            print(f'connect {skeleton_labels[start[0]][start[1]]} and {skeleton_labels[end[0]][end[1]]}')
            path = route_through_array(
                np.where(orig_mask, 1, np.inf),  # 仅允许在原始区域内移动
                start=start,  # 转换为 (x,y) 格式
                end=end,
                fully_connected=True
            )
            print(path)
            for p in path[0]:  # 转换回 (y,x) 格式
                repaired[p[0]][p[1]]=1 

            plt.imshow(repaired)
            plt.show()
            # 重新标记骨架连通域
            skel_labels, _ = label(repaired, structure=np.ones((3,3)))
            print(f'check {skel_labels[start[0]][start[1]]} and {skel_labels[end[0]][end[1]]}')
            skel_ids = np.unique(skel_labels[orig_mask])
            skel_ids = skel_ids[skel_ids != 0]  # 更新当前骨架连通域列表
            skeleton_labels = skel_labels
    
    return repaired

# 1. 加载图像
image_path = '1_lineart.png'  # 图像路径
image = Image.open(image_path).convert('L')  # 转换为灰度图像

# 2. 二值化处理
binary_image = np.array(image) < 220  # 黑色为True，白色为False
plt.imsave('output/binary_image.png', binary_image, cmap='gray')  # 保存二值化图像

# Step 1: 约束骨架化
# skeleton = constrained_guo_hall(binary_image)
# plt.imsave('output/guo_hall_thinning_image.png', skeleton, cmap='gray')  # 保存细化图像
skeleton = np.array(Image.open('output/guo_hall_thinning_image.png').convert('L'))

# Step 2: 修复断裂
original_labels, num_origin = label(binary_image, structure=np.ones((3,3)))
plt.imsave('output/original_labels_img.png',get_label_img(original_labels,num_origin)) 
skeleton_labels, num_skeleton = label(skeleton, structure=np.ones((3,3)))
plt.imsave('output/skeleton_labels_img.png',get_label_img(skeleton_labels,num_skeleton)) 

fractures = detect_fractures(original_labels, num_origin, skeleton_labels)
# for fracture in fractures:
#     print(fracture['original_id'], fracture['skeleton_ids'])

repaired_skeleton = repair_fractures(skeleton, fractures)

plt.imsave('output/repaired_skeleton.png',repaired_skeleton, cmap='gray') 


# print(fractures)

# repaired_skeleton = repair_disconnections(skeleton, original_labels)

# plt.imsave('output/guo_hall_thinning_image_repaired.png', repaired_skeleton, cmap='gray')  # 保存细化图像


# print("Image shape:", np.array(image).shape)          # 原图像维度
# print("Binary image shape:", binary_image.shape)      # 二值化后维度

# structure_3x3 = np.ones((3, 3))  # 全1的5×5矩阵

# # 3. 连通域分析（使用3×3邻域）
# labeled_image, num_features = label(binary_image, structure=structure_3x3)

# # 4. 创建一个RGB空白图像用于显示结果 (初始化为白色背景)
# output_image = np.ones((labeled_image.shape[0], labeled_image.shape[1], 3), dtype=float)  # 3个通道

# # 5. 给每个连通域分配不同的颜色（并使用透明度）
# colors = plt.cm.get_cmap('hsv', num_features + 1)  # 使用HSV色图分配颜色

# # 6. 生成结果图像
# for i in range(1, num_features + 1):
#     # 获取当前连通域的颜色
#     color = colors(i)[:3]  # 只取RGB部分，忽略Alpha通道
#     output_image[labeled_image == i] = color  # 分配颜色

# # 7. 将原始灰度图像作为透明背景叠加
# alpha = 1.0  # 设置透明度（0表示完全透明，1表示完全不透明）
# original_gray = np.expand_dims(np.array(image) / 255.0, axis=-1)  # 归一化到[0, 1]并扩展为3个通道

# # 将灰度图像的亮度与着色图像进行半透明叠加
# output_image = alpha * output_image + (1 - alpha) * original_gray

# plt.imsave('output.png', output_image)  # 保存结果图像
# # 8. 显示结果
# plt.imshow(output_image)
# plt.axis('off')  # 关闭坐标轴
# plt.show()

# 生成对比图
# label_images = [cv2.imread(f'output/label_{label}.png') for label in unique_labels]
# concat_labels = cv2.hconcat(label_images)
# final_image = cv2.vconcat([concat_labels, result])
# cv2.imwrite('output/comparison.png', final_image)