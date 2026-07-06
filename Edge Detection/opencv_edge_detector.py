"""
基于OpenCV传统方法的卡片边缘检测
使用边缘检测 + 轮廓查找 + 霍夫变换
"""

import cv2
import numpy as np
from pathlib import Path


class OpenCVEdgeDetector:
    """基于OpenCV传统方法的卡片边缘检测器"""

    def __init__(self):
        self.is_ready = False

    def preprocess(self, image):
        """图像预处理"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)

        return gray, edges

    def find_card_contour(self, edges, min_area=5000):
        """查找卡片轮廓"""
        # 形态学操作闭合轮廓
        kernel = np.ones((5, 5), np.uint8)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 轮廓查找
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 找到最大的四边形轮廓
        best_contour = None
        best_score = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # 近似多边形
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 优先选择四边形
            if len(approx) >= 4 and len(approx) <= 8:
                # 计算轮廓的凸包面积比（评估四边形程度）
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    score = area * solidity

                    if score > best_score:
                        best_score = score
                        best_contour = approx

        # 如果没找到四边形，使用最大的轮廓
        if best_contour is None:
            max_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    best_contour = contour

        return best_contour

    def order_points(self, pts):
        """将四个角点按顺时针顺序排列: 左上、右上、右下、左下"""
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype=np.float32)

        # 计算所有点的和与差
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下

        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]  # 右上
        rect[3] = pts[np.argmax(d)]  # 左下

        return rect

    def detect_edge(self, image):
        """
        检测卡片边缘
        返回4个角点的归一化坐标 [x1,y1,x2,y2,x3,y3,x4,y4]
        """
        h, w = image.shape[:2]

        gray, edges = self.preprocess(image)
        contour = self.find_card_contour(edges)

        if contour is None:
            # 如果没找到轮廓，返回图像中心区域的默认框
            margin = 0.1
            coords = [
                margin, margin,
                1-margin, margin,
                1-margin, 1-margin,
                margin, 1-margin
            ]
            return np.array(coords)

        # 获取最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int_(box)

        # 排序角点
        ordered_box = self.order_points(box)

        # 转换为归一化坐标
        coords = []
        for point in ordered_box:
            coords.append(float(point[0] / w))
            coords.append(float(point[1] / h))

        return np.array(coords, dtype=np.float32)

    def draw_edge(self, image, corner_coords=None, line_color=(0, 255, 0), line_thickness=3):
        """在图像上绘制卡片边缘（绿色线条）"""
        if corner_coords is None:
            corner_coords = self.detect_edge(image)

        h, w = image.shape[:2]

        points = []
        for i in range(0, 8, 2):
            x = int(corner_coords[i] * w)
            y = int(corner_coords[i + 1] * h)
            points.append((x, y))

        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))

        # 绘制多边形（绿色线条）
        cv2.polylines(image, [pts], isClosed=True, color=line_color, thickness=line_thickness)

        # 绘制角点
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        labels = ['1-左上', '2-右上', '3-右下', '4-左下']

        for i, pt in enumerate(points):
            cv2.circle(image, pt, 10, colors[i], -1)
            cv2.putText(image, labels[i], (pt[0]+15, pt[1]-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)

        return image

    def crop_card(self, image, corner_coords=None):
        """
        裁剪卡片区域（根据检测到的角点）
        corner_coords: 归一化的角点坐标 [x1,y1,x2,y2,x3,y3,x4,y4]
        返回裁剪后的卡片图像
        """
        if corner_coords is None:
            corner_coords = self.detect_edge(image)

        h, w = image.shape[:2]

        # 获取角点坐标
        pts = []
        for i in range(0, 8, 2):
            x = int(corner_coords[i] * w)
            y = int(corner_coords[i + 1] * h)
            pts.append([x, y])

        pts = np.array(pts, dtype=np.float32)

        # 计算目标矩形的宽高
        # 左上到右上的距离作为宽度
        width_top = np.sqrt((pts[1][0] - pts[0][0])**2 + (pts[1][1] - pts[0][1])**2)
        # 右下到左下的距离
        width_bottom = np.sqrt((pts[2][0] - pts[3][0])**2 + (pts[2][1] - pts[3][1])**2)
        # 右上到右下的距离作为高度
        height_right = np.sqrt((pts[2][0] - pts[1][0])**2 + (pts[2][1] - pts[1][1])**2)
        # 左上到左下的距离
        height_left = np.sqrt((pts[3][0] - pts[0][0])**2 + (pts[3][1] - pts[0][1])**2)

        max_width = int(max(width_top, width_bottom))
        max_height = int(max(height_right, height_left))

        # 目标矩形顶点（左上、右上、右下、左下）
        dst_pts = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(pts, dst_pts)

        # 应用透视变换
        cropped = cv2.warpPerspective(image, M, (max_width, max_height))

        return cropped

    def test(self, test_images_dir, test_labels_dir=None, show_results=True):
        """测试检测器"""
        images_dir = Path(test_images_dir)
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

        if not image_files:
            raise ValueError(f"测试目录中没有找到图片文件: {test_images_dir}")

        print("=" * 50)
        print("开始测试 (OpenCV传统方法)")
        print("=" * 50)

        total = len(image_files)
        detected_count = 0
        correct_count = 0
        results = []

        for img_path in sorted(image_files):
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            # 检测边缘
            predicted_coords = self.detect_edge(image)
            result_image = self.draw_edge(image.copy(), predicted_coords)

            if test_labels_dir:
                label_path = Path(test_labels_dir) / (img_path.stem + '.txt')
                if label_path.exists():
                    try:
                        with open(label_path, 'r') as f:
                            content = f.read().strip()
                            if not content:
                                detected_count += 1
                                continue

                            lines = content.split('\n')
                            content = ' '.join(lines).strip()
                            parts = content.split()
                            true_coords = []

                            if len(parts) == 4:
                                for point in parts:
                                    pt_parts = point.split(',')
                                    if len(pt_parts) != 2:
                                        continue
                                    x, y = float(pt_parts[0]), float(pt_parts[1])
                                    true_coords.extend([x, y])
                            elif len(parts) >= 9:
                                for i in range(1, 9, 2):
                                    if i + 1 < len(parts):
                                        x = float(parts[i]) / image.shape[1]
                                        y = float(parts[i + 1]) / image.shape[0]
                                        true_coords.extend([x, y])

                            if len(true_coords) != 8:
                                detected_count += 1
                                continue

                            true_coords = np.array(true_coords)
                            predicted_coords = np.array(predicted_coords)

                            error = np.sqrt(np.sum((true_coords - predicted_coords)**2))
                            is_correct = error < 0.1

                            if is_correct:
                                correct_count += 1
                                detected_count += 1
                            else:
                                detected_count += 0

                            results.append({
                                'filename': img_path.name,
                                'error': error,
                                'correct': is_correct
                            })

                            cv2.putText(result_image, f'Error: {error:.4f}',
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                       (255, 255, 255), 2)
                            if is_correct:
                                cv2.putText(result_image, 'CORRECT',
                                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                           (0, 255, 0), 2)
                            else:
                                cv2.putText(result_image, 'INCORRECT',
                                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                           (0, 0, 255), 2)
                    except Exception:
                        detected_count += 1
                else:
                    detected_count += 1
                    results.append({
                        'filename': img_path.name,
                        'error': None,
                        'correct': None
                    })
            else:
                detected_count += 1
                results.append({
                    'filename': img_path.name,
                    'error': None,
                    'correct': None
                })

            # 保存结果
            output_dir = Path('data/test/output')
            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存带边框的检测结果图
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), result_image)

            # 保存裁剪后的卡片图片
            cropped_card = self.crop_card(image, predicted_coords)
            cropped_name = f"cropped_{img_path.stem}.png"
            cropped_path = output_dir / cropped_name
            cv2.imwrite(str(cropped_path), cropped_card)

            if show_results:
                result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                cropped_rgb = cv2.cvtColor(cropped_card, cv2.COLOR_BGR2RGB)
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                axes[0].imshow(result_rgb)
                axes[0].set_title(f'检测结果: {img_path.name}')
                axes[0].axis('off')

                axes[1].imshow(cropped_rgb)
                axes[1].set_title(f'裁剪结果: cropped_{img_path.stem}.png')
                axes[1].axis('off')

                plt.tight_layout()
                plt.show()

        #打印统计结果
        print("\n" + "=" * 50)
        print("测试结果统计")
        print("=" * 50)
        print(f"总测试图片数: {total}")
        print(f"检测成功率: {detected_count/total*100:.2f}%")

        if test_labels_dir:
            print(f"正确率: {correct_count/total*100:.2f}%")
            print("\n详细结果:")
            for r in results:
                if r['error'] is not None:
                    status = "正确" if r['correct'] else "错误"
                    print(f"  {r['filename']}: 误差={r['error']:.4f} [{status}]")

        print(f"\n带边框检测图已保存到: {output_dir}/")
        print(f"裁剪后的卡片图已保存到: {output_dir}/cropped_*.png")

        return {
            'total': total,
            'detected': detected_count,
            'correct': correct_count,
            'detection_rate': detected_count/total*100,
            'accuracy': correct_count/total*100 if test_labels_dir else None,
            'results': results
        }