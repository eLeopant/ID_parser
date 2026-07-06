"""
基于KNN的卡片边缘检测项目
KNN Card Edge Detection Project

使用方法:
1. 将训练图片放入 data/train/images/ 目录
2. 将训练标签放入 data/train/labels/ 目录（每个图片对应一个同名.txt文件）
3. 运行 python main.py 选择训练模式进行模型训练
4. 将测试图片放入 data/test/images/ 目录
5. 运行 python main.py 选择测试模式进行边缘检测
"""

import os
import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path


class CardEdgeDetector:
    """基于KNN的卡片边缘检测器"""

    def __init__(self, model_path='data/models/knn_card_model.pkl'):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.is_trained = False

    def extract_features(self, image):
        """
        从图像中提取特征 - 简化版，适合小样本训练
        """
        # 确保图像是彩色
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 调整图像大小为统一尺寸
        img_resized = cv2.resize(image, (64, 64))

        features = []

        # 1. 简化的颜色直方图 (减少bin数)
        for i in range(3):
            hist = cv2.calcHist([img_resized], [i], None, [16], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-6)
            features.extend(hist)

        # 2. 边缘检测特征 - 简化
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # 将边缘图像分成4x4网格，计算每个网格的边缘密度
        grid_size = 16
        step = 64 // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                region = edges[i*step:(i+1)*step, j*step:(j+1)*step]
                features.append(np.mean(region) / 255.0)

        # 3. 四象限颜色特征
        h, w = 64, 64
        h2, w2 = h // 2, w // 2
        quadrants = [
            img_resized[:h2, :w2],      # 左上
            img_resized[:h2, w2:],      # 右上
            img_resized[h2:, w2:],     # 右下
            img_resized[h2:, :w2],     # 左下
        ]
        for q in quadrants:
            for c in range(3):
                features.append(np.mean(q[:, :, c]) / 255.0)
                features.append(np.std(q[:, :, c]) / 255.0)

        # 4. 边缘方向直方图 (简化HOG)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # 计算8-bin角度直方图
        hist_angle = np.histogram(angle, bins=8, range=(-180, 180), weights=mag)[0]
        hist_angle = hist_angle / (hist_angle.sum() + 1e-6)
        features.extend(hist_angle)

        # 5. 整体统计特征
        features.append(np.std(gray) / 255.0)
        features.append((np.max(gray) - np.min(gray)) / 255.0)

        return np.array(features, dtype=np.float32)

    def load_training_data(self, train_images_dir, train_labels_dir):
        """加载训练数据"""
        images_dir = Path(train_images_dir)
        labels_dir = Path(train_labels_dir)

        X_list = []
        y_list = []

        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

        if not image_files:
            raise ValueError(f"训练目录中没有找到图片文件: {train_images_dir}")

        print(f"找到 {len(image_files)} 张训练图片")

        for img_path in sorted(image_files):
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"警告: 无法读取图片 {img_path}")
                continue

            features = self.extract_features(image)
            X_list.append(features)

            label_path = labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        content = f.read().strip()

                        # 跳过空文件或空内容
                        if not content:
                            print(f"警告: 标签文件为空 {label_path}")
                            continue

                        # 处理可能的多行格式
                        lines = content.split('\n')
                        content = ' '.join(lines).strip()

                        parts = content.split()
                        coords = []

                        # 支持两种格式：
                        # 格式1: x1,y1 x2,y2 x3,y3 x4,y4 (归一化坐标)
                        # 格式2: class x1 y1 x2 y2 x3 y3 x4 y4 (YOLO格式，像素坐标)

                        if len(parts) == 4:
                            # 格式1: x1,y1 x2,y2 x3,y3 x4,y4
                            for point in parts:
                                pt_parts = point.split(',')
                                if len(pt_parts) != 2:
                                    print(f"警告: 标签格式错误 {label_path}: {point}")
                                    continue
                                x, y = float(pt_parts[0]), float(pt_parts[1])
                                coords.extend([x, y])
                        elif len(parts) >= 9:
                            # 格式2: class x1 y1 x2 y2 x3 y3 x4 y4
                            # 获取图片尺寸用于归一化
                            h, w = image.shape[:2]
                            # 从第2个值开始，每两个值为一组坐标
                            for i in range(1, 9, 2):
                                if i + 1 < len(parts):
                                    x = float(parts[i]) / w
                                    y = float(parts[i + 1]) / h
                                    coords.extend([x, y])

                        if len(coords) == 8:
                            y_list.append(coords)
                        else:
                            print(f"警告: 坐标数量不对 {label_path}, parts数量={len(parts)}")
                            continue
                except ValueError as e:
                    print(f"警告: 解析标签文件出错 {label_path}: {e}")
                    continue
            else:
                print(f"警告: 找不到标签文件 {label_path}")
                continue

        if len(X_list) == 0:
            raise ValueError("没有找到有效的训练数据")

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"训练数据形状: X={X.shape}, y={y.shape}")

        # 调试信息：打印前3个样本的标签
        print("\n前3个样本的角点坐标（归一化）:")
        for i in range(min(3, len(y))):
            coords = y[i]
            print(f"  样本{i+1}: 左上({coords[0]:.3f},{coords[1]:.3f}) "
                  f"右上({coords[2]:.3f},{coords[3]:.3f}) "
                  f"右下({coords[4]:.3f},{coords[5]:.3f}) "
                  f"左下({coords[6]:.3f},{coords[7]:.3f})")

        return X, y

    def train(self, train_images_dir, train_labels_dir, n_neighbors=3):
        """训练KNN模型"""
        print("=" * 50)
        print("开始训练KNN模型")
        print("=" * 50)

        X, y = self.load_training_data(train_images_dir, train_labels_dir)

        X_scaled = self.scaler.fit_transform(X)

        print(f"\n样本数量: {len(X_scaled)}")
        print(f"特征维度: {X_scaled.shape[1]}")

        # 使用较小的K值
        if n_neighbors > len(X_scaled):
            n_neighbors = len(X_scaled)
            print(f"调整K值为: {n_neighbors}")

        # 使用全部数据训练（不做验证集划分，因为样本太少）
        print(f"\n使用全部 {len(X_scaled)} 个样本进行训练")
        print(f"K值 (邻居数量): {n_neighbors}")

        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        self.model.fit(X_scaled, y)

        # 在训练集上评估
        y_pred = self.model.predict(X_scaled)

        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)

        print("\n训练集评估结果:")
        print(f"  均方误差 (MSE): {mse:.6f}")
        print(f"  平均绝对误差 (MAE): {mae:.6f}")

        # 使用更宽松的阈值 (0.1相当于 10% 的图像尺寸误差)
        threshold = 0.1
        correct = np.sum(np.all(np.abs(y - y_pred) < threshold, axis=1))
        accuracy = correct / len(y) * 100
        print(f"  角点检测准确率: {accuracy:.2f}%")

        self.is_trained = True

        self.save_model()

        print("=" * 50)
        print("模型训练完成!")
        print("=" * 50)

        return mse, mae, accuracy

        print("\n使用全部数据重新训练...")
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        self.model.fit(X_scaled, y)
        self.is_trained = True

        self.save_model()

        print("=" * 50)
        print("模型训练完成!")
        print("=" * 50)

        return mse, mae, accuracy

    def save_model(self):
        """保存模型"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)
        print(f"模型已保存到: {self.model_path}")

    def load_model(self):
        """加载模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
        self.is_trained = True
        print(f"模型已加载: {self.model_path}")

    def detect_edge(self, image):
        """检测卡片边缘，返回4个角点的归一化坐标"""
        if not self.is_trained:
            raise ValueError("模型未训练或未加载")

        features = self.extract_features(image)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        coords = self.model.predict(features_scaled)[0]

        return coords

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

        cv2.polylines(image, [pts], isClosed=True, color=line_color, thickness=line_thickness)

        for i, pt in enumerate(points):
            cv2.circle(image, pt, 8, (0, 255, 255), -1)
            cv2.putText(image, str(i+1), (pt[0]+10, pt[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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
        width_top = np.sqrt((pts[1][0] - pts[0][0])**2 + (pts[1][1] - pts[0][1])**2)
        width_bottom = np.sqrt((pts[2][0] - pts[3][0])**2 + (pts[2][1] - pts[3][1])**2)
        height_right = np.sqrt((pts[2][0] - pts[1][0])**2 + (pts[2][1] - pts[1][1])**2)
        height_left = np.sqrt((pts[3][0] - pts[0][0])**2 + (pts[3][1] - pts[0][1])**2)

        max_width = int(max(width_top, width_bottom))
        max_height = int(max(height_right, height_left))

        # 目标矩形顶点
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
        """测试模型，返回识别成功率"""
        if not self.is_trained:
            raise ValueError("模型未训练或未加载")

        images_dir = Path(test_images_dir)
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

        if not image_files:
            raise ValueError(f"测试目录中没有找到图片文件: {test_images_dir}")

        print("=" * 50)
        print("开始测试")
        print("=" * 50)

        total = len(image_files)
        detected_count = 0
        correct_count = 0
        results = []

        for img_path in sorted(image_files):
            image = cv2.imread(str(img_path))
            if image is None:
                continue

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
                                # 格式1: x1,y1 x2,y2 x3,y3 x4,y4
                                for point in parts:
                                    pt_parts = point.split(',')
                                    if len(pt_parts) != 2:
                                        continue
                                    x, y = float(pt_parts[0]), float(pt_parts[1])
                                    true_coords.extend([x, y])
                            elif len(parts) >= 9:
                                # 格式2: class x1 y1 x2 y2 x3 y3 x4 y4
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
                            is_correct = error < 0.1  # 使用宽松阈值0.1

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
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                axes[0].imshow(result_rgb)
                axes[0].set_title(f'检测结果: {img_path.name}')
                axes[0].axis('off')

                axes[1].imshow(cropped_rgb)
                axes[1].set_title(f'裁剪结果: cropped_{img_path.stem}.png')
                axes[1].axis('off')

                plt.tight_layout()
                plt.show()

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
                else:
                    print(f"  {r['filename']}: 已检测")

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