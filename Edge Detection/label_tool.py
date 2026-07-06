"""
标签标注工具 - 用于标注卡片的四个角点

使用方法:
1. 将图片放入 data/train/images/ 目录
2. 运行 python label_tool.py
3. 依次点击卡片的四个角点（左上、右上、右下、左下）
4. 按 's' 保存标签，按 'n' 下一个图片，按 'q' 退出
"""

import os
import cv2
import numpy as np
from pathlib import Path


class LabelTool:
    """卡片角点标注工具"""

    def __init__(self, images_dir='data/train/images', labels_dir='data/train/labels'):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # 获取未标注的图片
        self.image_files = []
        for img_path in self.images_dir.glob('*.jpg') + self.images_dir.glob('*.png'):
            label_path = self.labels_dir / (img_path.stem + '.txt')
            if not label_path.exists():
                self.image_files.append(img_path)

        self.current_idx = 0
        self.points = []
        self.image = None
        self.clone = None

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标点击回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                print(f"点击点 {len(self.points)}: ({x}, {y})")
                self.draw_points()

    def draw_points(self):
        """绘制点"""
        self.image = self.clone.copy()

        # 绘制已点击的点
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        labels = ['1-左上', '2-右上', '3-右下', '4-左下']

        for i, pt in enumerate(self.points):
            cv2.circle(self.image, pt, 10, colors[i], -1)
            cv2.putText(self.image, labels[i], (pt[0]+15, pt[1]-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)

        # 绘制连接线
        if len(self.points) >= 2:
            pts = np.array(self.points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(self.image, [pts], isClosed=False, color=(255, 255, 0), thickness=2)

        # 显示说明
        cv2.putText(self.image, f"Points: {len(self.points)}/4 | 's': save | 'n': next | 'q': quit",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Label Tool', self.image)

    def label_image(self, img_path):
        """标注单张图片"""
        self.image = cv2.imread(str(img_path))
        self.clone = self.image.copy()
        self.points = []

        print(f"\n{'='*50}")
        print(f"正在标注: {img_path.name}")
        print("请依次点击:1-左上, 2-右上, 3-右下, 4-左下")
        print("按 's' 保存, 按 'n' 跳过, 按 'q' 退出")
        print(f"{'='*50}")

        self.draw_points()

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('s') and len(self.points) == 4:
                # 保存标签
                h, w = self.image.shape[:2]
                label = []
                for pt in self.points:
                    label.append(f"{pt[0]/w:.4f},{pt[1]/h:.4f}")

                label_str = ' '.join(label)
                label_path = self.labels_dir / (img_path.stem + '.txt')

                with open(label_path, 'w') as f:
                    f.write(label_str)

                print(f"已保存: {label_path}")
                return True

            elif key == ord('n'):
                print("跳过此图片")
                return False

            elif key == ord('q'):
                print("退出标注工具")
                return None

            elif key == 27:  # ESC
                print("跳过此图片")
                return False

    def run(self):
        """运行标注工具"""
        cv2.namedWindow('Label Tool')

        while self.current_idx < len(self.image_files):
            img_path = self.image_files[self.current_idx]
            result = self.label_image(img_path)

            if result is None:
                break
            elif result:
                self.current_idx += 1

        cv2.destroyAllWindows()

        remaining = len([p for p in self.images_dir.glob('*.*')
                        if not (self.labels_dir / (p.stem + '.txt')).exists()])
        print(f"\n标注完成!剩余 {remaining} 张图片待标注")


if __name__ == '__main__':
    tool = LabelTool()
    tool.run()