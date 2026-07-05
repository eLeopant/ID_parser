#!/usr/bin/env python3
"""
身份证 OCR 训练数据生成脚本
生成模拟身份证图片及 PaddleOCR 格式标注
"""

import os
import sys
import json
import random
import string
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 配置
# ============================================================
ID_CARD_WIDTH = 1024
ID_CARD_HEIGHT = 658

# 各字段在图片上的固定位置 (left, top, right, bottom)
FIELD_POSITIONS = {
    "name":     (200, 108,  600, 168),
    "gender":   (200, 186,  310, 238),
    "ethnicity":(432, 186,  540, 238),
    "birthday": (200, 256,  700, 310),
    "address":  (200, 334,  900, 440),
    "id_number":(200, 472,  890, 530),
}

# ============================================================
# 跨平台中文字体检测
# ============================================================

def detect_font_path():
    """检测系统中可用的中文字体路径（Windows / Linux / macOS）"""
    # Windows 常见中文字体
    if sys.platform == "win32":
        win_fonts = [
            os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "msyh.ttc"),   # 微软雅黑
            os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "simsun.ttc"),  # 宋体
            os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "simhei.ttf"),  # 黑体
            os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "msyhbd.ttf"),  # 微软雅黑粗体
        ]
        for path in win_fonts:
            if os.path.isfile(path):
                print(f"  检测到字体: {path}")
                return path
        print("⚠️  未找到 Windows 系统中文字体，请手动设置 FONT_PATH")

    # macOS 常见中文字体
    elif sys.platform == "darwin":
        mac_fonts = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
        for path in mac_fonts:
            if os.path.isfile(path):
                print(f"  检测到字体: {path}")
                return path
        print("⚠️  未找到 macOS 系统中文字体，请手动设置 FONT_PATH")

    # Linux 常见中文字体
    else:
        linux_fonts = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]
        for path in linux_fonts:
            if os.path.isfile(path):
                print(f"  检测到字体: {path}")
                return path
        print("⚠️  未找到 Linux 系统中文字体，请安装 fonts-noto-cjk 或手动设置 FONT_PATH")

    return None


FONT_PATH = detect_font_path()

# 输出目录
OUTPUT_DIR_TRAIN = "data/train"
OUTPUT_DIR_VAL = "data/val"

# 生成数量
NUM_TRAIN = 200
NUM_VAL = 50

# ============================================================
# 随机数据生成器
# ============================================================

SURNAMES = [
    "王", "李", "张", "刘", "陈", "杨", "黄", "赵", "周", "吴",
    "徐", "孙", "马", "朱", "胡", "郭", "何", "林", "罗", "高",
    "梁", "郑", "谢", "宋", "唐", "韩", "曹", "许", "邓", "冯",
    "萧", "程", "蔡", "彭", "潘", "袁", "董", "余", "苏", "叶",
    "卢", "蒋", "田", "杜", "丁", "沈", "任", "姚", "卢", "傅",
    "钟", "汪", "范", "方", "石", "戴", "谭", "廖", "邹", "熊",
    "金", "陆", "郝", "孔", "白", "崔", "毛", "邱", "秦", "江",
    "史", "顾", "侯", "邵", "孟", "龙", "万", "段", "雷", "钱",
    "汤", "尹", "黎", "易", "常", "武", "乔", "贺", "赖", "龚",
    "文", "夏", "欧阳", "上官", "慕容", "司马", "诸葛", "夏侯",
]

GIVEN_NAMES = [
    "伟", "芳", "娜", "敏", "静", "丽", "强", "磊", "军", "洋",
    "勇", "艳", "杰", "娟", "涛", "明", "超", "秀英", "霞", "平",
    "刚", "建华", "文", "飞", "斌", "宇", "鑫", "浩", "雪", "琳",
    "志强", "海燕", "佳", "思雨", "晨", "子涵", "宇航", "欣怡",
    "浩然", "思琪", "俊杰", "雅琴", "天佑", "梓萱", "明辉", "雨桐",
]

GENDERS = ["男", "女"]

ETHNICITIES = [
    "汉", "蒙古", "回", "藏", "维吾尔", "苗", "彝", "壮",
    "布依", "朝鲜", "满", "侗", "瑶", "白", "土家", "哈尼",
    "哈萨克", "傣", "黎", "傈僳", "佤", "畲", "高山", "拉祜",
    "水", "东乡", "纳西", "景颇", "柯尔克孜", "土", "达斡尔",
    "仫佬", "羌", "布朗", "撒拉", "毛南", "仡佬", "锡伯",
    "阿昌", "普米", "塔吉克", "怒", "乌孜别克", "俄罗斯",
    "鄂温克", "德昂", "保安", "裕固", "京", "塔塔尔", "独龙",
    "鄂伦春", "赫哲", "门巴", "珞巴", "基诺",
]

CITIES = [
    "北京市", "上海市", "广州市", "深圳市", "杭州市", "成都市",
    "武汉市", "南京市", "西安市", "重庆市", "天津市", "苏州市",
    "长沙市", "郑州市", "青岛市", "大连市", "昆明市", "厦门市",
    "哈尔滨市", "长春市", "沈阳市", "济南市", "合肥市", "福州市",
    "南昌市", "南宁市", "海口市", "贵阳市", "兰州市", "西宁市",
]

DISTRICTS = [
    "朝阳区", "海淀区", "浦东新区", "天河区", "南山区", "西湖区",
    "武侯区", "洪山区", "鼓楼区", "雁塔区", "渝中区", "和平区",
    "姑苏区", "芙蓉区", "金水区", "市南区", "中山区", "思明区",
    "道里区", "南关区", "沈河区", "历下区", "蜀山区", "鼓楼区",
    "青山湖区", "龙华区", "云岩区", "城关区", "城中区",
]

STREETS = [
    "中山路", "人民路", "解放路", "建设路", "和平路", "新华路",
    "长江路", "黄河路", "南京路", "北京路", "上海路", "科技路",
    "高新路", "朝阳街", "文明街", "花园路", "学院路", "文化路",
    "东风路", "长安街", "建国路", "复兴路", "胜利路", "光明路",
    "幸福路", "平安街", "为民街", "团结路", "友好路", "健康路",
]


def random_name():
    surname = random.choice(SURNAMES)
    given = random.choice(GIVEN_NAMES)
    return surname + given


def random_ethnicity():
    return random.choice(ETHNICITIES)


def random_address():
    city = random.choice(CITIES)
    district = random.choice(DISTRICTS)
    street = random.choice(STREETS)
    num = random.randint(1, 999)
    unit = random.choice(["号", "号院", "号楼", "号小区"])
    room = f"{random.randint(1, 30)}单元{random.randint(101, 2501)}室"
    return f"{city}{district}{street}{num}{unit}{room}"


def random_id_number():
    """生成符合校验规则的 18 位身份证号"""
    addr_code = f"{random.randint(11, 82):02d}{random.randint(1, 99):02d}{random.randint(1, 99):02d}"
    # 出生日期：1960-2005 年
    year = random.randint(1960, 2005)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    birth = f"{year:04d}{month:02d}{day:02d}"
    seq = f"{random.randint(1, 999):03d}"

    base = addr_code + birth + seq
    # 校验码
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    check_codes = "10X98765432"
    total = sum(int(base[i]) * weights[i] for i in range(17))
    check = check_codes[total % 11]
    return base + check


def random_birthday(id_number):
    """从身份证号提取出生年月日"""
    year = id_number[6:10]
    month = id_number[10:12]
    day = id_number[12:14]
    return f"{year}年{month}月{day}日"


def random_gender():
    return random.choice(GENDERS)


# ============================================================
# 图片生成
# ============================================================

def generate_id_card_image(name, gender, ethnicity, birthday, address, id_number):
    """生成一张模拟身份证图片，返回 PIL Image"""
    # 灰白色背景
    img = Image.new("RGB", (ID_CARD_WIDTH, ID_CARD_HEIGHT), color=(248, 248, 248))
    draw = ImageDraw.Draw(img)

    # 加载字体（不同字段用不同字号）
    font_large = ImageFont.truetype(FONT_PATH, 44)
    font_mid = ImageFont.truetype(FONT_PATH, 38)
    font_small = ImageFont.truetype(FONT_PATH, 32)

    # 内容字段颜色 (深灰色)
    text_color = (50, 50, 50)

    # ----- 绘制各个字段 -----
    # 姓名
    x1, y1, x2, y2 = FIELD_POSITIONS["name"]
    draw.text((x1, y1), name, fill=text_color, font=font_mid)

    # 性别
    x1, y1, x2, y2 = FIELD_POSITIONS["gender"]
    draw.text((x1, y1), gender, fill=text_color, font=font_mid)

    # 民族
    x1, y1, x2, y2 = FIELD_POSITIONS["ethnicity"]
    draw.text((x1, y1), ethnicity, fill=text_color, font=font_mid)

    # 出生
    x1, y1, x2, y2 = FIELD_POSITIONS["birthday"]
    draw.text((x1, y1), birthday, fill=text_color, font=font_mid)

    # 住址（如果文字太长，分两行）
    x1, y1, x2, y2 = FIELD_POSITIONS["address"]
    font_addr = ImageFont.truetype(FONT_PATH, 32)
    addr_lines = split_text_to_lines(address, font_addr, x2 - x1)
    for i, line in enumerate(addr_lines):
        draw.text((x1, y1 + i * 42), line, fill=text_color, font=font_addr)

    # 身份证号
    x1, y1, x2, y2 = FIELD_POSITIONS["id_number"]
    draw.text((x1, y1), id_number, fill=(50, 50, 50), font=font_large)

    # 略微添加噪点让图片更真实
    add_noise(img)

    return img


def split_text_to_lines(text, font, max_width):
    """将长文本按宽度拆分为多行"""
    lines = []
    current = ""
    for char in text:
        test_line = current + char
        bbox = font.getbbox(test_line)
        w = bbox[2] - bbox[0]
        if w > max_width and current:
            lines.append(current)
            current = char
        else:
            current = test_line
    if current:
        lines.append(current)
    return lines


def add_noise(img, intensity=0.3):
    """添加极少的随机噪点模拟真实卡片质感"""
    from PIL import ImageStat
    pixels = img.load()
    w, h = img.size
    noise_count = int(w * h * 0.002 * intensity)
    for _ in range(noise_count):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        noise = random.randint(-8, 8)
        r, g, b = pixels[x, y]
        pixels[x, y] = (
            max(0, min(255, r + noise)),
            max(0, min(255, g + noise)),
            max(0, min(255, b + noise)),
        )


# ============================================================
# 标注生成（PaddleOCR 格式）
# ============================================================

def get_text_bbox(x, y, text, font):
    """计算文字在图片上的实际包围框"""
    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return [x, y, x + w, y + h]


def generate_annotation(name, gender, ethnicity, birthday, address, id_number):
    """
    为一张身份证图生成 PaddleOCR 格式标注。
    返回列表: [{"transcription": "...", "points": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]}, ...]
    """
    font_mid = ImageFont.truetype(FONT_PATH, 38)
    font_large = ImageFont.truetype(FONT_PATH, 44)
    font_addr = ImageFont.truetype(FONT_PATH, 32)

    annotation = []

    # 姓名
    x, y, _, _ = FIELD_POSITIONS["name"]
    lx, ly, rx, ry = get_text_bbox(x, y, name, font_mid)
    annotation.append({
        "transcription": name,
        "points": [[lx, ly], [rx, ly], [rx, ry], [lx, ry]]
    })

    # 性别
    x, y, _, _ = FIELD_POSITIONS["gender"]
    lx, ly, rx, ry = get_text_bbox(x, y, gender, font_mid)
    annotation.append({
        "transcription": gender,
        "points": [[lx, ly], [rx, ly], [rx, ry], [lx, ry]]
    })

    # 民族
    x, y, _, _ = FIELD_POSITIONS["ethnicity"]
    lx, ly, rx, ry = get_text_bbox(x, y, ethnicity, font_mid)
    annotation.append({
        "transcription": ethnicity,
        "points": [[lx, ly], [rx, ly], [rx, ry], [lx, ry]]
    })

    # 出生
    x, y, _, _ = FIELD_POSITIONS["birthday"]
    lx, ly, rx, ry = get_text_bbox(x, y, birthday, font_mid)
    annotation.append({
        "transcription": birthday,
        "points": [[lx, ly], [rx, ly], [rx, ry], [lx, ry]]
    })

    # 地址（可能多行）
    x, y, x2, y2 = FIELD_POSITIONS["address"]
    addr_lines = split_text_to_lines(address, font_addr, x2 - x)
    for i, line in enumerate(addr_lines):
        lx, ly, rx, ry = get_text_bbox(x, y + i * 42, line, font_addr)
        annotation.append({
            "transcription": line,
            "points": [[lx, ly], [rx, ly], [rx, ry], [lx, ry]]
        })

    # 身份证号
    x, y, _, _ = FIELD_POSITIONS["id_number"]
    lx, ly, rx, ry = get_text_bbox(x, y, id_number, font_large)
    annotation.append({
        "transcription": id_number,
        "points": [[lx, ly], [rx, ly], [rx, ry], [lx, ry]]
    })

    return annotation


# ============================================================
# 主流程
# ============================================================

def generate_dataset(num_samples, output_dir):
    """生成数据集"""
    os.makedirs(output_dir, exist_ok=True)
    label_path = os.path.join(output_dir, "label.txt")

    with open(label_path, "w", encoding="utf-8") as f_label:
        for i in range(num_samples):
            # 生成随机数据
            name = random_name()
            gender = random_gender()
            ethnicity = random_ethnicity()
            id_number = random_id_number()
            birthday = random_birthday(id_number)
            address = random_address()

            # 生成图片
            img = generate_id_card_image(name, gender, ethnicity, birthday, address, id_number)

            # 保存图片
            filename = f"id_card_{i:04d}.jpg"
            img_path = os.path.join(output_dir, filename)
            img.save(img_path, quality=92)

            # 生成标注
            annotation = generate_annotation(name, gender, ethnicity, birthday, address, id_number)
            label_line = f"{filename}\t{json.dumps(annotation, ensure_ascii=False)}\n"
            f_label.write(label_line)

            if (i + 1) % 50 == 0:
                print(f"  ✨ 已生成 {i + 1}/{num_samples} 张")

    print(f"✅ 数据集生成完成！共 {num_samples} 张，存放于 {output_dir}")


def main():
    if FONT_PATH is None:
        print("❌ 未找到可用的中文字体！")
        print("   Windows: 确保 C:\\Windows\\Fonts 下有 msyh.ttc 或 simsun.ttc")
        print("   Linux:   sudo apt install fonts-noto-cjk -y")
        print("   macOS:   系统自带 PingFang 字体，无需额外安装")
        sys.exit(1)

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=" * 50)
    print("  身份证 OCR 训练数据生成")
    print("=" * 50)

    print("\n📁 生成训练集 (200 张)...")
    generate_dataset(NUM_TRAIN, OUTPUT_DIR_TRAIN)

    print("\n📁 生成验证集 (50 张)...")
    generate_dataset(NUM_VAL, OUTPUT_DIR_VAL)

    total = NUM_TRAIN + NUM_VAL
    print(f"\n{'=' * 50}")
    print(f"  🎉 全部完成！共生成 {total} 张身份证图片")
    print(f"  📂 训练集: {OUTPUT_DIR_TRAIN}/")
    print(f"  📂 验证集: {OUTPUT_DIR_VAL}/")
    print(f"  📄 标注文件: {OUTPUT_DIR_TRAIN}/label.txt, {OUTPUT_DIR_VAL}/label.txt")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
