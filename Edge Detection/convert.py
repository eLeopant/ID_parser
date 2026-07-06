"""import json
import os

# 你的json目录（labelme保存的json）
json_dir = "json_labels"

# 输出txt目录
txt_dir = "data/train/labels"
os.makedirs(txt_dir, exist_ok=True)

for json_name in os.listdir(json_dir):
    if not json_name.endswith(".json"):
        continue

    try:
        with open(os.path.join(json_dir, json_name), encoding="utf-8") as f:
            data = json.load(f)

        # 获取标注点（修复：兼容所有LabelMe格式）
        shapes = data.get("shapes", [])
        if not shapes:
            print(f"⚠️  {json_name} 没有标注内容，跳过")
            continue
        
        points = shapes[0]["points"]  # 4个角坐标

        # 自动读取4个点，不管顺序
        coords = []
        for (x, y) in points:
            coords.append(str(round(x, 2)))
            coords.append(str(round(y, 2)))
        
        # 生成标签行
        label_line = "0 " + " ".join(coords) + "\n"

        # 写入txt
        txt_name = json_name.replace(".json", ".txt")
        with open(os.path.join(txt_dir, txt_name), "w") as f:
            f.write(label_line)
        
        print(f"✅ 转换成功：{json_name} -> {txt_name}")
    
    except Exception as e:
        print(f"❌ 处理 {json_name} 失败：{str(e)}")

print("\n🏁 全部转换完成！")"""
import json
import os

# 你的json目录（labelme保存的json）
json_dir = "json_labels"

# 输出txt目录
txt_dir = "data/train/labels"
os.makedirs(txt_dir, exist_ok=True)

for json_name in os.listdir(json_dir):
    if not json_name.endswith(".json"):
        continue

    json_path = os.path.join(json_dir, json_name)
    txt_name = os.path.splitext(json_name)[0] + ".txt"
    txt_path = os.path.join(txt_dir, txt_name)

    try:
        # 1. 读取JSON：指定utf-8编码，遇到无法解码的字节自动忽略，防止程序崩溃
        with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)

        # 2. 获取标注点（兼容LabelMe格式）
        shapes = data.get("shapes", [])
        if not shapes:
            print(f"⚠️  {json_name} 没有标注内容，跳过")
            continue
        
        # 增加对 points 键的防御性检查
        points = shapes[0].get("points", [])
        if not points:
            print(f"⚠️  {json_name} 标注点为空，跳过")
            continue

        # 3. 自动读取坐标并格式化
        coords = []
        for (x, y) in points:
            coords.append(str(round(x, 2)))
            coords.append(str(round(y, 2)))
        
        # 生成标签行
        label_line = "0 " + " ".join(coords) + "\n"

        # 4. 写入TXT：显式指定utf-8编码，防止Windows下中文乱码
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(label_line)
        
        print(f"✅ 转换成功：{json_name} -> {txt_name}")
    
    except Exception as e:
        print(f"❌ 处理 {json_name} 失败：{str(e)}")

print("\n🏁 全部转换完成！")