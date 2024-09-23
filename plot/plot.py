import matplotlib.pyplot as plt
import re

# 定义文件路径
file_name = "results/reflow/score_record.txt"

# 初始化字典存储不同类别的 FID 记录
data = {
    "Rectified flow": [],
    "Collapse-avoid": []
}

# 当前正在读取的类别
current_category = None

# 使用正则表达式读取文件中的数据
with open(file_name, 'r') as file:
    for line in file:
        # 检查类别行
        if "Rectified flow" in line:
            current_category = "Rectified flow"
        elif "Cpllapse-avoid" in line:
            current_category = "Collapse-avoid"
        
        # 使用正则表达式提取 FID 值
        fid_match = re.search(r"Record \d+: FID: (\d+\.\d+)", line)
        if fid_match:
            fid_value = float(fid_match.group(1))
            if current_category:
                data[current_category].append(fid_value)

# 绘制曲线图
plt.figure(figsize=(10, 6))

# 绘制 Rectified flow 的 FID 曲线
plt.plot(data["Rectified flow"], label="Rectified flow", marker='o')

# 绘制 Collapse-avoid 的 FID 曲线
plt.plot(data["Collapse-avoid"], label="Collapse-avoid", marker='s')

# 添加图例和标签
plt.title("FID Values Comparison")
plt.xlabel("Reflow Number")
plt.ylabel("FID Value")
plt.legend()

# 显示图形
plt.grid(True)
plt.savefig("fid_comparison_chart.png", format='png', dpi=300)
# plt.show()
