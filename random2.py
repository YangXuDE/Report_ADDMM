import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('combined_data.csv')  # 请将此处替换为实际输入文件名

# 分离表头和数据
header = df.columns
data = df.values

# 随机打乱数据行
np.random.shuffle(data)

# 创建新的DataFrame，保留原始表头
shuffled_df = pd.DataFrame(data, columns=header)

# 保存完整打乱后的数据
shuffled_df.to_csv('missed_data.csv', index=False)

# 从打乱后的数据中随机抽取2000行
if len(shuffled_df) >= 2000:
    random_2000_df = shuffled_df.sample(n=1000, random_state=42)  # random_state可选，用于可重现性
    random_2000_df.to_csv('missed_data_random1000_10.csv', index=False)
    print("已完成随机打乱和抽样:")
    print(f"- 完整数据保存到 'missed_data.csv' ({len(shuffled_df)}行)")
    print(f"- 随机抽取的2000行数据保存到 'missed_data_random1000.csv'")
else:
    print("警告：数据行数少于2000行")
    print(f"已将全部 {len(shuffled_df)} 行保存到 'missed_data.csv'")

# 可选：查看结果
print("\n打乱后完整数据前5行预览:")
print(shuffled_df.head())
print("\n随机抽取数据前5行预览:")
if len(shuffled_df) >= 2000:
    print(random_2000_df.head())