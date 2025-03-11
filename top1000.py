import pandas as pd

# 读取CSV文件的前5000行
df = pd.read_csv('combined_data.csv', nrows=7532)  # 请将此处替换为实际输入文件名

# 直接保存前5000行到新文件
df.to_csv(f'missed_data_top{len(df)}.csv', index=False)

print("已完成提取:")
print(f"- 前5000行数据保存到 'missed_data.csv' ({len(df)}行)")

# 可选：查看结果
print("\n提取数据前5行预览:")
print(df.head())