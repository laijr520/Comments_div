import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# 读取已标注的样本
df = pd.read_csv('review_sample_labeled.csv')

# 过滤掉未标注的行（如果有）
df = df.dropna(subset=['人工标签'])
df = df[df['人工标签'].str.strip() != '']
print(f'有效标注: {len(df)} 条\n')

y_llm = df['primary'].values
y_human = df['人工标签'].values

# ========= 1. 整体一致率 + Cohen's kappa =========
agree_rate = (y_llm == y_human).mean()
kappa = cohen_kappa_score(y_human, y_llm)

print('='*50)
print('整体信度指标')
print('='*50)
print(f'样本数 N = {len(df)}')
print(f'整体一致率 (Agreement Rate): {agree_rate:.1%}')
print(f"Cohen's Kappa: {kappa:.3f}")

# kappa 等级解读（Landis & Koch 1977）
if kappa < 0:   level = '劣于随机'
elif kappa < 0.21: level = 'slight (轻微一致)'
elif kappa < 0.41: level = 'fair (尚可)'
elif kappa < 0.61: level = 'moderate (中等)'
elif kappa < 0.81: level = 'substantial (高度一致)'
else:             level = 'almost perfect (近乎完美)'
print(f'一致性等级: {level}')

# ========= 2. 分类报告（精确率/召回率/F1） =========
print('\n' + '='*50)
print('分类报告（以人工标注为真值）')
print('='*50)
print(classification_report(y_human, y_llm, zero_division=0, digits=3))

# ========= 3. 各类别一致率拆解 =========
print('='*50)
print('各 LLM 预测类别的一致率')
print('='*50)
rows = []
for cat in sorted(set(y_llm)):
    mask = y_llm == cat
    n = mask.sum()
    correct = (y_llm[mask] == y_human[mask]).sum()
    rate = correct / n if n else 0
    rows.append((cat, n, correct, rate))
    print(f'  {cat:<35s} {correct:>3d}/{n:<3d} = {rate:.1%}')

# ========= 4. 混淆矩阵可视化 =========
labels = sorted(set(list(y_llm) + list(y_human)))
cm = confusion_matrix(y_human, y_llm, labels=labels)

# 中文标签映射
label_map = {
    'confirmation_bias':'确认偏误','ingroup_bias':'组内偏见',
    'fundamental_attribution_error':'基本归因错误','motivated_reasoning':'动机性推理',
    'hostile_attribution_bias':'敌意归因偏差','none':'无明显偏误'
}
labels_cn = [label_map.get(l, l) for l in labels]

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels_cn, rotation=30, ha='right')
ax.set_yticklabels(labels_cn)
ax.set_xlabel('LLM 预测')
ax.set_ylabel('人工标注（真值）')
ax.set_title(f'LLM vs 人工标注混淆矩阵\n一致率={agree_rate:.1%}, κ={kappa:.3f}')

# 格子里写数字
for i in range(len(labels)):
    for j in range(len(labels)):
        color = 'white' if cm[i,j] > cm.max()/2 else 'black'
        ax.text(j, i, cm[i,j], ha='center', va='center', color=color)

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('fig_confusion.png', dpi=150)
print('\n混淆矩阵已保存: fig_confusion.png')

# ========= 5. 导出分歧清单供论文附录 =========
disagree = df[df['primary'] != df['人工标签']].copy()
disagree = disagree[['评论ID','评论内容','primary','人工标签','reason']]
disagree.columns = ['评论ID','评论内容','LLM分类','人工分类','LLM判断理由']
disagree.to_csv('disagreement_cases.csv', index=False, encoding='utf-8-sig')
print(f'分歧案例已导出: disagreement_cases.csv ({len(disagree)} 条)')