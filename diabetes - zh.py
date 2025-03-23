"""
糖尿病早期检测模型    后端
这个脚本实现了一个完整的机器学习工作流程，用于预测糖尿病风险。
"""

# 导入必要的库
import pandas as pd  # 用于数据处理和分析
import numpy as np  # 用于数学计算和数组操作
from sklearn.model_selection import train_test_split  # 用于将数据集分割为训练集和测试集
from sklearn.preprocessing import StandardScaler  # 用于特征标准化
from sklearn.impute import SimpleImputer  # 用于处理缺失值
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器
from xgboost import XGBClassifier  # XGBoost梯度提升分类器
from sklearn.linear_model import LogisticRegression  # 逻辑回归分类器
from sklearn.metrics import (accuracy_score, precision_score,  # 导入各种评估指标
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt  # 用于绘图
import seaborn as sns  # 用于高级数据可视化
import joblib  # 用于保存和加载模型
import os  # 用于操作系统功能，如创建文件夹


# 临床整合指南
def print_clinical_integration_guidelines():
    """
    打印如何将机器学习工具整合到临床工作流程中的建议指南。
    包括筛查阶段、风险分层、临床决策支持、数据整合和持续改进等方面的建议。
    """
    print("\n" + "=" * 50)
    print("临床整合指南")
    print("=" * 50)
    print("""
    基于本项目的研究结果，我们建议将这个机器学习工具整合到临床工作流程中，具体如下：

    1. 筛查阶段：
       - 在常规体检中收集患者数据
       - 使用本系统进行初步风险评估
       - 为高风险患者安排进一步的糖尿病筛查测试

    2. 风险分层：
       - 高风险(>70%)：立即转诊专业评估
       - 中等风险(40-70%)：干预措施和6个月内随访
       - 低风险(<40%)：预防建议和年度评估

    3. 临床决策支持：
       - 将预测结果作为临床决策的辅助工具
       - 医生应将结果与患者病史和其他临床发现结合起来
       - 本工具不能替代专业医疗判断

    4. 数据整合：
       - 在电子健康记录系统中记录风险评估结果
       - 建立追踪机制以监测高风险患者

    5. 持续改进：
       - 定期用新数据更新模型
       - 收集临床反馈以优化算法
       - 每年评估工具的临床有效性
    """)
    print("=" * 50)


# 1. 数据加载和探索
df = pd.read_csv(r"D:\AAterm2\524 ML\Group\1\Dataset of Diabetes .csv")  # 从指定路径加载CSV文件数据

# 查看基本数据信息
print("数据维度:", df.shape)  # 打印数据集的行数和列数
print("\n数据前5行:")
print(df.head())  # 显示数据集的前5行，用于初步了解数据结构
print("\n数据统计信息:")
print(df.describe())  # 显示数据的统计摘要，包括计数、均值、标准差、最小值、最大值等
print("\n缺失值统计:")
print(df.isnull().sum())  # 统计每列的缺失值数量

# 2. 数据预处理
# 处理性别编码
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})  # 将'M'(男性)编码为1，'F'(女性)编码为0
# 处理类别变量编码（合并糖尿病前期和糖尿病类别）
if 'CLASS' in df.columns:
    if df['CLASS'].dtype == object:  # 检查CLASS列是否为对象类型（字符串）
        df['CLASS'] = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 1})  # 将'N'(正常)编码为0，'P'(糖尿病前期)和'Y'(糖尿病)均编码为1

# 使用中位数填充缺失值
imputer = SimpleImputer(strategy='median')  # 创建中位数填充器
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)  # 使用中位数填充缺失值并保留原列名

# 特征选择
features = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'Gender']  # 选择用于训练模型的特征
target = 'CLASS'  # 目标变量（我们要预测的变量）

# 确保所有特征都存在于数据集中
missing_features = [f for f in features if f not in df.columns]  # 检查哪些特征不在数据集中
if missing_features:
    print(f"警告：数据集中缺少以下特征: {missing_features}")
    print("请检查您的数据集或修改特征列表。可用列: ", df.columns.tolist())
    features = [f for f in features if f in df.columns]  # 移除不存在的特征
    print(f"将继续使用以下特征: {features}")

# 标准化连续特征
scaler = StandardScaler()  # 创建标准化器
X = scaler.fit_transform(df_imputed[features])  # 对特征进行标准化处理（均值为0，方差为1）
y = df_imputed[target]  # 提取目标变量

# 分割为训练集和测试集（80-20分割，带分层抽样）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)  # random_state确保结果可复现，stratify确保类别比例保持一致

print(f"\n使用以下特征训练模型: {features}")
print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 3. 模型训练和比较
models = {
    "逻辑回归": LogisticRegression(max_iter=1000),  # 逻辑回归模型，增加最大迭代次数以确保收敛
    "随机森林": RandomForestClassifier(n_estimators=100),  # 随机森林模型，使用100棵决策树
    "XGBoost": XGBClassifier(eval_metric='logloss')  # XGBoost模型，使用对数损失作为评估指标
}

# 训练每个模型并评估性能
results = {}
for name, model in models.items():
    print(f"\n训练模型: {name}")
    model.fit(X_train, y_train)  # 使用训练集训练模型
    y_pred = model.predict(X_test)  # 对测试集进行预测

    # 计算评估指标
    results[name] = {
        '准确率': accuracy_score(y_test, y_pred),  # 正确预测的比例
        '精确率': precision_score(y_test, y_pred),  # 预测为阳性的样本中真正为阳性的比例
        '召回率': recall_score(y_test, y_pred),  # 真正为阳性的样本中被正确预测为阳性的比例
        'F1分数': f1_score(y_test, y_pred),  # 精确率和召回率的调和平均数
        'AUC-ROC': roc_auc_score(y_test, y_pred)  # ROC曲线下面积，衡量模型区分两个类别的能力
    }

# 4. 结果可视化和评估
# 输出性能比较
results_df = pd.DataFrame(results).T  # 将结果转换为DataFrame以便于显示
print("\n模型性能比较:")
print(results_df)

# 基于F1分数找出最佳模型
best_model_name = results_df['F1分数'].idxmax()  # 找出F1分数最高的模型
print(f"\n基于F1分数的最佳模型: {best_model_name}")

# 创建类别分布可视化
plt.figure(figsize=(10, 6))  # 设置图形大小
class_counts = df_imputed[target].value_counts()  # 计算每个类别的样本数量
colors = ['green', 'red'] if len(class_counts) == 2 else ['green', 'orange', 'red']  # 设置颜色
ax = class_counts.plot(kind='bar', color=colors)  # 创建条形图
plt.title('糖尿病病例分布', fontsize=16)  # 设置标题
plt.xlabel('类别 (0: 非糖尿病, 1: 糖尿病前期 & 糖尿病)', fontsize=14)  # 设置x轴标签
plt.ylabel('计数', fontsize=14)  # 设置y轴标签
plt.xticks(rotation=0, fontsize=12)  # 设置x轴刻度标签
plt.yticks(fontsize=12)  # 设置y轴刻度标签

# 添加数值标签
for i, v in enumerate(class_counts):
    ax.text(i, v + 5, str(v), ha='center', fontsize=12)  # 在每个条形上方显示数值

# 添加百分比标签
total = class_counts.sum()  # 计算总样本数
for i, v in enumerate(class_counts):
    percentage = v / total * 100  # 计算百分比
    ax.text(i, v / 2, f"{percentage:.1f}%", ha='center', color='white', fontsize=12, fontweight='bold')  # 在条形中间显示百分比

plt.tight_layout()  # 自动调整子图参数，使之填充整个图形区域
plt.savefig('class_distribution.png')  # 保存图片
plt.show()  # 显示图形
plt.close()  # 关闭图形
print("\n类别分布图已保存为: class_distribution.png")

# 创建特征重要性可视化（使用随机森林）
rf = models['随机森林']  # 获取随机森林模型
importances = rf.feature_importances_  # 获取特征重要性
indices = np.argsort(importances)[::-1]  # 按重要性降序排序

plt.figure(figsize=(12, 6))  # 设置图形大小
plt.title("特征重要性 - 随机森林", fontsize=16)  # 设置标题
plt.bar(range(len(features)), importances[indices], align='center', color='dodgerblue')  # 创建条形图
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha='right', fontsize=12)  # 设置x轴刻度标签
plt.yticks(fontsize=12)  # 设置y轴刻度标签
plt.xlabel("特征", fontsize=14)  # 设置x轴标签
plt.ylabel("重要性", fontsize=14)  # 设置y轴标签
plt.grid(axis='y', linestyle='--', alpha=0.6)  # 添加网格线

# 添加数值标签
for i, v in enumerate(importances[indices]):
    plt.text(i, v + 0.005, f"{v:.3f}", ha='center', fontsize=10)  # 在每个条形上方显示特征重要性值

plt.tight_layout()  # 自动调整子图参数
plt.savefig('feature_importance.png')  # 保存图片
plt.show()  # 显示图形
plt.close()  # 关闭图形
print("\n特征重要性图已保存为: feature_importance.png")

# 创建混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)  # 计算混淆矩阵
plt.figure(figsize=(8, 6))  # 设置图形大小
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)  # 创建热图
plt.title(f'混淆矩阵 - {best_model_name}', fontsize=16)  # 设置标题
plt.xlabel('预测标签', fontsize=14)  # 设置x轴标签
plt.ylabel('真实标签', fontsize=14)  # 设置y轴标签
plt.xticks([0.5, 1.5], ['非糖尿病 (0)', '糖尿病 (1)'], fontsize=12)  # 设置x轴刻度标签
plt.yticks([0.5, 1.5], ['非糖尿病 (0)', '糖尿病 (1)'], fontsize=12, rotation=0)  # 设置y轴刻度标签

# 为混淆矩阵添加百分比标签
total = cm.sum()  # 计算总样本数
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        percentage = cm[i, j] / total * 100  # 计算每个单元格的百分比
        plt.text(j + 0.5, i + 0.7, f"{percentage:.1f}%", ha='center',
                 color='black' if cm[i, j] < cm.max() / 2 else 'white', fontsize=10)  # 在单元格中显示百分比

plt.tight_layout()  # 自动调整子图参数
plt.savefig('confusion_matrix.png')  # 保存图片
plt.show()  # 显示图形
plt.close()  # 关闭图形
print("\n混淆矩阵已保存为: confusion_matrix.png")

# 生成相关性分析热图
print("\n生成相关性热图...")
# 添加目标变量进行相关性分析
corr_features = features.copy()  # 复制特征列表
if target in df.columns:
    corr_features.append(target)  # 添加目标变量到相关性分析

# 计算相关矩阵
corr_matrix = df_imputed[corr_features].corr()  # 计算特征之间的相关系数
plt.figure(figsize=(12, 10))  # 设置图形大小
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")  # 创建热图，显示相关系数
plt.title('特征相关性热图')  # 设置标题
plt.tight_layout()  # 自动调整子图参数
plt.savefig('correlation_heatmap.png')  # 保存图片
plt.show()  # 显示图形
print("相关性热图已保存为: correlation_heatmap.png")

# 5. 最佳模型选择和保存
best_model = models[best_model_name]  # 获取最佳模型

# 生成分类报告
y_pred = best_model.predict(X_test)  # 使用最佳模型对测试集进行预测
print("\n最佳模型分类报告:")
print(classification_report(y_test, y_pred))  # 打印详细的分类性能报告

# 保存模型和相关文件
os.makedirs('models', exist_ok=True)  # 创建models目录（如果不存在）
joblib.dump(best_model, 'models/diabetes_model.pkl')  # 保存最佳模型
joblib.dump(scaler, 'models/scaler.pkl')  # 保存标准化器
joblib.dump(features, 'models/features.pkl')  # 保存特征列表

print(f"\n模型和相关文件已保存到'models'目录。最佳模型: {best_model_name}")

# 输出模型文件大小信息
model_file = 'models/diabetes_model.pkl'
print(f"模型文件大小: {os.path.getsize(model_file) / (1024 * 1024):.2f} MB")  # 显示模型文件的大小（MB）

# 输出临床整合指南
print_clinical_integration_guidelines()  # 调用函数打印临床整合指南

# 输出Web界面启动说明
print("\n" + "=" * 50)
print("启动Web界面")
print("=" * 50)
print("""
要启动糖尿病风险评估Web界面，请运行以下命令：

    streamlit run app.py

确保您已安装Streamlit库：

    pip install streamlit

Web界面提供了一个适合医疗专业人员使用的用户友好的风险评估工具。
""")
print("=" * 50)