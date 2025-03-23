"""
糖尿病早期检测系统 - Web界面
该应用程序使用Streamlit创建一个用户友好的界面，用于进行糖尿病风险评估。
"""
import streamlit as st  # 导入Streamlit库，用于创建Web应用
import pandas as pd     # 导入Pandas库，用于数据处理和分析
import numpy as np      # 导入NumPy库，用于科学计算
import joblib           # 导入joblib库，用于加载保存的模型
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于数据可视化
import os               # 导入os库，用于操作系统相关功能
import seaborn as sns   # 导入Seaborn库，用于高级数据可视化
from PIL import Image   # 导入PIL库的Image模块，用于处理图像

# 页面配置设置
st.set_page_config(
    page_title="糖尿病早期检测系统",  # 设置页面标题
    page_icon="🩺",                 # 设置页面图标（医疗符号）
    layout="wide",                 # 使用宽屏布局
    initial_sidebar_state="expanded"  # 初始侧边栏状态为展开
)

# 加载模型和相关文件
@st.cache_resource  # 使用缓存装饰器，提高加载效率（避免重复加载）
def load_model():
    """
    加载保存的模型、标准化器和特征列表

    返回:
        tuple: 包含模型、标准化器和特征列表的元组
    """
    model = joblib.load('models/diabetes_model.pkl')     # 加载训练好的模型
    scaler = joblib.load('models/scaler.pkl')           # 加载标准化器
    features = joblib.load('models/features.pkl')        # 加载特征列表
    return model, scaler, features

# 尝试加载模型
try:
    model, scaler, features = load_model()  # 调用加载模型函数
    model_loaded = True                    # 模型加载成功标志
except FileNotFoundError:
    # 如果找不到模型文件，显示错误信息
    st.error("未找到模型文件。请先运行训练脚本(diabetes.py)。")
    model_loaded = False                   # 模型加载失败标志

# 界面标题
st.title("糖尿病早期检测系统")
st.markdown("---")  # 添加分隔线

# 创建侧边栏
st.sidebar.header("关于本系统")
st.sidebar.info(
    """
    本系统使用机器学习预测糖尿病风险。

    **功能特点:**
    - 基于临床数据的风险评估
    - 个性化风险预测
    - 可视化风险评估报告
    - 批量患者预测功能
    
    Team: TiDuXiaFan
    """
)

# 在侧边栏添加临床整合指南
st.sidebar.markdown("---")
st.sidebar.header("临床整合指南")
st.sidebar.markdown(
    """
    **医疗专业人员参考:**
    1. 作为辅助诊断工具使用，不可替代临床判断
    2. 高风险预测应触发进一步临床评估
    3. 考虑患者病史和家族史进行全面评估
    4. 每6个月重新评估中等风险患者
    5. 定期更新系统以纳入新的临床知识

    **最佳实践工作流程:**
    - 在常规体检中收集数据
    - 使用本系统进行初步风险筛查
    - 为高风险患者安排进一步检查
    - 将结果整合到电子健康记录中
    - 制定个性化预防计划
    """
)

# 主要内容区域，使用标签页分隔
tab1, tab2, tab3 = st.tabs(["风险评估", "模型信息", "数据可视化"])

# 风险评估标签页
with tab1:
    st.header("个体风险评估")

    # 创建两个子标签页用于个体和批量评估
    individual_tab, batch_tab = st.tabs(["个体患者", "批量处理"])

    # 个体患者标签页
    with individual_tab:
        st.markdown("输入个人健康指标进行糖尿病风险评估。")

        # 创建两列布局
        col1, col2 = st.columns(2)

        # 第一列输入
        with col1:
            # 创建各种输入控件，用于收集用户数据
            age = st.number_input("年龄", min_value=18, max_value=100, value=50, step=1)
            gender = st.selectbox("性别", ["男性", "女性"])
            bmi = st.number_input("BMI指数", min_value=10.0, max_value=50.0, value=24.0, format="%.1f", step=0.1)
            hba1c = st.number_input("糖化血红蛋白(%)", min_value=4.0, max_value=15.0, value=6.5, format="%.1f", step=0.1)
            chol = st.number_input("总胆固醇", min_value=2.0, max_value=10.0, value=4.2, format="%.1f",
                                   step=0.1)

        # 第二列输入
        with col2:
            urea = st.number_input("尿素", min_value=1.0, max_value=20.0, value=4.7, format="%.1f", step=0.1)
            cr = st.number_input("肌酐(Cr)", min_value=20, max_value=200, value=46, step=1)
            tg = st.number_input("甘油三酯(TG)", min_value=0.2, max_value=10.0, value=0.9, format="%.1f",
                                 step=0.1)
            hdl = st.number_input("高密度脂蛋白胆固醇(HDL)", min_value=0.5, max_value=4.0, value=2.4, format="%.1f", step=0.1)
            ldl = st.number_input("低密度脂蛋白胆固醇(LDL)", min_value=0.5, max_value=5.0, value=1.4, format="%.1f", step=0.1)
            vldl = st.number_input("极低密度脂蛋白(VLDL)", min_value=0.1, max_value=3.0, value=0.5, format="%.1f", step=0.1)

        # 预测按钮
        predict_btn = st.button("评估风险")

        # 显示预测结果
        if predict_btn and model_loaded:  # 如果点击了预测按钮且模型已加载
            # 准备数据
            gender_code = 1 if gender == "男性" else 0  # 性别编码：男性=1，女性=0
            input_data = pd.DataFrame([[age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi, gender_code]],
                                      columns=features)  # 创建输入数据DataFrame

            # 数据标准化和预测
            input_scaled = scaler.transform(input_data)  # 使用之前训练的标准化器进行标准化
            prediction = model.predict(input_scaled)[0]  # 预测类别
            prediction_prob = model.predict_proba(input_scaled)[0][1]  # 预测糖尿病的概率

            # 显示结果
            st.markdown("---")
            st.subheader("风险评估结果")

            # 根据预测概率显示不同的风险级别和建议
            if prediction_prob >= 0.7:
                risk_level = "高风险"
                risk_color = "red"
                recommendations = """
                **建议:**
                - 立即咨询医生进行全面的糖尿病检测
                - 监测血糖水平
                - 考虑饮食和生活方式调整
                - 安排定期随访检查
                """
            elif prediction_prob >= 0.4:
                risk_level = "中等风险"
                risk_color = "orange"
                recommendations = """
                **建议:**
                - 在3个月内安排糖尿病筛查
                - 增加身体活动
                - 专注于健康饮食习惯
                - 定期监测健康指标
                """
            else:
                risk_level = "低风险"
                risk_color = "green"
                recommendations = """
                **建议:**
                - 保持健康的生活方式
                - 继续定期体检
                - 专注于均衡饮食
                - 保持适当体重
                """

            # 显示风险级别和概率
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"<h1 style='text-align: center; color: {risk_color};'>{risk_level}</h1>",
                            unsafe_allow_html=True)  # 使用HTML格式化显示风险级别
                st.markdown(f"<h3 style='text-align: center;'>风险概率: {prediction_prob:.1%}</h3>",
                            unsafe_allow_html=True)  # 显示风险概率

            with col2:
                st.markdown("### 个性化建议")
                st.markdown(recommendations)  # 显示基于风险级别的个性化建议

                # 显示需要监测的关键指标
                st.markdown("### 需要监测的关键指标")
                key_indicators = []

                # 检查关键健康指标是否超出正常范围
                if hba1c > 5.7:
                    key_indicators.append(f"- 糖化血红蛋白: {hba1c}% - 高于正常范围 (4.0-5.7%)")

                if bmi > 25:
                    key_indicators.append(f"- BMI指数: {bmi} - 高于理想范围 (18.5-24.9)")

                if tg > 1.7:
                    key_indicators.append(f"- 甘油三酯: {tg} - 高于正常范围 (0.4-1.7)")

                # 显示指标列表
                if not key_indicators:
                    st.markdown("所有指标均在正常范围内")
                else:
                    for indicator in key_indicators:
                        st.markdown(indicator)

    # 批量处理标签页
    with batch_tab:
        st.markdown("上传包含患者数据的CSV文件进行批量风险评估。")

        st.markdown("""
        ### CSV文件格式要求
        您的CSV文件应包含以下所有必需的健康指标列:
        - AGE (年龄，单位：岁)
        - Gender (性别，男性=1，女性=0，或使用'M'/'F')
        - BMI (体重指数)
        - HbA1c (糖化血红蛋白，单位：%)
        - Chol (总胆固醇)
        - TG (甘油三酯)
        - HDL (高密度脂蛋白胆固醇)
        - LDL (低密度脂蛋白胆固醇)
        - VLDL (极低密度脂蛋白)
        - Urea (尿素)
        - Cr (肌酐)
        """)

        # 上传CSV文件
        uploaded_file = st.file_uploader("上传患者数据CSV文件", type=['csv'])

        if uploaded_file is not None and model_loaded:  # 如果上传了文件且模型已加载
            # 加载数据
            try:
                df = pd.read_csv(uploaded_file)  # 读取上传的CSV文件
                st.success(f"文件上传成功。找到{len(df)}条记录。")

                # 显示数据预览
                st.subheader("数据预览")
                st.dataframe(df.head())  # 显示前几行数据

                # 检查是否存在必要的列
                missing_features = [f for f in features if f not in df.columns]  # 找出缺失的特征列

                if missing_features:
                    # 尝试处理常见的列名差异
                    if 'Gender' in df.columns and 'gender' not in [col.lower() for col in df.columns]:
                        df['gender'] = df['Gender'].map({'M': 1, 'F': 0, 'Male': 1, 'Female': 0})  # 标准化性别编码
                        if 'Gender' in missing_features:
                            missing_features.remove('Gender')

                    # 如果仍有特征缺失
                    if missing_features:
                        st.error(f"上传的CSV文件缺少必需的列: {', '.join(missing_features)}")
                        st.info("请确保您的CSV文件包含所有必要的健康指标。")
                else:
                    # 处理数据
                    process_btn = st.button("处理批量数据")

                    if process_btn:
                        # 预处理数据
                        # 处理性别列（如果是文本格式）
                        if 'Gender' in df.columns and df['Gender'].dtype == object:
                            df['Gender'] = df['Gender'].map({'M': 1, 'F': 0, 'Male': 1, 'Female': 0})

                        # 确保所有列名匹配模型特征
                        for col in features:
                            if col not in df.columns:
                                if col.lower() in [c.lower() for c in df.columns]:
                                    match = [c for c in df.columns if c.lower() == col.lower()][0]
                                    df[col] = df[match]  # 使用匹配的列名

                        # 选择所需特征并转换数据类型
                        input_data = df[features].copy()
                        for col in input_data.columns:
                            if input_data[col].dtype == object:
                                input_data[col] = pd.to_numeric(input_data[col], errors='coerce')  # 将字符串转换为数值

                        # 标准化数据
                        input_scaled = scaler.transform(input_data)

                        # 预测
                        predictions = model.predict(input_scaled)  # 预测类别
                        prediction_probs = model.predict_proba(input_scaled)[:, 1]  # 预测概率

                        # 将预测结果添加到数据框
                        results_df = df.copy()
                        results_df['糖尿病风险'] = predictions
                        results_df['风险概率'] = prediction_probs

                        # 添加风险级别
                        def get_risk_level(prob):
                            """
                            根据概率确定风险级别

                            Args:
                                prob (float): 风险概率

                            Returns:
                                str: 风险级别描述
                            """
                            if prob >= 0.7:
                                return "高风险"
                            elif prob >= 0.4:
                                return "中等风险"
                            else:
                                return "低风险"

                        results_df['风险级别'] = results_df['风险概率'].apply(get_risk_level)  # 应用函数确定风险级别

                        # 显示结果
                        st.subheader("批量处理结果")
                        st.dataframe(results_df)  # 显示包含预测结果的数据框

                        # 风险分布可视化
                        st.subheader("风险分布")
                        fig, ax = plt.subplots(figsize=(10, 6))  # 创建图表

                        # 确保风险级别按严重程度排序
                        ordered_levels = ['低风险', '中等风险', '高风险']
                        risk_counts = results_df['风险级别'].value_counts().reindex(ordered_levels).fillna(0)  # 计算各风险级别的数量

                        # 颜色映射
                        colors = {'高风险': 'red', '中等风险': 'orange', '低风险': 'green'}
                        bar_colors = [colors[x] for x in risk_counts.index]  # 为每个风险级别分配颜色

                        # 创建条形图
                        bars = ax.bar(risk_counts.index, risk_counts.values, color=bar_colors)

                        # 为每个条形添加数值标签
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
                                    f'{int(height)}',  # 显示数量
                                    ha='center', va='bottom', fontsize=11, fontweight='bold')

                        ax.set_title('糖尿病风险级别分布', fontsize=14)  # 设置标题
                        ax.set_ylabel('患者数量', fontsize=12)  # 设置y轴标签
                        ax.set_ylim(0, max(risk_counts.values) * 1.15)  # 设置y轴范围，留出空间显示标签
                        ax.grid(axis='y', linestyle='--', alpha=0.3)  # 添加水平网格线
                        ax.spines['top'].set_visible(False)  # 隐藏顶部边框
                        ax.spines['right'].set_visible(False)  # 隐藏右侧边框
                        st.pyplot(fig)  # 显示图表

                        # 添加文本摘要
                        st.markdown(f"""
                        **风险分布摘要:**
                        - **高风险**: {int(risk_counts.get('高风险', 0))} 名患者 ({(risk_counts.get('高风险', 0) / risk_counts.sum() * 100):.1f}%)
                        - **中等风险**: {int(risk_counts.get('中等风险', 0))} 名患者 ({(risk_counts.get('中等风险', 0) / risk_counts.sum() * 100):.1f}%)
                        - **低风险**: {int(risk_counts.get('低风险', 0))} 名患者 ({(risk_counts.get('低风险', 0) / risk_counts.sum() * 100):.1f}%)
                        """)

            except Exception as e:
                st.error(f"处理文件时出错: {e}")  # 显示错误信息
                st.info("请检查您的CSV文件格式并重试。")  # 提供解决建议

# 模型信息标签页
with tab2:
    st.header("模型信息")

    # 添加CSS样式
    st.markdown("""
        <style>
        .card {
            border-radius: 5px;
            background-color: #f8f9fa;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .header {
            color: #2c3e50;
            margin-bottom: 15px;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)  # 使用HTML添加自定义CSS样式

    # 主要模型描述卡片
    st.markdown("""
        <div class="card">
            <h3 class="header">关于预测模型</h3>
            <p>本系统使用多种机器学习算法通过分析各种健康指标来预测糖尿病风险。我们训练并比较了三种不同的模型：</p>
            <ul>
                <li><strong>逻辑回归</strong>：一种用于二元分类的基线模型</li>
                <li><strong>随机森林</strong>：一种构建多个决策树的集成方法</li>
                <li><strong>XGBoost</strong>：一种高级梯度提升算法</li>
            </ul>
            <p>使用多种评估指标对模型进行评估以确保可靠性能，并选择最佳表现的模型进行部署。</p>
        </div>
        """, unsafe_allow_html=True)  # 使用HTML创建卡片样式的模型描述

    # 分为两列布局
    col1, col2 = st.columns(2)

    with col1:
        # 类别分布可视化
        st.markdown('<h3 class="header">数据集类别分布</h3>', unsafe_allow_html=True)
        st.markdown("""
            此可视化展示了我们训练数据集中类别的分布。
            我们的模型被训练用于区分：
            - 非糖尿病患者（类别0）
            - 糖尿病前期和糖尿病患者（类别1）
            """)

        # 尝试加载类别分布图像
        try:
            if os.path.exists('class_distribution.png'):
                img = Image.open('class_distribution.png')
                st.image(img, caption="训练数据中糖尿病病例的分布", use_container_width=True)
        except Exception as e:
            st.error(f"无法加载类别分布图像: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

        # 混淆矩阵可视化
        st.markdown('<h3 class="header">模型性能：混淆矩阵</h3>', unsafe_allow_html=True)
        st.markdown("""
            混淆矩阵展示了我们的模型如何对患者进行分类：
            - **真阳性**：正确识别的糖尿病患者
            - **真阴性**：正确识别的非糖尿病患者
            - **假阳性**：被错误分类为糖尿病的非糖尿病患者
            - **假阴性**：被错误分类为非糖尿病的糖尿病患者
            """)

        # 尝试加载混淆矩阵图像
        try:
            if os.path.exists('confusion_matrix.png'):
                img = Image.open('confusion_matrix.png')
                st.image(img, caption="模型预测的混淆矩阵", use_container_width=True)
        except Exception as e:
            st.error(f"无法加载混淆矩阵图像: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # 特征重要性可视化
        st.markdown('<h3 class="header">特征重要性分析</h3>', unsafe_allow_html=True)
        st.markdown("""
            此图表显示了哪些健康指标对预测糖尿病风险最重要。
            更高的值表示对模型预测的影响更大。

            关键影响因素包括：
            - HbA1c（糖化血红蛋白）：平均血糖水平的度量
            - BMI（体重指数）：基于身高和体重的体脂测量
            - 其他血液标志物如胆固醇水平和肾功能标志物
            """)

        # 尝试加载特征重要性图像
        try:
            if os.path.exists('feature_importance.png'):
                img = Image.open('feature_importance.png')
                st.image(img, caption="特征重要性分析", use_container_width=True)
        except Exception as e:
            st.error(f"无法加载特征重要性图像: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

        # 相关性热图可视化
        st.markdown('<h3 class="header">特征相关性分析</h3>', unsafe_allow_html=True)
        st.markdown("""
            相关性热图揭示了不同健康指标之间的关系：
            - 正相关（蓝色）：两个指标趋向于一起增加
            - 负相关（红色）：当一个指标增加时，另一个趋向于减少
            - 接近±1的强相关：指标之间更强的关系

            理解这些关系有助于识别冗余信息和潜在的风险因素组合。
            """)

        # 尝试加载相关性热图图像
        try:
            if os.path.exists('correlation_heatmap.png'):
                img = Image.open('correlation_heatmap.png')
                st.image(img, caption="特征相关性热图", use_container_width=True)
        except Exception as e:
            st.error(f"无法加载相关性热图图像: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # 显示评估指标解释
    st.markdown("""
    ### 模型评估指标

    使用多种指标对模型进行评估以确保可靠性能：

    - **准确率**：正确预测的总比例
    - **精确率**：在预测为阳性的样本中真正为阳性的比例（最小化误报）
    - **召回率**：正确识别的真阳性比例（最小化漏报）
    - **F1分数**：精确率和召回率的调和平均值
    - **AUC-ROC**：衡量模型区分类别的能力
    """)

# 数据可视化标签页实现
with tab3:
    st.header("数据可视化")
    st.markdown("本节展示糖尿病相关健康指标的分布和关系。")

    st.info("请上传CSV数据集生成可视化图表。")

    # CSV文件上传器
    uploaded_file = st.file_uploader("上传CSV文件进行可视化", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)  # 读取上传的CSV文件

            # 检查上传的CSV中是否存在所需列
            required_cols = ['HbA1c', 'CLASS']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"上传的CSV文件缺少所需列: {', '.join(missing_cols)}")
            else:
                # 处理CLASS列（如果是字符串类型）
                if 'CLASS' in df.columns and df['CLASS'].dtype == object:
                    df['CLASS'] = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 1})  # 将分类编码为数值

                # 创建HbA1c分布可视化
                st.subheader("HbA1c分布")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=df, x='HbA1c', hue='CLASS', palette=['green', 'red'], kde=True, ax=ax)
                # 使用seaborn创建直方图，按CLASS着色，并添加核密度估计曲线
                ax.set_title('HbA1c分布')
                ax.set_xlabel('HbA1c')
                ax.set_ylabel('频率')
                st.pyplot(fig)  # 显示图表

                # 从上传的数据创建相关性热图
                st.subheader("特征相关性分析")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # 获取所有数值列
                corr = df[numeric_cols].corr()  # 计算相关系数矩阵
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)  # 创建热图
                ax.set_title('特征相关性热图')
                st.pyplot(fig)  # 显示图表

        except Exception as e:
            st.error(f"处理文件时出错: {e}")  # 显示错误信息

# 页脚
st.markdown("---")  # 添加分隔线