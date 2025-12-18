# 智能炼厂碳足迹数据核算及分析小程序（Streamlit + MySQL + DeepSeek‑V3）

聚焦碳足迹核算、结转与分析。支持 Excel 导入 → 列映射与校验 → 计算 → 可视化 → AI 中文总结与连续对话，并可将结果保存到 MySQL。

## 功能总览
- Excel 导入与标准化：自动猜测列映射 + 手动修正；校验必要列、非负值、引用完整性。
- 结转与核算：线性/迭代/基于 routing 的装置‑物料流因子；多来源默认“平均分摊”；不支持负流。
- 可视化（Plotly）：
  - 单位能耗 CO2e TopN（横向条形）
  - Pareto（柱形 + 累计占比）
  - 流因子 TopN（横向条形）
  - 排放强度散点（可高亮 IQR 异常点）与箱线图（可选）
  - 桑基图（装置间流向：节点=蓝色强度，连线=红色强度）
- AI 总结与连续对话：直连硅基流动 DeepSeek‑V3（/chat/completions），输出中文结构化报告（总体结论/关键发现/优化建议/数据质量与下一步），并基于“结构化摘要 + 报告”持续追问。
- MySQL 持久化：保存 run、单位级结果、流因子与 AI 总结；页面查看历史记录。

---

## Excel 模板（建议列）
必需 sheets：
- materials(material_id, material_name, unit, category)
- fixed_factors(material_id, factor_per_unit)
- units(unit_id, unit_name)
- consumption(unit_id, material_id, amount)
- production(unit_id, material_id, amount)
- carryover(material_id, factor_init)

可选 sheet：
- routing(consumer_unit_id, material_id, source_unit_id, amount, unit)

注意：amount/factor_init 必须非负；material_id 与 unit_id 应在字典表中可查。

---

## 安装与运行
### 1) 新建（或使用）Conda 环境
```
conda create -n CFapp python=3.11 -y
conda run -n CFapp python -m pip install -r virtual_refinery_ai_full_project\requirements.txt
```

### 2) 启动小程序
- 推荐：双击根目录的 `run_ui.bat`（默认环境 CFapp，端口 8501）。
- 指定环境与端口：
```
virtual_refinery_ai_full_project\run_ui.bat CFapp 8501
```
- 纯命令行（Anaconda Prompt）：
```
conda run -n CFapp python -m streamlit run virtual_refinery_ai_full_project\project\ui\app.py --server.port 8501
```

---

## LLM（硅基流动 DeepSeek‑V3）
- 默认配置在 `project/config.py`：
  - LLM_DEFAULT_BASE_URL = https://api.siliconflow.cn/v1
  - LLM_DEFAULT_MODEL = deepseek-ai/DeepSeek-V3
  - LLM_DEFAULT_API_KEY = （将其替换为你的正式密钥）
- 使用：进入“AI 总结”页点击“生成/重新生成总结”；生成后可在“与 AI 继续交流”区域对话。
- 切换到其他 OpenAI 兼容服务：修改 BASE_URL、MODEL 与 API_KEY（接口遵循 /chat/completions 风格）。

---

## MySQL 设置
1) 在 `project/config.py` 配置 `MYSQL = MySQLConfig(...)`。
2) 侧边栏点击“初始化数据库（建库建表）”。
3) 计算完成后在“计算与结果”页点击“保存到 MySQL”。
4) 在“历史记录”页查看最近 runs 并加载详情。

表结构（摘要）：
- import_runs(id, file_name, file_hash, params_json, status, created_at)
- units_results(id, run_id, unit_id, unit_name, total_input_kgco2e, total_output_qty)
- stream_results(id, run_id, stream_id, factor_stream)
- ai_insights(id, run_id, model, content, created_at)

---

## 目录结构
```
virtual_refinery_ai_full_project/
├─ project/
│  ├─ ui/app.py                  # Streamlit 页面（上传/映射/计算/因子/可视化/AI/下载/历史）
│  ├─ services/                  # 导入/求解/报表/AI/存储（含 MySQL）
│  │  ├─ excel_loader.py
│  │  ├─ solver_linear.py | solver_iterative.py | solver_stream.py
│  │  ├─ reporting.py | energy_report.py | ai_report.py | llm_adapter.py
│  │  └─ db/mysql_store.py
│  ├─ utils/viz.py               # Plotly 绘图（TopN、Pareto、散点/箱线、桑基）
│  ├─ models/                    # 数据模型（单位/物料/公司）
│  └─ README.md                  # 本文件
├─ tools/generate_ppt.py         # 生成汇报 PPT 模板（python-pptx）
├─ requirements.txt              # 依赖清单（Python 3.11）
└─ run_ui.bat                    # 一键启动脚本（可传入环境名与端口）
```

---

## 使用提示
- 可视化：在“可视化”页选择 Plotly 主题；TopN 可调；强度散点默认高亮 IQR 异常点；可选显示箱线图并导出“异常清单（Excel）”。
- AI 对话：上下文包含“结构化摘要（Top 装置/流因子/异常）+ 已生成报告 + 历史对话”，回答更稳定；失败显示通用错误。
- Excel 模板：侧边栏可下载示例模板，建议从示例开始。

---

## 故障排查（FAQ）
1) NumPy 报错 “you should not try to import numpy from its source directory …”
   - 确认使用 Conda 环境 CFapp：
     ```
     conda run -n CFapp python -c "import sys, numpy; print(sys.executable); print(numpy.__version__, numpy.__file__)"
     ```
   - 若路径异常或混入用户站点包，执行：
     ```
     conda install -n CFapp -c conda-forge "numpy=1.26.*" "pandas=2.2.*" -y
     ```
2) Plotly 或 PNG 导出失败：`pip install plotly kaleido`
3) LLM 401 鉴权失败：检查 Key 是否有效、是否开通模型权限、base_url 是否正确。
4) 双击 bat 无法启动：
   - 使用命令：`virtual_refinery_ai_full_project\run_ui.bat CFapp 8501`
   - 或在 Anaconda Prompt 中：
     `conda run -n CFapp python -m streamlit run virtual_refinery_ai_full_project\project\ui\app.py`

---

## 约束与默认
- 粒度：年；范畴：S1 + S2；不支持负流；多来源默认“平均分摊”。
- LLM：默认联网启用 DeepSeek‑V3；调用失败给出通用错误提示。

---
