## 环境

- Python 3.11，CPU 即可：`pip install -r requirements.txt`
- 绘图字体：宋体（SimSun）与 Times New Roman（Windows 自带；其他系统需自行安装）
- 推理线程数默认取全部 CPU 核，可用环境变量 `PANGU_THREADS` 指定
- 在代码包根目录依次运行 `python -m pipeline.step01_era5_track` … `python -m pipeline.step09_figures`；step02、step04、step05、step07 可中断后重跑续算

## 数据来源以及存放位置

- 盘古 3 h 模型 `pangu_weather_3.onnx`（https://github.com/198808xc/Pangu-Weather ，Google Drive：https://drive.google.com/file/d/1EdoLlAXqE9iZLt9Ej9i-JW9LTJ9Jtewt ）：`models/pangu_weather_3.onnx`
- IBTrACS v04r01 最近三年表（https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.last3years.list.v04r01.csv ）：`data/ibtracs.last3years.list.v04r01.csv`
- GSHHG 2.3.7 二进制海岸线（https://www.soest.hawaii.edu/pwessel/gshhg/gshhg-bin-2.3.7.zip ）：`data/gshhg-bin-2.3.7.zip`
- ERA5 再分析（Copernicus CDS，https://cds.climate.copernicus.eu ，0.25°，GRIB，每天一个文件，文件名为 `YYYYMMDD.grib`，包含当天所需的全部时次）：
  - 全球气压层（reanalysis-era5-pressure-levels；geopotential、specific_humidity、temperature、u/v_component_of_wind；1000、925、850、700、600、500、400、300、250、200、150、100、50 hPa；2026-07-10 03—21 UTC 与 2026-07-11 00—09 UTC，逐 3 h）：`data/era5/global_upper/`
  - 全球单层（reanalysis-era5-single-levels；mean_sea_level_pressure、10m_u/v_component_of_wind、2m_temperature；时次同上）：`data/era5/global_surface/`
  - 区域气压层（geopotential、temperature、u/v_component_of_wind，层次同全球气压层；area 为 北62、西78、南-2、东182；2026-06-28 03 UTC 至 2026-07-23 03 UTC，逐 3 h）：`data/era5/regional_upper/`
  - 区域单层（mean_sea_level_pressure、10m_u/v_component_of_wind、surface_pressure；区域与时次同区域气压层）：`data/era5/regional_single/`

## 结果

结果图像位于 `./img`（`fig2_2.png` 对应图 2.2，`fig4_1.png`—`fig4_5.png` 对应图 4.1—4.5，`fig5_1.png`—`fig5_3.png` 对应图 5.1—5.3）。中间结果位于 `./output`。

## 程序

- `common/config.py`：路径、个例（台风“巴威”，扰动起报 2026-07-10 15 UTC）、网格、区域与试验参数
- `common/era5.py`：读取 ERA5 GRIB；生成盘古初始场
- `common/tracker.py`：台风中心识别、亚格点定位、线性外推、海岸线交点与首次大陆登陆点判定、登陆误差
- `common/physics.py`：干能量范数、气压层厚度权重、由两时刻 ERA5 场之差构造等能量初始扰动
- `common/pangu.py`：盘古 3 h 模型逐 3 h 迭代预报
- `common/ensemble.py`：扰动成员的初始场构造、预报、路径跟踪与评价区响应
- `common/style.py`：绘图版式、字体与海岸线
- `pipeline/step01_era5_track.py`：生成海岸线多边形；由 IBTrACS 定位起点，诊断 ERA5 参照路径（格点与亚格点）
- `pipeline/step02_control_runs.py`：登陆前 36—6 h 的 11 次控制预报及其路径；7 月 10 日 15 时起报的 72 h 控制预报场
- `pipeline/step03_start_scores.py`：各次起报的盘古与线性外推路径误差、登陆点误差与登陆时间偏差
- `pipeline/step04_library.py`：100 个发现方向与 25 个留出方向的初始扰动
- `pipeline/step05_discovery.py`：200 个扰动成员的 72 h 预报、评价区响应、路径与登陆点
- `pipeline/step06_regions.py`：20 个高响应成员的初始干能量分布，敏感区 RC 与对照区 R0—R8
- `pipeline/step07_holdout.py`：50 个留出成员在不削减及各区域按 4 档比例削减初始误差下的 36 h 预报路径
- `pipeline/step08_improvement.py`：各区域、各削减比例下路径与中心气压误差的预报改进率及 95% 置信区间
- `pipeline/step09_figures.py`：绘制图 2.2、4.1—4.5、5.1—5.3
