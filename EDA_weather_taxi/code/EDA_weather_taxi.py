import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import folium
from folium.plugins import HeatMap
import logging
from pathlib import Path
import requests
import json

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/data_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("taxi_eda_pipeline")

# 全局配置参数
CONFIG = {
    "data": {
        "raw_dir": "../data/raw",
        "processed_dir": "../data/processed",
        "visualization_dir": "../visualizations",
        "download": {
            "taxi_base_url": "https://d37ci6vzurychx.cloudfront.net/trip-data",
            "fhv_base_url": "https://d37ci6vzurychx.cloudfront.net/trip-data",
            "weather_url": "https://www.ncei.noaa.gov/access/services/data/v1",
            "weather_params": {
                "dataset": "daily-summaries",
                "stations": "USW00094728",
                "start_date": "2016-01-01",
                "end_date": "2016-12-31",
                "format": "csv",
                "units": "standard",
                "dataTypes": "PRCP,SNOW,TMAX,TMIN,TAVG"
            }
        },
        "priority_months": ["2016-01", "2016-04", "2016-07", "2016-10"]
    },
    "eda": {
        "sample_size": 100000,
        "visualization": {
            "dpi": 300,
            "figsize": (20, 15),
            "formats": ["png", "pdf"]
        }
    }
}


def setup_directories():
    """创建必要目录结构"""
    dirs = [
        CONFIG["data"]["raw_dir"],
        CONFIG["data"]["processed_dir"],
        CONFIG["data"]["visualization_dir"]
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info("目录结构初始化完成")


def download_data():
    """数据下载模块"""
    logger.info("开始数据下载流程")

    # 下载出租车数据
    for month in CONFIG["data"]["priority_months"]:
        taxi_url = f"{CONFIG['data']['download']['taxi_base_url']}/yellow_tripdata_{month}.parquet"
        file_path = os.path.join(CONFIG["data"]["raw_dir"], f"yellow_{month}.parquet")
        if not os.path.exists(file_path):
            logger.info(f"下载出租车数据: {month}")
            try:
                response = requests.get(taxi_url, stream=True)
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"成功下载: {file_path}")
            except Exception as e:
                logger.error(f"下载失败: {taxi_url} - {str(e)}")
        else:
            logger.warning(f"文件已存在: {file_path}")

    # 下载天气数据
    weather_file = os.path.join(CONFIG["data"]["raw_dir"], "nyc_weather_2016.csv")
    if not os.path.exists(weather_file):
        logger.info("开始下载天气数据")
        try:
            response = requests.get(
                CONFIG["data"]["download"]["weather_url"],
                params=CONFIG["data"]["download"]["weather_params"]
            )
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data['results'])
            df.to_csv(weather_file, index=False)
            logger.info(f"天气数据保存至: {weather_file}")
        except Exception as e:
            logger.error(f"天气数据下载失败: {str(e)}")
    else:
        logger.warning("天气数据已存在")


def process_data():
    """数据清洗模块"""
    logger.info("启动数据清洗流程")

    for month in CONFIG["data"]["priority_months"]:
        # 加载原始数据
        raw_taxi_path = os.path.join(CONFIG["data"]["raw_dir"], f"yellow_{month}.parquet")
        try:
            df = pd.read_parquet(raw_taxi_path)
        except Exception as e:
            logger.error(f"读取原始数据失败: {str(e)}")
            continue

        # 执行清洗步骤
        try:
            # 时间字段处理
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
            df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

            # 计算行程时间
            df['trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds()

            # 空间数据验证
            df = df[
                (df['pickup_longitude'].between(-74.05, -73.75)) &
                (df['pickup_latitude'].between(40.55, 40.95)) &
                (df['dropoff_longitude'].between(-74.05, -73.75)) &
                (df['dropoff_latitude'].between(40.55, 40.95))
                ]

            # 异常值处理
            df = df[
                (df['trip_distance'] > 0.5) &
                (df['trip_distance'] < 50) &
                (df['fare_amount'] >= 2.5) &
                (df['fare_amount'] <= 1000)
                ]

            # 保存清洗结果
            clean_path = os.path.join(CONFIG["data"]["processed_dir"], f"cleaned_taxi_{month}.parquet")
            df.to_parquet(clean_path)
            logger.info(f"保存清洗数据: {clean_path}")
        except Exception as e:
            logger.error(f"数据清洗失败: {str(e)}")


def analyze_time_distribution(df, month):
    """增强时间分析模块"""
    try:
        # 基础时间分析
        df['date'] = df['pickup_datetime'].dt.date
        daily_counts = df.groupby('date').size()

        # 移动平均分析
        rolling_avg = daily_counts.rolling(window=7).mean()

        # 可视化增强
        fig, ax = plt.subplots(figsize=CONFIG["eda"]["visualization"]["figsize"])
        ax.plot(daily_counts.index, daily_counts.values, label='原始数据')
        ax.plot(rolling_avg.index, rolling_avg.values,
                label=f'{7}日滑动平均', linestyle='--')
        ax.set_title(f"{month} 出租车需求趋势分析")
        plt.savefig(os.path.join(CONFIG["data"]["visualization_dir"],
                                 f"time_trend_{month}.pdf"),
                    format=CONFIG["eda"]["visualization"]["formats"])
    except Exception as e:
        logger.error(f"时间分析失败: {str(e)}")


def analyze_hotspot_areas(df, month):
    """空间热点分析"""
    try:
        # 采样优化
        sample_df = df.sample(min(10000, len(df)))

        # 密度热力图
        plt.figure(figsize=(12, 10))
        sns.kdeplot(
            x=sample_df['pickup_longitude'],
            y=sample_df['pickup_latitude'],
            cmap='hot',
            fill=True,
            thresh=0.05
        )
        plt.title('出租车上车点密度热力图')
        plt.savefig(os.path.join(CONFIG["data"]["visualization_dir"],
                                 f"pickup_heatmap_{month}.png"))
    except Exception as e:
        logger.error(f"热点分析失败: {str(e)}")


def analyze_weather_impact(df, month):
    """天气影响分析"""
    try:
        # 合并天气数据
        weather_df = pd.read_csv(os.path.join(CONFIG["data"]["raw_dir"], "nyc_weather_2016.csv"))
        merged_df = pd.merge(
            df.groupby('pickup_datetime').size().reset_index(name='trips'),
            weather_df,
            left_on='pickup_datetime',
            right_on='DATE',
            how='left'
        )

        # 温度影响可视化
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=merged_df,
            x='TAVG',
            y='trips',
            hue='PRCP',
            size='SNOW',
            alpha=0.6
        )
        plt.title('天气因素对出租车需求的影响')
        plt.savefig(os.path.join(CONFIG["data"]["visualization_dir"],
                                 f"weather_impact_{month}.png"))
    except Exception as e:
        logger.error(f"天气分析失败: {str(e)}")


def main():
    """主控制流程"""
    try:
        setup_directories()
        download_data()
        process_data()

        # 加载所有清洗后的数据
        all_data = {}
        for month in CONFIG["data"]["priority_months"]:
            file_path = os.path.join(CONFIG["data"]["processed_dir"], f"cleaned_taxi_{month}.parquet")
            all_data[month] = pd.read_parquet(file_path)

        # 执行分析模块
        for month, df in all_data.items():
            logger.info(f"分析 {month} 数据")
            analyze_time_distribution(df, month)
            analyze_hotspot_areas(df, month)
            analyze_weather_impact(df, month)

        logger.info("数据处理流水线执行完成")
    except Exception as e:
        logger.error(f"流水线执行失败: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()