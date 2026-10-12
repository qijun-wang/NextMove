# MATLAB LiDAR Planning Pipeline

该目录保存从 LiDAR 点云处理到路径规划与路径跟踪的 MATLAB 核心实现，供后续迁移到 ROS 2 使用。

## 处理流程

```text
点云预处理
  -> 障碍物检测
  -> 二维障碍物跟踪
  -> 风险感知局部代价地图
  -> A* 路径规划
  -> Pure Pursuit / MPC 路径跟踪
```

## 文件说明

- `nextmove_processPointCloud.m`：对单帧点云进行 ROI 裁剪、自车区域去除、降采样、去噪和地面分割。
- `detectLidarObstacles.m`：通过欧式聚类检测障碍物，并计算质心、尺寸、方向和包围框。
- `run_obstacle_detection.m`：批量读取预处理后的 PCD 文件并导出逐帧障碍物检测结果。
- `trackObstacles2D.m`：使用最近邻数据关联和常速度卡尔曼滤波跟踪障碍物，同时估计速度、TTC 和风险等级。
- `run_obstacle_tracking.m`：读取检测结果并批量生成连续障碍物轨迹。
- `build_costmap.m`：融合静态点云和动态障碍物，生成软风险代价层与硬碰撞层。
- `run_offline_mpc_pp.m`：使用 A* 生成局部路径，并通过 Pure Pursuit 和有限时域 MPC 完成离线路径跟踪与碰撞检查。

## 运行顺序

```matlab
run_obstacle_detection
run_obstacle_tracking
build_costmap
run_offline_mpc_pp
```

`run_obstacle_detection.m` 默认从同级 `pointcloud_preprocessed` 目录读取 `.pcd` 文件。原始点云可先逐帧调用 `nextmove_processPointCloud` 进行预处理。

主要依赖 MATLAB、Lidar Toolbox、Computer Vision Toolbox 和 Navigation Toolbox。`run_offline_mpc_pp.m` 可选用 Optimization Toolbox 的 `fmincon`；不可用时会回退到内置搜索方法。

当前实现用于离线验证，并非 ROS 2 实时闭环节点。
