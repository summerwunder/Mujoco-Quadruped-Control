# Quadruped Controller

基于 MuJoCo 的四足机器人控制框架，支持 NMPC 步态控制。

## Demo

![walk_mpc](docs/walk_mpc.gif)

## 功能特性

- MPC 梯度优化控制器
- 摆动轨迹生成
- 全身控制接口
- 可视化调试工具

## 安装

```bash
pip install -e .
```

## 运行示例

```bash
cd simulation
python stay_demo.py
```

## 项目结构

```
quadruped_ctrl/
├── assets/          # 机器人模型文件
├── config/          # 配置文件
├── controllers/     # 控制器实现
├── interface/       # 参考接口
├── planning/        # 步态规划
└── utils/           # 工具函数
```

## 许可证

Apache2.0 License