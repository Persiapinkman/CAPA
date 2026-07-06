---
name: video-frame-extract
description: 从视频文件中自动提取帧图像。支持多种视频格式（mp4, avi, mov, mkv等），自动检测并使用GPU加速（NVIDIA CUDA或Intel QSV），支持多线程并发处理。支持三种输入方式：单个视频文件路径、视频文件所在目录、包含视频路径列表的txt文件。使用ffmpeg进行视频处理，自动回退到CPU模式。适用于视频分析、帧采样、视频预处理等任务。
---

# Video Frame Extract

## 概述

本技能提供从视频文件中提取帧图像的功能，支持批量处理、GPU加速和多线程并发。使用ffmpeg作为底层处理引擎，自动检测硬件加速能力。

## 快速开始

使用 `scripts/extract_frames.py` 脚本提取视频帧：

```bash
# 提取单个视频的帧
python scripts/extract_frames.py video.mp4

# 提取目录中所有视频的帧
python scripts/extract_frames.py /path/to/videos/

# 从txt文件读取视频列表并提取
python scripts/extract_frames.py video_list.txt
```

## 核心功能

### 支持的输入方式

1. **单个视频文件**: 直接提供视频文件路径
2. **视频目录**: 自动递归查找目录中的所有视频文件
3. **txt文件列表**: txt文件中每行一个视频路径（支持注释行，以#开头）

### 自动硬件加速

- **GPU检测**: 自动检测NVIDIA GPU（CUDA）或Intel GPU（QSV）
- **智能回退**: 如果GPU不可用，自动使用CPU处理
- **强制模式**: 可通过 `--gpu` 或 `--no-gpu` 参数强制指定

### 多线程并发

- 自动检测CPU核心数作为默认线程数
- 支持通过 `--threads` 参数自定义线程数
- 批量处理时自动并发处理多个视频

### 支持的视频格式

mp4, avi, mov, mkv, flv, wmv, webm, m4v, 3gp, ts, mts

## 工作流程

1. **检查环境**: 验证ffmpeg是否安装，检测GPU可用性
2. **收集视频**: 根据输入类型（文件/目录/txt）收集所有视频文件
3. **预估帧数**: 使用ffprobe预估总帧数（用于进度显示）
4. **并发处理**: 使用多线程并发处理多个视频
5. **帧提取**: 对每个视频使用ffmpeg提取帧
6. **输出结果**: 将提取的帧保存到指定目录

## 详细文档

**完整的使用指南、参数说明和示例请参考**: [使用指南](references/usage.md)

该文档包含：
- 详细的使用方法和示例
- 完整的参数参考表
- 输入/输出结构说明
- 性能优化建议
- 常见问题解答

## 依赖要求

- **ffmpeg**: 必须安装ffmpeg和ffprobe
  - 安装方法: https://ffmpeg.org/download.html
  - 验证安装: `ffmpeg -version`

- **GPU加速（可选）**:
  - NVIDIA GPU: 需要安装NVIDIA驱动和CUDA
  - Intel GPU: 需要支持QSV的Intel显卡

## Resources

### scripts/

- `extract_frames.py`: 主要的视频帧提取脚本，支持所有核心功能
- `logging_config.py`: 日志配置模块，支持工业级日志管理

### references/

- `usage.md`: 完整的使用指南，包含详细的使用方法、参数说明、示例和最佳实践
