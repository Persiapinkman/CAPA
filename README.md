## 仓库简介

`test-skills` 仓库用于存放和迭代各类 **Skill**，把常用能力（如视频处理、文档处理等）沉淀成可复用的技能。[官方skill说明文档](https://github.com/anthropics/skills/blob/main/README.md)

## 已有 Skill 概览

- `skills/skill-creator`：官方的 **Skill Creator** 指南，用于创建和设计新的 skill。
- `skills/video-frame-extract`：基于 ffmpeg 的 **视频抽帧 skill**，支持单视频文件，视频目录，视频路径txt文件等多种输入形式。

## cursor 如何使用当前仓库skill
```
# step1 clone 当前代码仓库
git clone https://gitlab.sz.sensetime.com/xiaokun1/test-skills.git

# step2 创建.cursor目录, 软连接当前的技能，cursor会自动加载对应的技能。
mkdir -p .cursor && cd .cursor && ln -s ../skills skills && cd -
```
其它地方使用skill, 参考[cursor说明](https://cursor.com/cn/docs/context/skills)

## 如何制作一个新的 Skill

### 1. 规划 Skill

- 明确 **单一职责**，避免「大而全」：
  - 例：`pdf-text-extract`（只做 PDF 文本抽取）
  - 例：`image-annotate`（只做图像标注）
- 使用 **kebab-case** 命名（如 `video-frame-extract`）。

### 2. 使用 Skill Creator 初始化骨架

在cursor聊天窗口个中，直接/skill-creator，然后告诉你的诉求，创建什么样的skill,输出放到skills目录下，即可：

```bash
/skill-crator 创建一个图片预览的skill技能，放置到本仓库skills目录下。 
```

会生成如下文件：

- `<skill-name>/SKILL.md`：带 frontmatter 的说明文档，尽量简单
- `<skill-name>/scripts/`：放置可执行脚本代码
- `<skill-name>/references/`：放置长文档、API 说明等
- `<skill-name>/assets/`：放置模板、图片等资源

### 3. 迭代skill的功能
如果初步生成的skill不满足，要求，可以通过cursor反复迭代和完善功能，代码需要具备扩展性、可读性、可靠性，注意日志规范等。


## 如何打包 Skill（生成 .skill）

本仓库提供 `package_skill.py`，用于将某个 Skill 目录打成 `.skill` 包，便于分发和安装。

### 基本用法

在仓库根目录运行,修改run_package.sh目录的参数，进行打包：

```bash
bash run_package.sh

参数含义：python3 util/package_skill.py -h
```


## 如何分发和使用 Skill

### 分发

将生成的 `.skill` 文件上传到团队可访问的位置，例如：

- GitLab 仓库（如 `dist/` 目录）
- GitLab Release 附件
- 内部制品库 / 对象存储 / 共享网盘等

### 在 Cursor 中使用

1. 从上述位置下载对应的 `.skill` 文件到本地。
2. 在 Cursor 中通过「导入 / 安装」功能加载该 `.skill` 文件。
3. 安装成功后，即可在对话中通过对应指令（如 `/video-frame-extract`）或相关自然语言描述触发该 Skill。

