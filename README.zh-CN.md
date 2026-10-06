<p align="center">
  <img src="ui/public/splash-art.png" alt="VTM Spark" width="100%">
</p>

<h1 align="center">VTM Spark</h1>

<p align="center">
  <a href="README.md">English</a> | <a href="README.ja.md">日本語</a> | <b>简体中文</b>
</p>

<p align="center">
  <b>一张动漫图片，变身实时 VTuber。</b><br>
  用摄像头或 iPhone 驱动角色，由 AI 模型绘制每一帧，OBS / Discord / Zoom 会把它当作普通摄像头识别。
</p>

<p align="center">
  <img alt="Windows 10/11" src="https://img.shields.io/badge/Windows-10%20%7C%2011-0078D6">
  <img alt="NVIDIA RTX 30/40/50" src="https://img.shields.io/badge/GPU-NVIDIA%20RTX%2030%20%7C%2040%20%7C%2050-76B900">
  <a href="#许可证"><img alt="许可证：Apache 2.0，另含第三方模型许可证" src="https://img.shields.io/badge/license-Apache%202.0%20%2B%20third--party%20models-blue"></a>
</p>

---

## 效果展示

<table>
  <tr>
    <th>你只需提供一张静态图</th>
    <th>VTM Spark 让它动起来</th>
  </tr>
  <tr>
    <td align="center"><img src="character-blueprint/character-blueprint.png" alt="输入的静态图：绿色背景上的胸部以上动漫角色" width="360"></td>
    <td align="center"><img src="docs/media/demo-live.gif" alt="同一个角色在转头、点头、眨眼和说话，由 VTM Spark 绘制" width="360"></td>
  </tr>
</table>

<sub>右侧的每一帧都是 VTM Spark 模型根据左侧那一张图片绘制的。驱动它的是与实时会话完全相同的追踪流程（转头、点头与歪头、眨眼、视线方向、口型），只是输入换成了脚本生成的 iPhone 式动作，而不是真人面部。眨眼使用的是在 Track Lab 中为该角色制作的 Eye closed（闭眼）形状。</sub>

<!-- 实时演示位：把一段真实会话的 .mp4 拖进 GitHub 的 README 编辑器，然后把它生成的链接粘贴到这里。 -->

## 功能特点

- **输入一张图，输出会动的角色。** 无需绑骨，也无需 Live2D 模型。只要一张角色的胸部以上图片即可。
- **实时追踪你的动作**：支持任意摄像头（OpenSeeFace）或运行 iFacialMocap 的 iPhone，可追踪转头与歪头、眨眼、视线方向、嘴巴和眉毛。
- **由 AI 模型绘制每一帧**（基于关键点驱动的 DiT），在你的 NVIDIA 显卡上实时运行。
- **以名为 "VTM Spark" 的摄像头出现**在 OBS、Discord、Zoom 以及任何支持摄像头的软件中。
- **每个角色就是一个 `.vtm` 文件**，方便保存、备份和分享。
- **Track Lab** 可以按角色分别调整口型、眨眼，以及头部的活动范围。

## 工作原理

<p align="center"><img src="docs/media/pipeline.svg" alt="摄像头或 iPhone，经过 Track Lab，生成 37 个关键点，输入 VTM-1.5.1 DiT（同时输入你的角色图片），再经过 SD VAE，最终输出到 VTM Spark 摄像头" width="100%"></p>

你的面部会被转换成 37 个关键点（脸部轮廓、眉毛、眼睛、虹膜、鼻子、嘴巴、上半身）。模型 `VTM-1.5.1` 根据那一张参考图片，一步就能以该姿势重新绘制你的角色，再由 VAE 将结果转换为 768 × 768 的画面。

## 系统要求

- Windows 10 或 11
- NVIDIA GeForce RTX 30、40 或 50 系列显卡，并安装最新驱动（暂不支持 AMD、macOS 和 Linux）
- 首次运行需要联网（工具和模型权重会自动下载；**无需 API 密钥**）
- 一个摄像头，或一部安装了 iFacialMocap 应用的 iPhone

## 安装

1. **下载**本仓库（点击绿色的 **Code** 按钮，再点 **Download ZIP**（下载 ZIP），然后解压），或者直接克隆。
2. **双击 `install.bat`。** 它会在应用文件夹内完成全部配置，然后打印安装摘要：
   - Python（通过 [uv](https://docs.astral.sh/uv/)）安装在 `.venv-build`，如果你已有 Python 3.10+ 则会直接使用
   - 便携版 Node.js 安装在 `.tools\node`，仅用于构建界面（不会影响你自己安装的 Node）
   - WebView2，微软为应用窗口提供的运行时，仅在缺失时安装（Windows 11 已自带）
   - AI 模型、Track Lab 追踪器以及虚拟摄像头
3. **双击 `run.exe`**（帽子图标）打开主界面。

> **会弹出一次 Windows 提示，这是正常的。** 添加虚拟摄像头需要一次管理员权限，因此 Windows 会弹出带有 VTM Spark 标志和 "VTM Spark Camera Setup" 字样的授权请求。请点击 **是**（Yes）。如果跳过，其他功能依然正常，之后也可以在应用中再添加摄像头。

你可以随时再次运行 `install.bat`。它会修复缺失的部分，并跳过已经完成的步骤。

要获取最新版本，请双击 `update.bat`。它会先检查 GitHub，只有在有新版本时才会关闭 VTM Spark。它通过 git 拉取更新（如果没有安装 git，则从 GitHub 下载），然后自动完成安装：新的依赖包、重新构建的界面、新模型。如果安装中途被打断，再次运行 `update.bat` 即可从中断处继续。你的角色和模型都会保留。

要卸载 VTM Spark，请双击旁边的 `uninstall.bat`。它会移除虚拟摄像头（需要一次管理员授权）、Python 环境、已下载的模型和缓存，并在删除你的角色前征求确认。最后删除整个文件夹即可。

## 开始你的第一次直播

1. 用 `run.exe` **打开主界面**，并选择语言（之后可以在 Settings（设置）中更改）。
2. **添加角色。** 点击空白的角色预览，然后点击 **Create**（创建），选择你的图片（参见下方的[制作角色图片](#制作角色图片)）。准备就绪后会打开 **Fit character**（适配角色）：给角色起个名字，然后点击 **Save**（保存）。
3. 在追踪区域**选择追踪方式**：
   - **Camera**（摄像头）：从列表中选择你的摄像头。
   - **iFacialMocap**：使用 iPhone（参见 [iPhone 追踪](#iphone-追踪)）。
4. 点击 **Start tracking**（开始追踪），目视正前方，面部放松、嘴巴闭合，然后点击 **Calibrate**（校准）。这一步会设定你的静止姿势。
5. 点击 **Start stream**（开始推流）。首次使用时会把模型加载到显卡上。
6. 点击 **Start cam**（启动摄像头），将角色输出到虚拟摄像头。
7. 在 **OBS / Discord / Zoom** 中选择名为 **VTM Spark** 的摄像头。在 OBS 中，添加一个*视频采集设备*（Video Capture Device）源并选择 **VTM Spark**。背景是绿色的，加一个*色度键*（Chroma Key）滤镜即可变为透明。

需要暂停一下时，**Pause**（暂停）会保持最后一帧画面。

## 制作角色图片

当图片的构图与 [`character-blueprint/character-blueprint.png`](character-blueprint/character-blueprint.png) 完全一致时，模型效果最佳：

| 请这样做 | 请避免 |
|---|---|
| 正方形，768 × 768 | 竖版或横版图片 |
| 纯绿色背景（`#00FF00`） | 室内场景、渐变、背景上的阴影 |
| 胸部以上，角色居中，正对镜头，下巴保持水平 | 头部倾斜或转向 |
| 双眼睁开，闭嘴微笑 | 张嘴、眨单眼、画面中出现手或道具 |
| 只有一个角色，没有文字或标志 | 水印、界面元素、其他人物 |

**最简单的方法：** 把两张图片——蓝图和你自己的角色——连同 [`character-blueprint/PROMPT.txt`](character-blueprint/PROMPT.txt) 中的提示词一起交给图像 AI。蓝图只用于确定构图和姿势，角色的外观来自你自己的图片。

## iPhone 追踪

1. 在支持 Face ID 的 iPhone 上安装 **iFacialMocap**。
2. 让 iPhone 和电脑连接到**同一个 Wi-Fi**。
3. 在主界面中选择 **iFacialMocap**。界面会显示 **This PC**（本机）的 IP 地址（点击即可复制）和端口，默认端口为 **49983**。
4. 在 iFacialMocap 中把该 IP 地址设为目标地址，然后开始发送。
5. 和平常一样，先点击 **Start tracking**，再点击 **Calibrate**。

收不到数据？请在 Windows 防火墙中允许 Python 访问**专用**（Private）网络，并确认 iPhone 发送的目标地址与主界面显示的一致。

## 角色（`.vtm` 文件）

一个角色就是 `characters` 文件夹中的一个 `.vtm` 文件。它包含图片以及为其适配的全部数据：头发遮罩、身体关键点、动作限制以及嘴部/眼部形状。

- **添加别人的角色：** 点击 **Import .vtm**（导入 .vtm），或者把 `.vtm` 文件拖到角色预览或角色库上。
- **分享你的角色：** 右键点击角色，选择 **Show in folder**（在文件夹中显示），然后发送那个 `.vtm` 文件。
- **编辑、重命名或删除：** 右键点击角色即可。
- **"Repair … Blend shapes do not match the current plan"（修复……混合形状与当前方案不匹配）：** 说明你在创建这个角色之后修改过 Track Lab 中的形状。点击 **Repair**（修复）会把当前的形状复制到该角色中。从别人那里导入的角色会保留各自的形状，不会出现此提示。

## Track Lab（精细调整）

主界面会在后台运行 Track Lab 的追踪器，所以仅仅直播的话无需打开它。想要精细调整时再自己打开：双击 `track_lab\start.bat`，然后访问 <http://127.0.0.1:5174>。

- **Blend**（混合）**面板：** 由你的面部驱动的各种形状。包括 Rest（静止）、Smile（微笑）、Sad（悲伤）、元音 A / I / U / E，以及用于眨眼的 **Eye open / Eye closed**（睁眼 / 闭眼）。点击 **›** 编辑形状，拖动其中的点，然后点击 **Apply**（应用）。形状之间的混合条可以添加中间过渡点，让动作更顺滑。
- **Limiters**（限制器）**：** 头部和身体在停止前最多可以移动、转动和点头的幅度，以及眼睛和尺寸的限制。**Fit**（适配）会根据图片自动设定这些值。
- **Tracking**（追踪）**：** 目视正前方、嘴巴闭合时点击 **Set Rest**（设为静止）。Response（响应）、Smooth（平滑）、Mouth（嘴部）和 Gaze（视线）滑块用于设置角色跟随你的方式。
- **Gen**（生成）会用当前的点绘制一帧角色画面，方便检查形状效果。

编辑形状时需要先停止追踪。

## 值得了解的设置

| 设置 | 作用 |
|---|---|
| **Smooth**（平滑） | 让头部动作更柔和。0 为原始数据；数值越高越飘逸。眨眼始终即时响应。 |
| **Mouth**（嘴部） | 角色嘴巴跟随你嘴巴的强度。 |
| **Mirror**（镜像） | 翻转视线和转头的方向。 |
| **Max FPS**（最大帧率） | 限制帧率，为游戏或 OBS 留出显卡资源。 |
| **Batch**（批量） | 每一步绘制的帧数。帧率更高，但会占用更多显存，并带来少许延迟。 |
| **Inbetweens**（中间帧） | 在绘制的帧之间插入额外画面，让动作更顺滑。 |
| **Compile**（编译） | 为你的显卡提速。首次构建可能需要一分钟左右。 |
| **GPU**（显卡） | 有多张 NVIDIA 显卡时，选择使用哪一张（重启后生效）。 |

## 故障排除

| 问题 | 解决方法 |
|---|---|
| `install.bat` 中途停止 | 检查网络连接。代理或防火墙拦截了 nodejs.org、github.com 或 pypi.org，或者杀毒软件锁定了 `.venv-build`，都可能导致中断。处理后再次运行即可。 |
| "No NVIDIA GPU found"（未找到 NVIDIA 显卡） | VTM Spark 需要 NVIDIA RTX 显卡和最新驱动。 |
| OBS/Discord 中没有 **VTM Spark** 摄像头或画面全黑 | 在摄像头安装提示中点击 **是**（Yes）。如果你移动过应用文件夹，请再次运行 `install.bat`，让摄像头指向新位置。 |
| iPhone："No packets yet"（尚未收到数据包） | 确保两台设备连接同一个 Wi-Fi，在 Windows 防火墙中允许 Python 访问专用网络，并发送到主界面显示的 IP 和端口。 |
| "Port 8780 is already in use"（端口 8780 已被占用） | 有其他程序占用了 Track Lab 的端口。关闭该程序后重新启动。 |
| "No face while calibrating"（校准时未检测到面部） | 在光线充足的环境下正对摄像头，然后再次点击 **Calibrate**。 |
| 应用提示正在运行，但没有窗口 | 当它询问时，允许它清理残留进程，然后重新打开 `run.exe`。 |

## 模型与许可证

`install.bat` 会自动为你下载模型，无需手动获取。

<p align="center"><img src="docs/media/model-downloads.svg" alt="按大小排列的模型下载：头发分割 432 MB，生成器 360 MB，SD VAE 335 MB，动漫面部关键点 39 MB，身体关键点 23 MB，OpenSeeFace 21 MB，tiny VAE 9.8 MB，虹膜 6.4 MB，动漫面部框 6.2 MB，身体追踪 5.8 MB" width="100%"></p>

| 模型 | 作用 | 大小 | 本地路径 | 许可证 | 商业用途 |
|---|---|---|---|---|---|
| `VTM-1.5.1.pt` | 绘制你的角色（DiT） | 360 MB | `models/dit/` | Apache-2.0（我们自有） | 可以 |
| `vtm-fast-decoder.pt` | 为实时推流快速解码画面 | 3.9 MB | `models/decoder/` | 我们自有，蒸馏自 `cqyan/hybrid-sd-tinyvae`，其许可证未声明（[声明文件](THIRD_PARTY_NOTICES.md)） | 不明确 |
| `animeseg_hair3.pt` | 分割头发区域，让头发跟随头部 | 432 MB | `models/trackers/` | 我们基于 Meta Mask2Former 的微调版本，**CC BY-NC 4.0** | **不可以** |
| `dwpose_v2.pt` | 识别图片中的身体关键点 | 23 MB | `models/trackers/` | 我们基于 Ultralytics YOLO-pose 的微调版本，**AGPL-3.0** | 需遵守 AGPL 条款 |
| `iris_pose.pt` | 识别图片中的虹膜 / 瞳孔 | 6.4 MB | `models/trackers/` | 我们基于 Ultralytics YOLO-pose 的微调版本，**AGPL-3.0** | 需遵守 AGPL 条款 |
| `pose_landmarker_lite.task` | 实时身体追踪（MediaPipe） | 5.8 MB | `models/trackers/` | Apache-2.0（Google） | 可以 |
| OpenSeeFace（5 个文件） | 摄像头面部追踪 | 21 MB | `vendor/tools/openseeface/models/` | BSD 2-Clause | 可以 |
| `face_yolov8n.pt` | 动漫面部框 | 6.2 MB | `models/trackers/` | Apache-2.0（[Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer)） | 可以 |
| `mmpose_anime-face_hrnetv2.pth` | 动漫面部关键点 | 39 MB | `models/trackers/` | MIT（[hysts/anime-face-detector](https://github.com/hysts/anime-face-detector/releases/tag/v0.0.1)） | 可以 |
| `stabilityai/sd-vae-ft-mse` | 将模型输出转换为图片 | 335 MB | Hugging Face 缓存 | MIT | 可以 |
| `cqyan/hybrid-sd-tinyvae` | 更快的画面解码 | 9.8 MB | Hugging Face 缓存 | 发布者未声明 | 不明确 |

前七个模型来自我们的 Hugging Face 仓库 [sinBoo1/VTM-Spark](https://huggingface.co/sinBoo1/VTM-Spark)，其余模型直接来自各自的原始发布者。总计约 1.24 GB；Python 和 PyTorch 另行下载。

如果点击 Start 时 DiT 权重缺失，会在启动前自动下载。

## 开发者指南

- `backend/`：应用本体（`python -m backend`），其中 `backend/packaging/` 存放安装和启动脚本
- `ui/`：基于 Vite/React 的主界面
- `track_lab/`：追踪器和 Track Lab 界面（在 `track_lab/` 目录下运行 `python -m backend.pair`）
- `vendor/`：推理、LivePoser、OpenSeeFace 和虚拟摄像头
- `models/`：DiT、追踪器、下载映射表、参考资料
- `characters/`：你的 `.vtm` 角色包（位于应用根目录旁）

测试（可选）：`python -m pytest backend/tests`，以及在 `track_lab/` 目录下运行 `python -m pytest backend harness`。

第三方代码已提交在 `vendor/` 目录下。只有当你在可选的上层 monorepo 中开发、需要刷新 vendor 副本时，才需要给 `backend\packaging\build.ps1` 传入 `-SyncVendor`。

## 许可证

Apache License 2.0 适用于 VTM Spark 源代码以及**我们**的原创 / 微调训练成果（参见 [LICENSE](LICENSE)）。该授权**不**涵盖第三方模型权重，也不适用于之后添加的模型。

我们随附的模型中有两个基于条款更严格的权重，这些条款仍然有效：

- **`animeseg_hair3.pt`**（头发追踪）基于 Meta 的 Mask2Former 权重：**CC BY-NC 4.0，仅限非商业用途**。
- **`iris_pose.pt` 和 `dwpose_v2.pt`** 基于 Ultralytics YOLO-pose 权重：**AGPL-3.0**。

完整的逐文件列表见上方的[模型与许可证](#模型与许可证)表格。

第三方权重沿用其发布者的官方许可证。清单及来源链接：[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。OpenSeeFace 二进制库的许可声明：`vendor/tools/openseeface/Licenses/`。

本文为译文，如与英文版 [README.md](README.md) 或 [LICENSE](LICENSE) 存在出入，以英文版为准。
