# Imervue 插件

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-TW.md">繁體中文</a> ·
  <strong>简体中文</strong> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.pt-BR.md">Português (BR)</a> ·
  <a href="README.ru.md">Русский</a>
</p>

[Imervue](https://github.com/JeffreyChen-s-Utils/Imervue) 的官方插件，这是一款 GPU 加速的图片查看器。

## 安装

在 Imervue 中打开 **Plugins → Download Plugins**，选一个插件并点击 **Download**，然后重启 Imervue（或使用 **Plugins → Reload Plugins**）。下载器会读取本仓库的 `main` 分支，并把每个插件安装到 Imervue 旁边的 `plugins/` 目录（**Plugins → Open Plugin Folder**）。

若要手动安装，把像 `plugins/npr_filters/` 这样的插件目录复制到那个 `plugins/` 目录。安装时会去掉分类层级（`plugins/`、`languages/`）。

需要大型软件包（onnxruntime、rembg、OpenCV……）的插件，会在你第一次使用需要它的功能时提示安装。基于 ONNX 的插件会在自己插件目录里的 `models/` 文件夹中查找模型文件；这些文件不会被下载，所以请把你想用的模型放到那里。针对较新版 Imervue 编写的插件，可能无法在较旧版上加载；若出现这种情况，请更新 Imervue。

## 插件

| 插件 | 功能 | 需求 |
| --- | --- | --- |
| `plugins/ai_background_remover` | 用 rembg（U²-Net）移除图片背景，可单张或批量处理 | rembg、onnxruntime |
| `plugins/ai_colorize` | 用预设色板或 ONNX 模型为黑白照片上色 | ONNX 路径需要 onnxruntime |
| `plugins/ai_denoise` | 双边滤波或 ONNX 神经网络降噪 | ONNX 路径需要 onnxruntime |
| `plugins/ai_motion_deblur` | Wiener 反卷积或 ONNX 运动模糊去除 | ONNX 路径需要 onnxruntime |
| `plugins/ai_object_remove` | 点选物体以选取并将其修补移除；可选用 SAM 点提示 | SAM 需要 onnxruntime |
| `plugins/ai_outpaint` | 扩展画布并填满新增的边界 | — |
| `plugins/ai_portrait_relight` | 方向性人像重新打光，启发式或 ONNX | ONNX 路径需要 onnxruntime |
| `plugins/ai_smart_resize` | 内容感知缩放（接缝裁剪） | — |
| `plugins/ai_style_transfer` | 使用 ONNX 风格模型的快速神经风格迁移 | onnxruntime |
| `plugins/cloud_share` | 把当前图片上传到 WebDAV 或 Imgur（仅限 HTTPS） | — |
| `plugins/npr_filters` | 铅笔素描、油画、水彩与线稿风格 | opencv-python |
| `plugins/object_splitter` | 移除背景并把每个物体各自保存为透明 PNG | rembg、onnxruntime |
| `plugins/png_to_icon` | 把 PNG 转换为多尺寸的 `.png` 与 `.ico` 图标 | — |
| `plugins/portrait_mode` | 模糊检测到的主体背后的背景 | rembg、onnxruntime |
| `plugins/safety_review` | 检测并马赛克露骨区域，附手动编辑器与数据集导出 | 照片用 nudenet + onnxruntime，动漫用 ultralytics + huggingface_hub |
| `plugins/video_source` | 拖动浏览视频并把静态帧提取到浏览器中 | — |
| `languages/spanish_translation` | 在语言菜单中加入西班牙语（Español） | — |

“—”表示该插件只靠 Imervue 的默认依赖即可运行。

## 编写插件

插件 API（钩子、发现、依赖、后台工作、i18n、发布规则）都集中记录在一个地方：Imervue 仓库中的 [`PLUGIN_DEV_GUIDE.md`](https://github.com/JeffreyChen-s-Utils/Imervue/blob/main/PLUGIN_DEV_GUIDE.md)。

插件是在 Imervue 仓库的 `plugins/<name>/` 下开发与测试，再镜像到这里；本仓库本身不含任何代码。请把插件需要的每个文件都直接放在它的目录里：下载器不会抓取子目录。

## 许可证

见 [LICENSE](LICENSE)。
