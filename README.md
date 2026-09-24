# Imervue Plugins

<p align="center">
  <strong>English</strong> ·
  <a href="README.zh-TW.md">繁體中文</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.pt-BR.md">Português (BR)</a> ·
  <a href="README.ru.md">Русский</a>
</p>

Official plugins for [Imervue](https://github.com/JeffreyChen-s-Utils/Imervue), the GPU-accelerated image viewer.

## Installing

In Imervue, open **Plugins → Download Plugins**, pick a plugin and click **Download**, then restart Imervue (or use **Plugins → Reload Plugins**). The downloader reads the `main` branch of this repository and installs each plugin into the `plugins/` directory next to Imervue (**Plugins → Open Plugin Folder**).

To install by hand, copy a plugin directory such as `plugins/npr_filters/` into that `plugins/` directory. The category level (`plugins/`, `languages/`) is dropped on install.

Plugins that need a heavy package (onnxruntime, rembg, OpenCV, ...) offer to install it the first time you use the feature that needs it. The ONNX-based plugins look for model files in a `models/` folder inside their own plugin directory; those files are not downloaded, so put the model you want to use there. A plugin written against a newer Imervue can fail to load on an older one; update Imervue if that happens.

## Plugins

| Plugin | What it does | Needs |
| --- | --- | --- |
| `plugins/ai_background_remover` | Remove image backgrounds with rembg (U²-Net), single image or batch | rembg, onnxruntime |
| `plugins/ai_colorize` | Colour black-and-white photos with preset palettes or an ONNX model | onnxruntime for the ONNX path |
| `plugins/ai_denoise` | Bilateral-filter or ONNX neural denoise | onnxruntime for the ONNX path |
| `plugins/ai_motion_deblur` | Wiener deconvolution or ONNX motion deblur | onnxruntime for the ONNX path |
| `plugins/ai_object_remove` | Click an object to select and inpaint it away; optional SAM point prompt | onnxruntime for SAM |
| `plugins/ai_outpaint` | Extend the canvas and fill the new border | — |
| `plugins/ai_portrait_relight` | Directional portrait relighting, heuristic or ONNX | onnxruntime for the ONNX path |
| `plugins/ai_smart_resize` | Content-aware resize (seam carving) | — |
| `plugins/ai_style_transfer` | Fast neural style transfer with ONNX style models | onnxruntime |
| `plugins/cloud_share` | Upload the current image to WebDAV or Imgur (HTTPS only) | — |
| `plugins/npr_filters` | Pencil sketch, oil painting, watercolour and line-art styles | opencv-python |
| `plugins/object_splitter` | Remove the background and save each object as its own transparent PNG | rembg, onnxruntime |
| `plugins/png_to_icon` | Convert a PNG into multi-size `.png` and `.ico` icons | — |
| `plugins/portrait_mode` | Blur the background behind the detected subject | rembg, onnxruntime |
| `plugins/safety_review` | Detect and mosaic explicit regions, with a manual editor and dataset export | nudenet + onnxruntime (photos), ultralytics + huggingface_hub (anime) |
| `plugins/video_source` | Scrub a video and extract still frames into the browser | — |
| `languages/spanish_translation` | Adds Spanish (Español) to the Language menu | — |

"—" means the plugin runs on Imervue's default dependencies.

## Writing a plugin

The plugin API (hooks, discovery, dependencies, background work, i18n, distribution rules) is documented in one place: [`PLUGIN_DEV_GUIDE.md`](https://github.com/JeffreyChen-s-Utils/Imervue/blob/main/PLUGIN_DEV_GUIDE.md) in the Imervue repository.

Plugins are developed and tested in the Imervue repository under `plugins/<name>/` and mirrored here; this repository holds no code of its own. Keep every file a plugin needs directly inside its directory: the downloader does not fetch subdirectories.

## License

See [LICENSE](LICENSE).
