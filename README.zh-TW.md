# Imervue 外掛

<p align="center">
  <a href="README.md">English</a> ·
  <strong>繁體中文</strong> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.pt-BR.md">Português (BR)</a> ·
  <a href="README.ru.md">Русский</a>
</p>

[Imervue](https://github.com/JeffreyChen-s-Utils/Imervue) 的官方外掛，這是一款 GPU 加速的圖片檢視器。

## 安裝

在 Imervue 中開啟 **Plugins → Download Plugins**，挑一個外掛並點 **Download**，然後重新啟動 Imervue（或使用 **Plugins → Reload Plugins**）。下載器會讀取本儲存庫的 `main` 分支，並把每個外掛安裝到 Imervue 旁邊的 `plugins/` 目錄（**Plugins → Open Plugin Folder**）。

若要手動安裝，把像 `plugins/npr_filters/` 這樣的外掛目錄複製進那個 `plugins/` 目錄。安裝時會去掉分類層級（`plugins/`、`languages/`）。

需要大型套件（onnxruntime、rembg、OpenCV……）的外掛，會在你第一次使用需要它的功能時提議安裝。以 ONNX 為基礎的外掛會在自己外掛目錄裡的 `models/` 資料夾尋找模型檔；這些檔案不會被下載，所以請把你想用的模型放到那裡。針對較新版 Imervue 撰寫的外掛，可能無法在較舊版上載入；若發生這種情形，請更新 Imervue。

## 外掛

| 外掛 | 功能 | 需求 |
| --- | --- | --- |
| `plugins/ai_background_remover` | 用 rembg（U²-Net）移除圖片背景，可單張或批次處理 | rembg、onnxruntime |
| `plugins/ai_colorize` | 用預設色盤或 ONNX 模型為黑白照片上色 | ONNX 路徑需要 onnxruntime |
| `plugins/ai_denoise` | 雙邊濾波或 ONNX 神經網路降噪 | ONNX 路徑需要 onnxruntime |
| `plugins/ai_motion_deblur` | Wiener 反卷積或 ONNX 動態模糊去除 | ONNX 路徑需要 onnxruntime |
| `plugins/ai_object_remove` | 點選物件以選取並將其修補移除；可選用 SAM 點提示 | SAM 需要 onnxruntime |
| `plugins/ai_outpaint` | 擴展畫布並填滿新增的邊界 | — |
| `plugins/ai_portrait_relight` | 方向性人像重打光，啟發式或 ONNX | ONNX 路徑需要 onnxruntime |
| `plugins/ai_smart_resize` | 內容感知縮放（接縫裁剪） | — |
| `plugins/ai_style_transfer` | 使用 ONNX 風格模型的快速神經風格轉換 | onnxruntime |
| `plugins/cloud_share` | 把目前的圖片上傳到 WebDAV 或 Imgur（僅限 HTTPS） | — |
| `plugins/npr_filters` | 鉛筆素描、油畫、水彩與線稿風格 | opencv-python |
| `plugins/object_splitter` | 移除背景並把每個物件各自存成透明 PNG | rembg、onnxruntime |
| `plugins/png_to_icon` | 把 PNG 轉成多尺寸的 `.png` 與 `.ico` 圖示 | — |
| `plugins/portrait_mode` | 模糊偵測到的主體背後的背景 | rembg、onnxruntime |
| `plugins/safety_review` | 偵測並馬賽克露骨區域，附手動編輯器與資料集匯出 | 照片用 nudenet + onnxruntime，動漫用 ultralytics + huggingface_hub |
| `plugins/video_source` | 拖曳瀏覽影片並把靜態畫格擷取到瀏覽器中 | — |
| `languages/spanish_translation` | 在語言選單中加入西班牙文（Español） | — |

「—」代表該外掛只靠 Imervue 的預設相依套件即可執行。

## 撰寫外掛

外掛 API（鉤子、探索、相依套件、背景工作、i18n、發佈規則）都集中記載在一個地方：Imervue 儲存庫中的 [`PLUGIN_DEV_GUIDE.md`](https://github.com/JeffreyChen-s-Utils/Imervue/blob/main/PLUGIN_DEV_GUIDE.md)。

外掛是在 Imervue 儲存庫的 `plugins/<name>/` 底下開發與測試，再鏡像到這裡；本儲存庫本身不含任何程式碼。請把外掛需要的每個檔案都直接放在它的目錄裡：下載器不會抓取子目錄。

## 授權

見 [LICENSE](LICENSE)。
