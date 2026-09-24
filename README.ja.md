# Imervue プラグイン

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-TW.md">繁體中文</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <strong>日本語</strong> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.pt-BR.md">Português (BR)</a> ·
  <a href="README.ru.md">Русский</a>
</p>

GPU アクセラレーション対応の画像ビューア [Imervue](https://github.com/JeffreyChen-s-Utils/Imervue) の公式プラグインです。

## インストール

Imervue で **Plugins → Download Plugins** を開き、プラグインを選んで **Download** をクリックし、Imervue を再起動します（または **Plugins → Reload Plugins** を使用）。ダウンローダはこのリポジトリの `main` ブランチを読み取り、各プラグインを Imervue の隣にある `plugins/` ディレクトリにインストールします（**Plugins → Open Plugin Folder**）。

手動でインストールするには、`plugins/npr_filters/` のようなプラグインディレクトリをその `plugins/` ディレクトリにコピーします。インストール時にカテゴリ階層（`plugins/`、`languages/`）は取り除かれます。

大きなパッケージ（onnxruntime、rembg、OpenCV など）を必要とするプラグインは、それを必要とする機能を初めて使うときにインストールを提案します。ONNX ベースのプラグインは、自身のプラグインディレクトリ内の `models/` フォルダにあるモデルファイルを探します。これらのファイルはダウンロードされないので、使いたいモデルをそこに置いてください。より新しい Imervue 向けに書かれたプラグインは、古い Imervue では読み込めないことがあります。その場合は Imervue を更新してください。

## プラグイン

| プラグイン | 機能 | 必要なもの |
| --- | --- | --- |
| `plugins/ai_background_remover` | rembg（U²-Net）で画像の背景を除去、単一画像またはバッチ | rembg、onnxruntime |
| `plugins/ai_colorize` | プリセットパレットまたは ONNX モデルで白黒写真に着色 | ONNX パスには onnxruntime |
| `plugins/ai_denoise` | バイラテラルフィルタまたは ONNX ニューラルノイズ除去 | ONNX パスには onnxruntime |
| `plugins/ai_motion_deblur` | Wiener デコンボリューションまたは ONNX モーションブラー除去 | ONNX パスには onnxruntime |
| `plugins/ai_object_remove` | オブジェクトをクリックして選択し、インペイントで消去。オプションで SAM ポイントプロンプト | SAM には onnxruntime |
| `plugins/ai_outpaint` | キャンバスを拡張し、新しい境界を埋める | — |
| `plugins/ai_portrait_relight` | 方向性のあるポートレート再ライティング、ヒューリスティックまたは ONNX | ONNX パスには onnxruntime |
| `plugins/ai_smart_resize` | コンテンツ認識リサイズ（シームカービング） | — |
| `plugins/ai_style_transfer` | ONNX スタイルモデルによる高速ニューラルスタイル転送 | onnxruntime |
| `plugins/cloud_share` | 現在の画像を WebDAV または Imgur にアップロード（HTTPS のみ） | — |
| `plugins/npr_filters` | 鉛筆スケッチ、油絵、水彩、線画スタイル | opencv-python |
| `plugins/object_splitter` | 背景を除去し、各オブジェクトを個別の透明 PNG として保存 | rembg、onnxruntime |
| `plugins/png_to_icon` | PNG を複数サイズの `.png` と `.ico` アイコンに変換 | — |
| `plugins/portrait_mode` | 検出された被写体の背後の背景をぼかす | rembg、onnxruntime |
| `plugins/safety_review` | 露骨な領域を検出してモザイク化、手動エディタとデータセットエクスポート付き | 写真は nudenet + onnxruntime、アニメは ultralytics + huggingface_hub |
| `plugins/video_source` | 動画をスクラブして静止フレームをブラウザに抽出 | — |
| `languages/spanish_translation` | 言語メニューにスペイン語（Español）を追加 | — |

「—」は、そのプラグインが Imervue のデフォルトの依存関係だけで動作することを意味します。

## プラグインの作成

プラグイン API（フック、検出、依存関係、バックグラウンド処理、i18n、配布ルール）は一箇所にまとめて記載されています。Imervue リポジトリ内の [`PLUGIN_DEV_GUIDE.md`](https://github.com/JeffreyChen-s-Utils/Imervue/blob/main/PLUGIN_DEV_GUIDE.md) を参照してください。

プラグインは Imervue リポジトリの `plugins/<name>/` 以下で開発・テストされ、ここにミラーリングされます。このリポジトリ自体はコードを持ちません。プラグインが必要とするすべてのファイルをそのディレクトリの直下に置いてください。ダウンローダはサブディレクトリを取得しません。

## ライセンス

[LICENSE](LICENSE) を参照してください。
