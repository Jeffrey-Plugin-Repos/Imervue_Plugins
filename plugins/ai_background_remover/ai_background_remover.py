"""
AI 去背插件
AI Background Remover — remove image backgrounds using rembg (U2-Net).

Dependencies are auto-installed on first use via the main app's pip installer.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QFileDialog, QLineEdit, QProgressBar,
    QCheckBox, QMenu,
)

from Imervue.plugin.plugin_base import ImervuePlugin
from Imervue.plugin.pip_installer import ensure_dependencies
from Imervue.multi_language.language_wrapper import language_wrapper

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMenuBar
    from Imervue.Imervue_main_window import ImervueMainWindow
    from Imervue.gpu_image_view.gpu_image_view import GPUImageView

logger = logging.getLogger("Imervue.plugin.ai_bg_remover")

# ===========================
# 套件需求定義
# ===========================

REQUIRED_PACKAGES = [
    # (import_name, pip_name)
    ("rembg", "rembg"),
    ("onnxruntime", "onnxruntime"),
]

# rembg 支援的模型
MODELS = [
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
]

MODEL_DESCRIPTIONS = {
    "u2net": "General purpose (173 MB)",
    "u2netp": "Lightweight (4 MB, faster)",
    "u2net_human_seg": "Human segmentation",
    "u2net_cloth_seg": "Clothing segmentation",
    "silueta": "General purpose (compact)",
    "isnet-general-use": "IS-Net general (high quality)",
    "isnet-anime": "IS-Net anime/illustration",
}


# ===========================
# 去背 Workers
# ===========================

class _RemoveBackgroundWorker(QThread):
    """背景執行緒處理去背"""
    progress = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, input_path: str, output_path: str, model_name: str,
                 alpha_matting: bool):
        super().__init__()
        self._input = input_path
        self._output = output_path
        self._model = model_name
        self._alpha_matting = alpha_matting

    def run(self):
        try:
            self.progress.emit("Loading rembg...")
            from rembg import remove, new_session

            self.progress.emit(f"Loading model: {self._model}...")
            session = new_session(self._model)

            self.progress.emit("Processing image...")
            from PIL import Image
            input_img = self._load_image(self._input)

            output_img = remove(
                input_img,
                session=session,
                alpha_matting=self._alpha_matting,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=10,
            )

            self.progress.emit("Saving result...")
            output_img.save(self._output)

            self.finished.emit(True, self._output)
        except Exception as exc:
            logger.error(f"Background removal failed: {exc}")
            self.finished.emit(False, str(exc))

    @staticmethod
    def _load_image(path: str):
        from PIL import Image
        if Path(path).suffix.lower() == ".svg":
            from Imervue.gpu_image_view.images.image_loader import _load_svg
            arr = _load_svg(path, thumbnail=False)
            return Image.fromarray(arr)
        return Image.open(path)


class _BatchRemoveWorker(QThread):
    """批次去背"""
    progress = Signal(int, int, str)
    finished = Signal(int, int)

    def __init__(self, paths: list[str], output_dir: str, model_name: str,
                 alpha_matting: bool):
        super().__init__()
        self._paths = paths
        self._output_dir = output_dir
        self._model = model_name
        self._alpha_matting = alpha_matting

    def run(self):
        from rembg import remove, new_session
        from PIL import Image

        session = new_session(self._model)
        success = 0
        failed = 0
        total = len(self._paths)

        for i, src in enumerate(self._paths):
            try:
                self.progress.emit(i, total, Path(src).name)

                if Path(src).suffix.lower() == ".svg":
                    from Imervue.gpu_image_view.images.image_loader import _load_svg
                    arr = _load_svg(src, thumbnail=False)
                    input_img = Image.fromarray(arr)
                else:
                    input_img = Image.open(src)

                output_img = remove(
                    input_img,
                    session=session,
                    alpha_matting=self._alpha_matting,
                    alpha_matting_foreground_threshold=240,
                    alpha_matting_background_threshold=10,
                    alpha_matting_erode_size=10,
                )

                out_name = Path(src).stem + "_nobg.png"
                out_path = Path(self._output_dir) / out_name
                counter = 1
                while out_path.exists():
                    out_name = f"{Path(src).stem}_nobg_{counter}.png"
                    out_path = Path(self._output_dir) / out_name
                    counter += 1

                output_img.save(str(out_path))
                success += 1
            except Exception as exc:
                logger.error(f"Batch bg removal failed for {src}: {exc}")
                failed += 1

        self.finished.emit(success, failed)


# ===========================
# 對話框
# ===========================

class RemoveBackgroundDialog(QDialog):
    """單張圖片去背對話框"""

    def __init__(self, main_gui: GPUImageView, image_path: str):
        super().__init__(main_gui.main_window)
        self._gui = main_gui
        self._image_path = image_path
        self._lang = language_wrapper.language_word_dict
        self._worker = None

        self.setWindowTitle(self._lang.get("bg_remove_title", "AI Background Removal"))
        self.setMinimumWidth(480)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Source
        layout.addWidget(QLabel(
            self._lang.get("bg_remove_source", "Source:") + f"  {Path(self._image_path).name}"
        ))

        # Model
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel(self._lang.get("bg_remove_model", "Model:")))
        self._model_combo = QComboBox()
        for m in MODELS:
            desc = MODEL_DESCRIPTIONS.get(m, "")
            self._model_combo.addItem(f"{m}  —  {desc}", m)
        self._model_combo.setCurrentIndex(0)
        model_row.addWidget(self._model_combo, 1)
        layout.addLayout(model_row)

        # Alpha matting
        self._alpha_check = QCheckBox(
            self._lang.get("bg_remove_alpha_matting", "Alpha matting (smoother edges, slower)")
        )
        self._alpha_check.setChecked(False)
        layout.addWidget(self._alpha_check)

        # Output path
        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        default_out = Path(self._image_path).parent / (Path(self._image_path).stem + "_nobg.png")
        self._path_edit.setText(str(default_out))
        browse_btn = QPushButton(self._lang.get("export_browse", "Browse..."))
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(self._path_edit, 1)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        # Progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton(self._lang.get("export_cancel", "Cancel"))
        cancel_btn.clicked.connect(self.reject)
        self._run_btn = QPushButton(self._lang.get("bg_remove_run", "Remove Background"))
        self._run_btn.clicked.connect(self._do_remove)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self._run_btn)
        layout.addLayout(btn_row)

    def _browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self, self._lang.get("export_save", "Save"),
            self._path_edit.text(), "PNG (*.png)",
        )
        if path:
            self._path_edit.setText(path)

    def _do_remove(self):
        output = self._path_edit.text().strip()
        if not output:
            return

        self._run_btn.setEnabled(False)
        self._progress_bar.setVisible(True)

        model = self._model_combo.currentData()
        self._worker = _RemoveBackgroundWorker(
            self._image_path, output, model, self._alpha_check.isChecked()
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, msg: str):
        self._status_label.setText(msg)

    def _on_finished(self, success: bool, result: str):
        self._progress_bar.setVisible(False)
        self._run_btn.setEnabled(True)
        self._worker = None

        if success:
            self._status_label.setText(
                self._lang.get("bg_remove_done", "Done! Saved to: {path}").format(path=result)
            )
            if hasattr(self._gui.main_window, "toast"):
                self._gui.main_window.toast.success(
                    self._lang.get("bg_remove_done_short", "Background removed!")
                )
            self.accept()
        else:
            self._status_label.setText(f"Error: {result}")
            if hasattr(self._gui.main_window, "toast"):
                self._gui.main_window.toast.info(f"Error: {result}")


class BatchRemoveBackgroundDialog(QDialog):
    """批次去背對話框"""

    def __init__(self, main_gui: GPUImageView, paths: list[str]):
        super().__init__(main_gui.main_window)
        self._gui = main_gui
        self._paths = paths
        self._lang = language_wrapper.language_word_dict
        self._worker = None

        self.setWindowTitle(self._lang.get("bg_remove_batch_title", "Batch AI Background Removal"))
        self.setMinimumWidth(480)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            self._lang.get("batch_export_count", "{count} image(s) selected").format(
                count=len(self._paths))
        ))

        # Model
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel(self._lang.get("bg_remove_model", "Model:")))
        self._model_combo = QComboBox()
        for m in MODELS:
            desc = MODEL_DESCRIPTIONS.get(m, "")
            self._model_combo.addItem(f"{m}  —  {desc}", m)
        model_row.addWidget(self._model_combo, 1)
        layout.addLayout(model_row)

        # Alpha matting
        self._alpha_check = QCheckBox(
            self._lang.get("bg_remove_alpha_matting", "Alpha matting (smoother edges, slower)")
        )
        layout.addWidget(self._alpha_check)

        # Output dir
        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit()
        if self._paths:
            self._dir_edit.setText(str(Path(self._paths[0]).parent))
        browse_btn = QPushButton(self._lang.get("export_browse", "Browse..."))
        browse_btn.clicked.connect(self._browse)
        dir_row.addWidget(self._dir_edit, 1)
        dir_row.addWidget(browse_btn)
        layout.addLayout(dir_row)

        # Progress
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton(self._lang.get("export_cancel", "Cancel"))
        cancel_btn.clicked.connect(self.reject)
        self._run_btn = QPushButton(self._lang.get("bg_remove_run", "Remove Background"))
        self._run_btn.clicked.connect(self._do_remove)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self._run_btn)
        layout.addLayout(btn_row)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self._dir_edit.setText(folder)

    def _do_remove(self):
        output_dir = self._dir_edit.text().strip()
        if not output_dir or not Path(output_dir).is_dir():
            return

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setMaximum(len(self._paths))
        self._progress.setValue(0)

        model = self._model_combo.currentData()
        self._worker = _BatchRemoveWorker(
            self._paths, output_dir, model, self._alpha_check.isChecked()
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, current, total, name):
        self._progress.setValue(current)
        self._status_label.setText(f"{current}/{total}  {name}")

    def _on_finished(self, success, failed):
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._worker = None

        msg = self._lang.get(
            "bg_remove_batch_done", "Processed {success}/{total} image(s)"
        ).format(success=success, total=success + failed)
        self._status_label.setText(msg)

        if hasattr(self._gui.main_window, "toast"):
            if failed:
                self._gui.main_window.toast.info(msg)
            else:
                self._gui.main_window.toast.success(msg)
        self.accept()


# ===========================
# Plugin 本體
# ===========================

def _ensure_deps(parent, on_ready):
    """透過主程式的 pip_installer 確認依賴"""
    ensure_dependencies(parent, REQUIRED_PACKAGES, on_ready)


class AIBackgroundRemoverPlugin(ImervuePlugin):
    plugin_name = "AI Background Remover"
    plugin_version = "1.0.0"
    plugin_description = "Remove image backgrounds using AI (rembg / U2-Net)"
    plugin_author = "Imervue"

    def on_build_menu_bar(self, menu_bar: QMenuBar) -> None:
        lang = language_wrapper.language_word_dict
        self._menu = menu_bar.addMenu(lang.get("bg_remove_menu", "AI Tools"))

        action = self._menu.addAction(lang.get("bg_remove_title", "AI Background Removal"))
        action.triggered.connect(self._open_single_dialog)

        batch_action = self._menu.addAction(
            lang.get("bg_remove_batch_title", "Batch AI Background Removal")
        )
        batch_action.triggered.connect(self._open_batch_dialog)

    def on_build_context_menu(self, menu: QMenu, viewer: GPUImageView) -> None:
        lang = language_wrapper.language_word_dict

        if viewer.deep_zoom:
            images = viewer.model.images
            if images and 0 <= viewer.current_index < len(images):
                path = images[viewer.current_index]
                action = menu.addAction(lang.get("bg_remove_title", "AI Background Removal"))
                action.triggered.connect(lambda: self._remove_single(path))

        if (viewer.tile_grid_mode and viewer.tile_selection_mode
                and viewer.selected_tiles and len(viewer.selected_tiles) >= 1):
            paths = list(viewer.selected_tiles)
            action = menu.addAction(
                lang.get("bg_remove_batch_title", "Batch AI Background Removal")
            )
            action.triggered.connect(lambda: self._remove_batch(paths))

    # ----- 入口：全部經過 ensure_dependencies -----

    def _open_single_dialog(self):
        images = self.viewer.model.images
        if not images or self.viewer.current_index >= len(images):
            return
        path = images[self.viewer.current_index]
        self._remove_single(path)

    def _remove_single(self, path: str):
        if not Path(path).is_file():
            return
        _ensure_deps(
            self.main_window,
            lambda: RemoveBackgroundDialog(self.viewer, path).exec(),
        )

    def _open_batch_dialog(self):
        if (self.viewer.tile_grid_mode and self.viewer.tile_selection_mode
                and self.viewer.selected_tiles):
            paths = list(self.viewer.selected_tiles)
        else:
            paths = list(self.viewer.model.images)
        if paths:
            self._remove_batch(paths)

    def _remove_batch(self, paths: list[str]):
        _ensure_deps(
            self.main_window,
            lambda: BatchRemoveBackgroundDialog(self.viewer, paths).exec(),
        )

    def get_translations(self) -> dict[str, dict[str, str]]:
        return {
            "English": {
                "bg_remove_menu": "AI Tools",
                "bg_remove_title": "AI Background Removal",
                "bg_remove_batch_title": "Batch AI Background Removal",
                "bg_remove_source": "Source:",
                "bg_remove_model": "Model:",
                "bg_remove_alpha_matting": "Alpha matting (smoother edges, slower)",
                "bg_remove_run": "Remove Background",
                "bg_remove_done": "Done! Saved to: {path}",
                "bg_remove_done_short": "Background removed!",
                "bg_remove_batch_done": "Processed {success}/{total} image(s)",
            },
            "Traditional_Chinese": {
                "bg_remove_menu": "AI 工具",
                "bg_remove_title": "AI 去背",
                "bg_remove_batch_title": "批次 AI 去背",
                "bg_remove_source": "來源：",
                "bg_remove_model": "模型：",
                "bg_remove_alpha_matting": "Alpha matting（邊���更平滑，較慢）",
                "bg_remove_run": "去除背景",
                "bg_remove_done": "完成！已儲存至：{path}",
                "bg_remove_done_short": "去背完成！",
                "bg_remove_batch_done": "已處理 {success}/{total} 張圖片",
            },
            "Chinese": {
                "bg_remove_menu": "AI 工具",
                "bg_remove_title": "AI 去背",
                "bg_remove_batch_title": "批量 AI 去背",
                "bg_remove_source": "来源：",
                "bg_remove_model": "模型：",
                "bg_remove_alpha_matting": "Alpha matting（边缘更平滑，较慢）",
                "bg_remove_run": "去除背景",
                "bg_remove_done": "完成！已保存至：{path}",
                "bg_remove_done_short": "去背完成！",
                "bg_remove_batch_done": "已处理 {success}/{total} 张图片",
            },
            "Japanese": {
                "bg_remove_menu": "AI ツール",
                "bg_remove_title": "AI 背景除去",
                "bg_remove_batch_title": "一括 AI 背景除去",
                "bg_remove_source": "ソース：",
                "bg_remove_model": "モデル：",
                "bg_remove_alpha_matting": "アルファマッティング（より滑らかな境界、低速）",
                "bg_remove_run": "背景を除去",
                "bg_remove_done": "完了！保存先：{path}",
                "bg_remove_done_short": "背景除去完了！",
                "bg_remove_batch_done": "{success}/{total} 枚の画像を処理しました",
            },
            "Korean": {
                "bg_remove_menu": "AI 도구",
                "bg_remove_title": "AI 배경 제거",
                "bg_remove_batch_title": "일괄 AI 배경 제거",
                "bg_remove_source": "소스:",
                "bg_remove_model": "모델:",
                "bg_remove_alpha_matting": "알파 매팅 (더 부드러운 경계, 느림)",
                "bg_remove_run": "배경 제거",
                "bg_remove_done": "완료! 저장 위치: {path}",
                "bg_remove_done_short": "배경 제거 완료!",
                "bg_remove_batch_done": "{success}/{total}개의 이미지를 처리했습니다",
            },
        }
