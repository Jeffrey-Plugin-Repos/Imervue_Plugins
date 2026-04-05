"""
PNG to Icon Converter Plugin
Convert PNG images into multi-size .ico and .png icon files.

Requires Pillow — auto-installed on first use via the main app's pip installer.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QMessageBox, QFileDialog

from Imervue.plugin.plugin_base import ImervuePlugin
from Imervue.plugin.pip_installer import ensure_dependencies
from Imervue.multi_language.language_wrapper import language_wrapper

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMenu, QMenuBar
    from Imervue.gpu_image_view.gpu_image_view import GPUImageView

logger = logging.getLogger("Imervue.plugin.png_to_icon")

REQUIRED_PACKAGES = [
    ("PIL", "Pillow"),
]

SIZES = [16, 32, 48, 64, 128, 256]


def _ensure_deps(parent, on_ready):
    ensure_dependencies(parent, REQUIRED_PACKAGES, on_ready)


class IconConverterPlugin(ImervuePlugin):
    plugin_name = "PNG to Icon Converter"
    plugin_version = "1.5.0"
    plugin_description = "Convert PNG images into multi-size icons"
    plugin_author = "JE Chen"

    def _lang(self) -> dict:
        return language_wrapper.language_word_dict

    # ===========================
    # Menu Hooks
    # ===========================

    def on_build_menu_bar(self, menu_bar: QMenuBar) -> None:
        lang = self._lang()
        menu = menu_bar.addMenu(lang.get("icon_tools_menu", "Icon Tools"))

        action = menu.addAction(lang.get("convert_current", "Convert Current Image to Icon"))
        action.triggered.connect(self._convert_current_guarded)

        action2 = menu.addAction(lang.get("select_png", "Select PNG to Convert"))
        action2.triggered.connect(self._select_and_convert_guarded)

    def on_build_context_menu(self, menu: QMenu, viewer: GPUImageView) -> None:
        if not viewer.deep_zoom:
            return
        lang = self._lang()
        action = menu.addAction(lang.get("context_convert", "Convert to Icon"))
        action.triggered.connect(self._convert_current_guarded)

    # ===========================
    # 入口（經 ensure_dependencies）
    # ===========================

    def _convert_current_guarded(self):
        lang = self._lang()
        if not self.viewer.deep_zoom:
            QMessageBox.warning(
                self.main_window,
                lang.get("error", "Error"),
                lang.get("no_image", "No image loaded"),
            )
            return

        path = self.viewer.model.images[self.viewer.current_index]
        _ensure_deps(self.main_window, lambda: self._convert_to_icon(path))

    def _select_and_convert_guarded(self):
        lang = self._lang()
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            lang.get("select_title", "Select PNG"),
            "",
            "PNG Files (*.png)",
        )
        if file_path:
            _ensure_deps(self.main_window, lambda: self._convert_to_icon(file_path))

    # ===========================
    # 核心轉換
    # ===========================

    def _convert_to_icon(self, file_path: str):
        from PIL import Image

        lang = self._lang()
        try:
            img = Image.open(file_path).convert("RGBA")

            output_dir = Path(file_path).parent / "icons"
            output_dir.mkdir(parents=True, exist_ok=True)

            for size in SIZES:
                resized = img.resize((size, size), Image.LANCZOS)

                resized.save(str(output_dir / f"icon_{size}x{size}.png"))
                resized.save(
                    str(output_dir / f"icon_{size}x{size}.ico"),
                    format="ICO",
                    sizes=[(size, size)],
                )

            QMessageBox.information(
                self.main_window,
                "OK",
                f"{lang.get('success', 'Icons saved to:')} \n{output_dir}",
            )

        except Exception as e:
            logger.error(f"Icon conversion failed: {e}")
            QMessageBox.critical(
                self.main_window,
                lang.get("error", "Error"),
                str(e),
            )

    # ===========================
    # 翻譯
    # ===========================

    def get_translations(self) -> dict[str, dict[str, str]]:
        return {
            "English": {
                "icon_tools_menu": "Icon Tools",
                "convert_current": "Convert Current Image to Icon",
                "select_png": "Select PNG to Convert",
                "context_convert": "Convert to Icon",
                "no_image": "No image loaded",
                "success": "Icons saved to:",
                "error": "Error",
                "select_title": "Select PNG",
            },
            "Traditional_Chinese": {
                "icon_tools_menu": "Icon 工具",
                "convert_current": "轉換目前圖片為 Icon",
                "select_png": "選擇 PNG 轉換",
                "context_convert": "轉換為 Icon",
                "no_image": "尚未載入圖片",
                "success": "已輸出到：",
                "error": "錯誤",
                "select_title": "選擇 PNG",
            },
            "Chinese": {
                "icon_tools_menu": "Icon 工具",
                "convert_current": "转换当前图片为 Icon",
                "select_png": "选择 PNG 转换",
                "context_convert": "转换为 Icon",
                "no_image": "尚未加载图片",
                "success": "已输出到：",
                "error": "错误",
                "select_title": "选择 PNG",
            },
            "Japanese": {
                "icon_tools_menu": "アイコンツール",
                "convert_current": "現在の画像をアイコンに変換",
                "select_png": "PNGを選択して変換",
                "context_convert": "アイコンに変換",
                "no_image": "画像が読み込まれていません",
                "success": "保存先：",
                "error": "エラー",
                "select_title": "PNGを選択",
            },
            "Korean": {
                "icon_tools_menu": "아이콘 도구",
                "convert_current": "현재 이미지를 아이콘으로 변환",
                "select_png": "PNG 선택 후 변환",
                "context_convert": "아이콘으로 변환",
                "no_image": "이미지가 로드되지 않음",
                "success": "저장 위치:",
                "error": "오류",
                "select_title": "PNG 선택",
            },
        }
