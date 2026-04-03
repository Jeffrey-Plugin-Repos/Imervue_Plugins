import os
from PIL import Image
from PySide6.QtWidgets import QMessageBox, QFileDialog
from Imervue.plugin.plugin_base import ImervuePlugin
from Imervue.multi_language.language_wrapper import language_wrapper


class IconConverterPlugin(ImervuePlugin):
    plugin_name = "PNG to Icon Converter"
    plugin_version = "1.2.0"
    plugin_description = "Convert PNG images into multi-size icons (.ico)"
    plugin_author = "JE Chen"

    SIZES = [16, 32, 48, 64, 128, 256]

    # 多語言
    def get_translations(self):
        return {
            "English": {
                "icon_tools_menu": "Icon Tools",
                "convert_current": "Convert Current Image to Icon",
                "select_png": "Select PNG to Convert",
                "context_convert": "Convert to Icon",
                "no_image": "No image loaded",
                "success": "Icon saved to:",
                "error": "Error",
                "select_title": "Select PNG",
                "select_save_path": "Save Icon",
            },
            "Traditional_Chinese": {
                "icon_tools_menu": "Icon 工具",
                "convert_current": "轉換目前圖片為 Icon",
                "select_png": "選擇 PNG 轉換",
                "context_convert": "轉換為 Icon",
                "no_image": "尚未載入圖片",
                "success": "Icon 已儲存至：",
                "error": "錯誤",
                "select_title": "選擇 PNG",
                "select_save_path": "儲存 Icon",
            },
            "Chinese": {
                "icon_tools_menu": "Icon 工具",
                "convert_current": "转换当前图片为 Icon",
                "select_png": "选择 PNG 转换",
                "context_convert": "转换为 Icon",
                "no_image": "尚未加载图片",
                "success": "Icon 已保存至：",
                "error": "错误",
                "select_title": "选择 PNG",
                "select_save_path": "保存 Icon",
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
                "select_save_path": "アイコンを保存",
            },
            "Korean": {
                "icon_tools_menu": "아이콘 도구",
                "convert_current": "현재 이미지를 아이콘으로 변환",
                "select_png": "PNG 선택 후 변환",
                "context_convert": "아이콘으로 변환",
                "no_image": "이미지가 로드되지 않음",
                "success": "아이콘 저장 위치:",
                "error": "오류",
                "select_title": "PNG 선택",
                "select_save_path": "아이콘 저장",
            },
        }

    def on_plugin_loaded(self):
        print(f"{self.plugin_name} loaded!")

    def lang(self):
        return language_wrapper.language_word_dict

    # 選單列
    def on_build_menu_bar(self, menu_bar):
        lang = self.lang()

        menu = menu_bar.addMenu(
            lang.get("icon_tools_menu", "Icon Tools")
        )

        action = menu.addAction(
            lang.get("convert_current", "Convert Current Image")
        )
        action.triggered.connect(self.convert_current_image)

        action2 = menu.addAction(
            lang.get("select_png", "Select PNG")
        )
        action2.triggered.connect(self.select_and_convert)

    # 右鍵選單
    def on_build_context_menu(self, menu, viewer):
        lang = self.lang()

        action = menu.addAction(
            lang.get("context_convert", "Convert to Icon")
        )
        action.triggered.connect(self.convert_current_image)

    # 轉換目前圖片
    def convert_current_image(self):
        lang = self.lang()
        viewer = self.viewer

        if not viewer.deep_zoom:
            QMessageBox.warning(
                self.main_window,
                lang.get("error", "Error"),
                lang.get("no_image", "No image loaded")
            )
            return

        image_path = viewer.model.images[viewer.current_index]
        self.convert_to_icon(image_path)

    # 手動選檔
    def select_and_convert(self):
        lang = self.lang()

        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            lang.get("select_title", "Select PNG"),
            "",
            "PNG Files (*.png)"
        )

        if file_path:
            self.convert_to_icon(file_path)

    # ✂裁切正方形
    def crop_to_square(self, img):
        w, h = img.size
        min_side = min(w, h)
        left = (w - min_side) // 2
        top = (h - min_side) // 2
        return img.crop((left, top, left + min_side, top + min_side))

    # 核心轉換（改成只輸出 ICO）
    def convert_to_icon(self, file_path):
        lang = self.lang()

        try:
            img = Image.open(file_path).convert("RGBA")

            # ✂防止 icon 變形
            img = self.crop_to_square(img)

            sizes = [(s, s) for s in self.SIZES]

            # 預設檔名
            default_name = os.path.splitext(os.path.basename(file_path))[0] + ".ico"

            save_path, _ = QFileDialog.getSaveFileName(
                self.main_window,
                lang.get("select_save_path", "Save Icon"),
                default_name,
                "ICO Files (*.ico)"
            )

            if not save_path:
                return

            # 自動補副檔名
            if not save_path.lower().endswith(".ico"):
                save_path += ".ico"

            # 直接輸出 ICO（多尺寸）
            img.save(save_path, format="ICO", sizes=sizes)

            QMessageBox.information(
                self.main_window,
                "OK",
                f"{lang.get('success', 'Saved to')} \n{save_path}"
            )

        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                lang.get("error", "Error"),
                str(e)
            )