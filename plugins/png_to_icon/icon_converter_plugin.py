import os
from PIL import Image
from PySide6.QtWidgets import QMessageBox, QFileDialog
from Imervue.plugin.plugin_base import ImervuePlugin
from Imervue.multi_language.language_wrapper import language_wrapper


class IconConverterPlugin(ImervuePlugin):
    plugin_name = "PNG to Icon Converter"
    plugin_version = "1.4.0"
    plugin_description = "Convert PNG images into multi-size icons"
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

    def on_plugin_loaded(self):
        print(f"{self.plugin_name} loaded!")

    # 取得語言字典（方便用）
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

    # 🖱右鍵選單（重點）
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

    # 核心轉換：每個尺寸輸出獨立的 .ico 檔案
    def convert_to_icon(self, file_path):
        lang = self.lang()

        try:
            # 開啟原始圖片並確保是 RGBA
            img = Image.open(file_path).convert("RGBA")

            # 建立輸出目錄
            output_dir = os.path.join(os.path.dirname(file_path), "icons")
            os.makedirs(output_dir, exist_ok=True)

            # 遍歷尺寸列表
            for size in self.SIZES:
                # 1. 調整尺寸
                resized = img.resize((size, size), Image.LANCZOS)

                # 2. 儲存為個別的 PNG (保留原本功能)
                png_name = f"icon_{size}x{size}.png"
                resized.save(os.path.join(output_dir, png_name))

                # 3. 儲存為個別的 ICO (關鍵修改)
                ico_name = f"icon_{size}x{size}.ico"
                ico_path = os.path.join(output_dir, ico_name)

                # 針對單一尺寸儲存為 ICO 格式
                resized.save(
                    ico_path,
                    format="ICO",
                    sizes=[(size, size)]
                )

            QMessageBox.information(
                self.main_window,
                "OK",
                f"{lang.get('success', 'Saved to')} \n{output_dir}"
            )

        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                lang.get("error", "Error"),
                str(e)
            )