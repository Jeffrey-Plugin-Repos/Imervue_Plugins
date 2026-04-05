"""
Spanish Language Plugin for Imervue
====================================

Adds Spanish (Espanol) language support to Imervue.

This is also a reference for plugin developers who want to create
their own language plugins. To create a new language plugin:

1. Copy this folder and rename it (e.g. ``french_language``).
2. Replace all Spanish strings with your target language.
3. Update ``__init__.py`` to point to your new class.
4. Place the folder in the ``plugins/`` directory and restart Imervue.
"""
from __future__ import annotations

from Imervue.multi_language.language_wrapper import language_wrapper
from Imervue.plugin.plugin_base import ImervuePlugin

_SPANISH_DICT: dict[str, str] = {
    # Main window / file
    "main_window_current_filename_format": "Nombre de archivo actual: {name}",
    "main_window_open_image": "Abrir archivo",
    "main_window_current_filename": "Nombre de archivo actual:",
    "main_window_current_file": "Archivo",
    "main_window_open_folder": "Abrir carpeta",
    "main_window_exit": "Salir",
    "main_window_tile_size": "Tamano de miniatura",
    "main_window_select_folder": "Seleccionar carpeta",
    "main_window_current_folder_format": "Carpeta actual: {path}",
    "main_window_remove_undo_stack": "Eliminar archivos temporales de deshacer",

    # Language menu
    "menu_bar_language": "Idioma",
    "language_menu_bar_please_restart_messagebox": "Por favor, reinicie la aplicacion",
    "language_menu_bar_english": "Ingles",
    "language_menu_bar_traditional_chinese": "Chino tradicional",
    "language_menu_bar_chinese": "Chino simplificado",
    "language_menu_bar_koren": "Coreano",
    "language_menu_bar_japanese": "Japones",

    # Tip / shortcuts menu
    "main_window_tip_menu": "Instrucciones",
    "tip_show_shortcuts": "Atajos de teclado y raton",
    "tip_close": "Cerrar",
    "main_window_mouse_tip_menu": "Control del raton",
    "main_window_keyboard_tip_menu": "Control del teclado",
    "tip_mouse_middle": "Boton central del raton",
    "mouse_control_middle_tip": "Puede desplazarse usando el boton central del raton",
    "tip_mouse_left_click": "Clic izquierdo",
    "mouse_control_left_tip": "Haga clic en una imagen con el boton izquierdo para entrar en el modo Deep Zoom",
    "tip_mouse_left_drag": "Arrastrar izquierdo",
    "mouse_control_multi_select_tip": "Mantenga presionado el boton izquierdo del raton para seleccionar multiples imagenes en el modo de miniaturas",
    "tip_mouse_right": "Clic derecho",
    "tip_mouse_right_desc": "Abrir menu contextual",
    "tip_mouse_scroll": "Rueda del raton",
    "tip_mouse_scroll_desc": "Acercar / alejar",
    "keyboard_control_esc_tip": "Presione ESC para salir del modo Deep Zoom",
    "keyboard_f_tip": "Pantalla completa",
    "keyboard_r_tip": "Restablecer las coordenadas / Restablecer todo (Shift+R)",
    "keyboard_home_tip": "Ajustar imagen a la ventana",
    "keyboard_e_tip": "Abrir editor de imagen",
    "keyboard_slideshow_tip": "Iniciar / detener presentacion de diapositivas",
    "keyboard_t_tip": "Abrir gestion de etiquetas y albumes",
    "keyboard_h_tip": "Voltear horizontalmente",
    "keyboard_w_tip": "Siguiente / anterior carpeta (Shift+W = anterior)",
    "keyboard_b_tip": "Agregar / quitar marcador",
    "keyboard_control_delete_tip": "Puede usar la tecla Suprimir para eliminar imagenes tanto en miniaturas como en Deep Zoom",
    "keyboard_ctrl_c_tip": "Copiar imagenes seleccionadas",
    "keyboard_ctrl_v_tip": "Pegar imagenes",
    "keyboard_undo_tip": "Deshacer",
    "keyboard_redo_tip": "Rehacer",
    "keyboard_search_tip": "Abrir busqueda",
    "keyboard_rating_tip": "Calificar imagen de 1 a 5 estrellas",
    "keyboard_favorite_tip": "Marcar / desmarcar como favorito",
    "keyboard_arrow_lr_tip": "Imagen anterior / siguiente (Deep Zoom)",
    "keyboard_arrow_tile_tip": "Navegar por miniaturas",
    "keyboard_anim_play_tip": "Reproducir / pausar animacion",
    "keyboard_anim_frame_tip": "Cuadro anterior / siguiente",
    "keyboard_anim_speed_tip": "Reducir / aumentar velocidad de animacion",

    # Right-click menu
    "right_click_menu_go_to_parent_folder": "Ir a la carpeta principal",
    "right_click_menu_next_image": "Imagen siguiente",
    "right_click_menu_previous_image": "Imagen anterior",
    "right_click_menu_delete_current": "Eliminar imagen actual",
    "right_click_menu_delete_selected": "Eliminar imagenes seleccionadas",
    "right_click_menu_image_info": "Informacion de la imagen",

    # Image info
    "image_info_filename": "Nombre de archivo: {info}\n",
    "image_info_fullpath": "Ruta completa: {full_path}\n",
    "image_info_image_size": "Tamano: {width} x {height}\n",
    "image_info_file_size": "Tamano de archivo: {file_size_mb} MB\n",
    "image_info_file_created_time": "Fecha de creacion: {created_time}\n",
    "image_info_file_modified_time": "Fecha de modificacion: {modified_time}\n",
    "image_info_messagebox_title": "Informacion de la imagen",

    # Image info exif
    "image_info_exif_datatime_original": "Fecha de captura: {DateTimeOriginal}\n",
    "image_info_exif_camera_model": "Camara: {Make} {Model}\n",
    "image_info_exif_camera_lens_model": "Lente: {LensModel}\n",
    "image_info_exif_camera_focal_length": "Distancia focal: {FocalLength}\n",
    "image_info_exif_camera_fnumber": "Apertura: {FNumber}\n",
    "image_info_exif_exposure_time": "Obturador: {ExposureTime}\n",
    "image_info_exif_iso": "ISO: {ISOSpeedRatings}",

    # Recent menu
    "recent_menu_title": "Abiertos recientes",

    # Plugin menu
    "plugin_menu_title": "Complementos",
    "plugin_menu_loaded": "Complementos cargados",
    "plugin_menu_no_plugins": "No se han cargado complementos",
    "plugin_menu_reload": "Recargar complementos",
    "plugin_menu_open_folder": "Abrir carpeta de complementos",
    "plugin_info_name": "Nombre: {name}",
    "plugin_info_version": "Version: {version}",
    "plugin_info_author": "Autor: {author}",
    "plugin_info_description": "Descripcion: {description}",

    # Sort menu
    "sort_menu_title": "Ordenar",
    "sort_by_name": "Por nombre",
    "sort_by_date": "Por fecha de modificacion",
    "sort_by_size": "Por tamano de archivo",
    "sort_ascending": "Ascendente",
    "sort_descending": "Descendente",

    # Filter menu
    "filter_menu_title": "Filtrar",
    "filter_by_rating": "Por calificacion",
    "filter_by_tag": "Por etiqueta",
    "filter_by_album": "Por album",
    "filter_favorites": "Solo favoritos",
    "filter_clear": "Limpiar filtro",

    # Tags & Albums
    "tag_album_title": "Etiquetas y Albumes",
    "tag_tab": "Etiquetas",
    "album_tab": "Albumes",
    "tag_create": "Crear etiqueta",
    "tag_rename": "Renombrar etiqueta",
    "tag_delete": "Eliminar etiqueta",
    "album_create": "Crear album",
    "album_rename": "Renombrar album",
    "album_delete": "Eliminar album",

    # Batch export
    "batch_export_title": "Exportacion por lotes",
    "batch_export_count": "{count} imagen(es) seleccionada(s)",
    "export_format": "Formato:",
    "export_quality": "Calidad:",
    "export_browse": "Examinar...",
    "export_save": "Guardar",
    "export_cancel": "Cancelar",
    "export_start": "Iniciar exportacion",

    # GIF / Video
    "gif_video_title": "Crear GIF / Video",
    "gif_video_format": "Formato:",
    "gif_video_fps": "FPS:",
    "gif_video_create": "Crear",

    # Bookmarks
    "bookmark_menu_title": "Marcadores",
    "bookmark_add": "Agregar marcador",
    "bookmark_remove": "Quitar marcador",
    "bookmark_manage": "Administrar marcadores",

    # Slideshow
    "slideshow_start": "Iniciar presentacion",
    "slideshow_stop": "Detener presentacion",
    "slideshow_interval": "Intervalo (segundos):",

    # Toast
    "toast_copied": "Copiado",
    "toast_pasted": "Pegado",
    "toast_deleted": "Eliminado",
    "toast_undo": "Deshacer",
    "toast_redo": "Rehacer",
}


class SpanishLanguagePlugin(ImervuePlugin):
    """Adds Spanish (Espanol) language support to Imervue."""

    plugin_name = "Spanish Language"
    plugin_version = "1.1.0"
    plugin_description = "Adds Spanish (Espanol) language support to Imervue."
    plugin_author = "Imervue"

    def on_plugin_loaded(self) -> None:
        language_wrapper.register_language(
            language_code="Spanish",
            display_name="Espanol",
            word_dict=_SPANISH_DICT,
        )
