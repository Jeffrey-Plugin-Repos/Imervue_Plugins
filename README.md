# Imervue Plugin Development Guide

## Overview

Imervue supports a plugin system that allows developers to extend the application with custom functionality. Plugins can add menu items, respond to image events, handle keyboard shortcuts, and more. In addition to developing your own plugins, you can also download ready-made plugins from the official GitHub repository, which serves as a centralized hub for browsing, downloading, updating, and sharing plugins within the community.

## Plugin GitHub Repository

The GitHub repository is the primary source for obtaining plugins. It allows you to explore available plugins, download their source code, keep up with updates, and contribute your own plugins. To install a plugin from GitHub, download or clone the repository, locate the plugin you want, copy its folder into your local `plugins/` directory, and restart Imervue. Each plugin must remain in its own folder and follow the correct structure.

## Quick Start

Create a folder inside the `plugins/` directory (next to the `Imervue/` package):

* plugins/
  * my_plugin/
    * \_\_init__.py
    * my_plugin.py


### Define your plugin class in `my_plugin.py`:

```python
from Imervue.plugin.plugin_base import ImervuePlugin

class MyPlugin(ImervuePlugin):
    plugin_name = "My Plugin"
    plugin_version = "1.0.0"
    plugin_description = "A short description of what this plugin does."
    plugin_author = "Your Name"

    def on_plugin_loaded(self):
        print(f"{self.plugin_name} loaded!")
```

### Register it in __init__.py:
```python
from my_plugin.my_plugin import MyPlugin

plugin_class = MyPlugin
```

Restart Imervue and the plugin will be automatically discovered and loaded.

## Plugin Structure
Each plugin must define the following class attributes: plugin_name (display name), plugin_version (version string such as "1.0.0"), plugin_description (short description), and plugin_author (author name or contact). Every plugin instance also has access to built-in properties including self.main_window (the ImervueMainWindow instance) and self.viewer (the GPUImageView instance), which provide access to application state and UI components.

## Available Hooks
Imervue provides several hooks that allow plugins to integrate into the application lifecycle and UI. The on_plugin_loaded() hook is called once after the plugin is initialized and is typically used for setup logic. The on_plugin_unloaded() hook is called during shutdown and should be used to clean up resources.

Menu-related hooks include on_build_menu_bar(menu_bar), which allows adding custom menu items to the main menu bar, and on_build_context_menu(menu, viewer), which allows adding right-click menu actions based on the current viewer state.

Image-related hooks include on_image_loaded(image_path, viewer) for when a single image is loaded, on_folder_opened(folder_path, image_paths, viewer) for when a folder is opened, on_image_switched(image_path, viewer) for navigation events, and on_image_deleted(deleted_paths, viewer) for handling deletions.

Input handling is done through on_key_press(key, modifiers, viewer), which can intercept keyboard events. Returning True consumes the event, while returning False allows default handling.

The on_app_closing(main_window) hook is called before the application exits and can be used to save state or perform final cleanup.

## Accessing Application State
The viewer object provides access to the current mode (tile_grid_mode or deep_zoom), image list (viewer.model.images), current index, selection state (selected_tiles), and zoom/pan information (zoom, dz_offset_x, dz_offset_y). The main_window object provides access to UI components such as the menu bar, filename label, file tree, and file system model.

## Plugin Discovery
At startup, Imervue scans the plugins/ directory and supports both package plugins and single-file plugins. A package plugin contains a folder with __init__.py (defining plugin_class) and one or more Python files. A single-file plugin is a standalone .py file that defines a subclass of ImervuePlugin.

## Internationalization (i18n)
Plugins can provide translations by overriding get_translations() and returning a dictionary of language mappings. These translations are merged into the global language system. Plugins can also register entirely new languages by calling language_wrapper.register_language() during on_plugin_loaded() and providing a full translation dictionary.

## Error Handling
All plugin hooks are wrapped in try/except blocks by the plugin manager. Errors are logged to the console but will not crash the application, making plugin development safer and easier to debug.

## Tips
Refer to plugins/example_plugin/ for a complete working example. Use print() statements for debugging output. Avoid blocking the main thread; use threading tools such as QThreadPool for heavy tasks. Store runtime state in the plugin instance, and use JSON files within the plugin directory for persistent data. Always prefer using the provided hooks and public APIs instead of modifying internal data structures directly.
