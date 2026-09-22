# Imervue_Plugins Architecture

> Short overview of Imervue's public plugin distribution repository. No `architecture_explore.md`
> here; plugin internals are mapped in the Imervue repo's `architecture_explore.md` §7.
>
> Last verified: 2026-09-22 against `13fd504` on `main`.

## 1. Purpose

This repository is what Imervue's in-app plugin downloader reads. It holds no code of its own:
each plugin directory is a manual mirror of `plugins/<name>/` in the Imervue repo
(`D:\Codes\Imervue`), where plugins are developed and tested.

## 2. Layout and consumer contract

| Path | Responsibility |
| --- | --- |
| `plugins/<name>/` | Feature plugins. `__init__.py` sets `plugin_class`; the class subclasses `Imervue.plugin.plugin_base.ImervuePlugin` |
| `languages/<name>/` | Language plugins (e.g. `spanish_translation/`) |
| `languages/__init__.py` | Plain file at category level; the downloader only descends into directories, so it is never fetched |
| `README.md` | Plugin development guide (structure, hooks, discovery, i18n) |
| `.idea/`, `.gitignore`, `LICENSE` | Tooling and licence; dot-directories are skipped by the downloader |

`Imervue/plugin/plugin_downloader.py` (in Imervue) lists
`https://api.github.com/repos/Jeffrey-Plugin-Repos/Imervue_Plugins/contents` on the default branch
(`main`), treats each top-level directory not starting with `.` as a category and each directory
inside it as one plugin, and downloads **only the files directly inside** `<category>/<plugin>/`.
Files land in `<app_dir>/plugins/<plugin>/` (the category is dropped) through a `.partial`
directory swapped in with `os.replace`; after a restart Imervue's `PluginManager` loads it.

## 3. Main flow

Edit and test `D:\Codes\Imervue\plugins\<name>\` → commit in Imervue (`git add -f`, since
`/plugins/` is gitignored there) → copy the directory into the matching category here (delete it
for a removed plugin) → commit and push to `main` → users pick it in Imervue's plugin downloader.

## 4. Extension points

- New feature plugin: `plugins/<name>/__init__.py` + the plugin module (convention
  `<name>_plugin.py`) + any pure-logic modules, all flat.
- New language pack: `languages/<name>/`, modelled on `languages/spanish_translation/`
  (`language_wrapper.register_language()` in its plugin class).
- Heavy dependencies are not vendored: Imervue's `Imervue/plugin/pip_installer.py` installs them at
  runtime, and model weights are discovered at runtime (`Imervue/plugin/model_dir.py`).

## 5. Cross-project boundaries and constraints

- Imervue is the source of truth; the rule is Imervue `CLAUDE.md` "Mirror plugin changes to the
  distribution repo". Never edit a plugin only here.
- Only `main` reaches users; a mirror committed only to `dev` is invisible to the downloader.
- Keep every runtime-required file flat; nested directories (`models/`, `assets/`) are never fetched.
- Plugins import `Imervue.*` modules at runtime, so they must match the Imervue version users run.
- The parity command in Imervue `CLAUDE.md` compares directory names only; diff file contents too
  when a plugin was edited rather than added.
- The no-attribution rule for commit messages in Imervue `CLAUDE.md` applies here as well.

## 6. When to update this file

When a category is added or removed, the downloader contract in `plugin_downloader.py` changes,
the mirroring procedure changes, or a plugin moves between categories. Refresh "Last verified".
