# Imervue_Plugins Architecture

> Short overview of Imervue's public plugin distribution repository. No `architecture_explore.md`
> here; plugin internals are mapped in the Imervue repo's `architecture_explore.md` §7.
>
> Last verified: 2026-10-07 on `main`; corresponding commit: `git log -1 -- architecture.md`.

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
| `README.md` | Plugin list and install notes; the development guide itself is Imervue's `PLUGIN_DEV_GUIDE.md` |
| `.gitignore`, `LICENSE`, `CLAUDE.md`, `architecture.md`, `progress.md` | Repository files; the downloader only looks inside `plugins/` and `languages/` |

`Imervue/plugin/plugin_downloader.py` (in Imervue) lists `main` with one recursive git-tree call
(`/git/trees/main?recursive=1`), accepts only the categories `plugins` and `languages`, treats each
directory inside them as one plugin, and downloads **only the files directly inside**
`<category>/<plugin>/` from raw.githubusercontent. Imervue releases up to 1.0.90 still treat every
top-level directory not starting with `.` as a category, so no other top-level directory may be
added while those versions are in use.
Files land in `<app_dir>/plugins/<plugin>/` (the category is dropped). Current Imervue uses
unique sibling staging and backup/rollback, preserves models/assets, and serializes same-target
installs. After download, reload each window or restart; older consumers used `.partial` staging.

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

## 5. When to update this file

When a category is added or removed, the downloader contract in `plugin_downloader.py` changes,
the mirroring procedure changes, or a plugin moves between categories. Refresh "Last verified".

## 6. Cross-project boundaries and constraints

- Imervue is the source of truth; the rule is Imervue `CLAUDE.md` "Mirror plugin changes to the
  distribution repo". Never edit a plugin only here.
- Only `main` reaches users; a mirror committed only to `dev` is invisible to the downloader.
- Keep every runtime-required file flat; nested directories (`models/`, `assets/`) are never fetched.
- Plugins import `Imervue.*` modules at runtime, so they must match the Imervue version users run.
  Shared helpers come from there instead of being copied into each plugin (`load_rgba`,
  `_find_python`, `_subprocess_kwargs`; listed in Imervue `architecture.md` §6).
  `pet_integrations` also needs the desktop pet's plugin surface (the `on_pet_created` hook,
  `IntegrationController`, the pet's `add_integration`), also listed there; on an older Imervue it
  loads but the pet never gets its integrations.
- A standalone runner script (`safety_review/_runner.py` and `finetune.py`,
  `object_splitter/_runner.py`) runs in an external Python that cannot import the plugin package, so
  it loads flat sibling modules instead: `_constants.py`, `_censor_core.py`, `_components.py`. Keep
  those free of Qt and `Imervue` imports, and import third-party packages only inside functions.
- The parity command in Imervue `CLAUDE.md` compares every flat file's content (line endings
  ignored); no output means the mirror is in sync.
- The no-attribution rule for commit messages in Imervue `CLAUDE.md` applies here as well.

GPU Develop requires plugin API 3 for observed backend status and provider-specific unregister. Reloading one window preserves providers held by another generation/window; imports stay optional and heavy dependencies remain behind the install gate.


GPU Develop keeps optional wgpu and discrete-device ownership in the flat plugin, with no model
weights. New Imervue Batch Export dialogs default to CPU; GPU requires explicit selection. Enabled
threshold/posterize render the entire recipe on the CPU reference because discontinuous stages
amplify small GPU rounding errors. Other mixed-color GPU recipes remain approximate; Modify
previews and exact CPU exports keep the core uint8 sRGB pipeline. No new core API is required.
