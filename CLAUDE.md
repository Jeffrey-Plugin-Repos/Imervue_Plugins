# CLAUDE.md - Imervue_Plugins

Distribution repository for Imervue plugins. Imervue's downloader (`Imervue/plugin/plugin_downloader.py` in the Imervue repository) reads branch `main` of `Jeffrey-Plugin-Repos/Imervue_Plugins`: every top-level directory not starting with `.` is a category (`plugins/`, `languages/`), every directory inside it is a plugin, and only the files directly inside a plugin directory are downloaded. See `architecture.md`.

The source of truth is `D:\Codes\Imervue\plugins\`; changes are made and tested there and then mirrored here. The mirroring procedure and all rules (no AI attribution, flat files only) are in Imervue's `CLAUDE.md`, section "Mirror plugin changes to the distribution repo".

## Stage commits, `progress.md` and `architecture.md`

Workspace rule shared by every repository under `D:\Codes` (full text: `D:\Codes\CLAUDE.md`).

- **Commit at every stage**: one mirrored plugin change is one stage. Stage only the files it touched (never `git add -A`), never add AI attribution, and push to `main` as the mirroring procedure says.
- **`progress.md`** holds outstanding work only.
- **No `docs/` directory here.** Any new top-level directory would show up as a plugin category in every Imervue downloader already shipped. The update log for this repository lives in Imervue's `docs/updates/` under the tag `#Imervue_Plugins`.
- **`architecture.md`** is the short overview of the repository layout and the download contract; update it in the same commit when the layout or contract changes.
