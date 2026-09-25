# CLAUDE.md - Imervue_Plugins

Distribution repository for Imervue plugins. Imervue's downloader (`Imervue/plugin/plugin_downloader.py` in the Imervue repository) reads branch `main` of `Jeffrey-Plugin-Repos/Imervue_Plugins`: every top-level directory not starting with `.` is a category (`plugins/`, `languages/`), every directory inside it is a plugin, and only the files directly inside a plugin directory are downloaded. See `architecture.md`.

The source of truth is `D:\Codes\Imervue\plugins\`; changes are made and tested there and then mirrored here. The mirroring procedure and all rules (no AI attribution, flat files only) are in Imervue's `CLAUDE.md`, section "Mirror plugin changes to the distribution repo".

## README (keep current)

**The README set must stay in sync with the code.** This repo ships a 9-language README set: the
English `README.md` plus flat, top-level translations `README.zh-TW.md`, `README.zh-CN.md`,
`README.ja.md`, `README.ko.md`, `README.es.md`, `README.fr.md`, `README.de.md`, `README.pt-BR.md`
and `README.ru.md`. They are top-level **files**, not a `README/` subdirectory: the downloader treats
top-level directories as plugin categories (see below), so a subdirectory would be mis-detected, but
plain files are ignored — hence the flat `README.<lang>.md` scheme. Any user-facing change — the
repository layout, the download contract, the plugin/category set — updates `README.md` **and every
language variant in the same commit**, structure and content aligned, never one language ahead of the
others. No test guards this, so it is a manual check.

## Stage commits, `progress.md` and `architecture.md`

Workspace rule shared by every repository under `D:\Codes` (full text: `D:\Codes\CLAUDE.md`).

- **Commit at every stage**: one mirrored plugin change is one stage. Stage only the files it touched (never `git add -A`), never add AI attribution, and push to `main` as the mirroring procedure says.
- **Commit and push frequently; do not batch.** After each big feature — a self-contained stage that passes this repository's checks — commit and push to `main`; do not pile up a large batch of work before committing or pushing. Smaller batches collide less with other sessions, let CI catch problems earlier, and are easier to revert. Follow the mirroring procedure's branch flow.
- **SonarCloud / Codacy findings.** When a PR or commit fails a SonarCloud or Codacy check, look the findings up through their APIs instead of guessing. The keys are in environment variables: `SonarCloudToken` (SonarCloud, e.g. `curl -s -u "$SonarCloudToken:" "https://sonarcloud.io/api/issues/search?componentKeys=<key>&pullRequest=<n>&resolved=false"`) and `CODACY_PROJECT_TOKEN` (Codacy; for a public repository `https://app.codacy.com/api/v3/analysis/organizations/gh/<org>/repositories/<repo>/pull-requests/<n>/issues?status=new` also answers without a key). **Never reveal a key or any personal credential while doing so**: refer to the variables by name only, never echo or print their values, and never put them in files, commit messages, PR or issue text, logs, or any output that leaves the machine.
- **`progress.md`** holds outstanding work only.
- **No `docs/` directory here.** Any new top-level directory would show up as a plugin category in every Imervue downloader already shipped. The update log for this repository lives in Imervue's `docs/updates/` under the tag `#Imervue_Plugins`.
- **`architecture.md`** is the short overview of the repository layout and the download contract; update it in the same commit when the layout or contract changes.
