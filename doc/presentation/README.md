# Презентация ВКР (HTML + JS)

Файлы:

- `index.html` — слайды на Reveal.js; диаграммы подключаются как `diagrams/<name>.svg`.
- `diagrams/*.mmd` — исходники [Mermaid](https://mermaid.js.org/); SVG **не хранится в репозитории** и создаётся при экспорте PDF (или вручную: `npm run build:diagrams`).
- `theme.css` — локальные стили.
- `export-pdf.mjs` — перед PDF: `mmdc` для всех `.mmd` → `.svg`, затем Playwright печатает слайды.

## Как открыть презентацию

Откройте `index.html` в браузере. Чтобы подгрузились картинки диаграмм, один раз сгенерируйте SVG:

- `npm run build:diagrams` (в каталоге `doc/presentation`, после `npm install`).

## Экспорт в PDF

В каталоге `doc/presentation`:

1. `npm install`
2. `npx playwright install chromium` (или `npx playwright install --with-deps chromium` в CI)
3. `npm run export:pdf` — автоматически соберёт SVG из `.mmd` и запишет PDF.

По умолчанию файл: `diplomawork-presentation.pdf`. Канонический путь в репозитории (как в CI):

- `PDF_OUT=../../docs/presentation.pdf npm run export:pdf`

## Режим печати вручную

Сначала `npm run build:diagrams`, затем откройте `index.html?print-pdf` и сохраните через печать браузера с фоном.
