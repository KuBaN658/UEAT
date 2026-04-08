# Презентация ВКР (HTML + JS)

Файлы:
- `index.html` — слайды на Reveal.js.
- `theme.css` — локальные стили.
- `export-pdf.mjs` — экспорт слайдов в PDF через Playwright.

## Как открыть презентацию

Откройте `index.html` в браузере.

## Экспорт в PDF

В каталоге `doc/presentation`:

1. Установить зависимости:
   - `npm install`
2. Установить браузер для Playwright:
   - `npx playwright install chromium`
3. Сгенерировать PDF:
   - `npm run export:pdf`

После этого появится файл `diplomawork-presentation.pdf` (в текущей директории).

Канонический путь в репозитории (как в CI):

- `PDF_OUT=../../docs/presentation.pdf npm run export:pdf` (из каталога `doc/presentation`)

## Режим печати вручную (альтернатива)

Можно открыть:

- `index.html?print-pdf`

и сохранить через печать браузера в PDF с включенной печатью фона.
