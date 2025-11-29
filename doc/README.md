# ВКР: Генерация персонализированных учебных конспектов

## Структура

```
doc/
├── src/          # LaTeX исходники
├── bib/          # Библиография
├── ref/          # Примеры и справочники
└── build/        # Скомпилированный PDF (gitignore)
```

## Компиляция

**VS Code:** LaTeX Workshop компилирует автоматически при сохранении.

**Командная строка:**
```bash
cd doc
latexmk
```

PDF создается в `build/Project_Plan_and_Review.pdf`.

## Требования

XeLaTeX, BibTeX, TeX Live
