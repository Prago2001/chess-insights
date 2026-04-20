#### Command to create .pdf

```
pandoc team029report.md \
  --pdf-engine=weasyprint \
  --css=style.css \
  -o team029report.pdf
```