# cse6242-group-project

Steps to create PDF using pandoc + weasyprint
```
pandoc team029proposal.md \
  --pdf-engine=weasyprint \
  --css=style.css \
  -o team029proposal.pdf
```