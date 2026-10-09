# data/ layout

One sub-folder per topic. The folder name IS the topic id (lowercase, underscores, no spaces).
Any mix of file types can sit in one topic folder. Sub-folders inside a topic are fine.

    data/
      <topic_a>/
        report.pdf        (text + tables + images)
        scan.pdf          (scanned, no text layer)
        sales.csv
        chart.png
        reviews.txt
      <topic_b>/
        ...

Supported: pdf, png, jpg, jpeg, webp, csv, xlsx, json, txt, md, html, docx, log
Anything else is skipped and logged.
