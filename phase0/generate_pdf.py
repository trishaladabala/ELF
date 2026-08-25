#!/usr/bin/env python3
"""Convert walkthrough.md to a styled PDF with embedded images."""

import re
import markdown
from pathlib import Path
from weasyprint import HTML

WALKTHROUGH = Path("/Users/adabalatrishal/.gemini/antigravity-ide/brain/2858da56-e430-48cf-9ecb-c1a66ae8d737/walkthrough.md")
OUTPUT_PDF = Path("/Users/adabalatrishal/7semminor/ELF/phase0/results/Phase0_Diagnostic_Results.pdf")

# Read the markdown
md_text = WALKTHROUGH.read_text()

# Convert file:///... image links to local absolute paths for weasyprint
# Pattern: ![caption](file:///path/to/image.png)
md_text = re.sub(r'!\[([^\]]*)\]\(file:///([^)]+)\)', r'![\1](/\2)', md_text)

# Also convert file:// links in regular markdown links to just text (non-images)
# Keep the link text but remove the file:// URL since PDFs can't have clickable file links
md_text = re.sub(r'\[([^\]]+)\]\(file:///[^)]+\)', r'**\1**', md_text)

# Convert GitHub-style alerts to styled HTML divs before markdown processing
alert_map = {
    'NOTE': ('#e8f4fd', '#1a73e8', 'ℹ️'),
    'TIP': ('#e6f4ea', '#1e8e3e', '💡'),
    'IMPORTANT': ('#fce8e6', '#d93025', '❗'),
    'WARNING': ('#fef7e0', '#f9ab00', '⚠️'),
    'CAUTION': ('#fce8e6', '#d93025', '🔴'),
}

def replace_alerts(text):
    """Replace GitHub-style > [!TYPE] alerts with styled HTML."""
    lines = text.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check for alert start
        alert_match = re.match(r'>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]', line)
        if alert_match:
            alert_type = alert_match.group(1)
            bg_color, border_color, icon = alert_map[alert_type]
            # Collect all lines of the alert (lines starting with >)
            alert_lines = []
            i += 1
            while i < len(lines) and lines[i].startswith('>'):
                content = lines[i].lstrip('> ').strip()
                if content:
                    alert_lines.append(content)
                i += 1
            alert_content = ' '.join(alert_lines)
            result.append(f'<div class="alert alert-{alert_type.lower()}">')
            result.append(f'<strong>{icon} {alert_type}</strong><br/>')
            result.append(f'{alert_content}')
            result.append('</div>')
            result.append('')
        else:
            result.append(line)
            i += 1
    return '\n'.join(result)

md_text = replace_alerts(md_text)

# Convert markdown to HTML
html_body = markdown.markdown(
    md_text,
    extensions=['tables', 'fenced_code', 'codehilite', 'toc'],
)

# Build full HTML with professional styling
html_full = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  @page {{
    size: A4;
    margin: 2cm 2.5cm;
    @top-center {{
      content: "Phase 0 Diagnostic Test Results — ELF ASBM Coupling Analysis";
      font-size: 8pt;
      color: #666;
    }}
    @bottom-center {{
      content: "Page " counter(page) " of " counter(pages);
      font-size: 8pt;
      color: #666;
    }}
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.6;
    color: #1a1a2e;
    max-width: 100%;
  }}

  h1 {{
    font-size: 20pt;
    color: #0f172a;
    border-bottom: 3px solid #3b82f6;
    padding-bottom: 8px;
    margin-top: 0;
  }}

  h2 {{
    font-size: 15pt;
    color: #1e3a5f;
    border-bottom: 1.5px solid #cbd5e1;
    padding-bottom: 5px;
    margin-top: 28px;
    page-break-after: avoid;
  }}

  h3 {{
    font-size: 12pt;
    color: #334155;
    margin-top: 20px;
    page-break-after: avoid;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
  }}

  th {{
    background-color: #1e3a5f;
    color: white;
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
  }}

  td {{
    padding: 7px 10px;
    border-bottom: 1px solid #e2e8f0;
  }}

  tr:nth-child(even) {{
    background-color: #f8fafc;
  }}

  tr:hover {{
    background-color: #eff6ff;
  }}

  img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 12px auto;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    page-break-inside: avoid;
  }}

  code {{
    background-color: #f1f5f9;
    padding: 2px 5px;
    border-radius: 3px;
    font-size: 9.5pt;
    font-family: "SF Mono", "Fira Code", Menlo, monospace;
  }}

  blockquote {{
    border-left: 4px solid #3b82f6;
    margin: 16px 0;
    padding: 10px 16px;
    background-color: #eff6ff;
    border-radius: 0 6px 6px 0;
    color: #1e40af;
  }}

  blockquote strong {{
    color: #1e3a5f;
  }}

  hr {{
    border: none;
    border-top: 2px solid #e2e8f0;
    margin: 24px 0;
  }}

  strong {{
    color: #0f172a;
  }}

  .alert {{
    padding: 12px 16px;
    border-radius: 6px;
    margin: 14px 0;
    border-left: 4px solid;
    page-break-inside: avoid;
  }}

  .alert-note {{
    background-color: #e8f4fd;
    border-color: #1a73e8;
    color: #174ea6;
  }}

  .alert-tip {{
    background-color: #e6f4ea;
    border-color: #1e8e3e;
    color: #137333;
  }}

  .alert-important {{
    background-color: #fce8e6;
    border-color: #d93025;
    color: #c5221f;
  }}

  .alert-warning {{
    background-color: #fef7e0;
    border-color: #f9ab00;
    color: #b06000;
  }}

  .alert-caution {{
    background-color: #fce8e6;
    border-color: #d93025;
    color: #c5221f;
  }}

  /* Keep headings with following content */
  h2, h3, h4 {{
    page-break-after: avoid;
  }}

  /* Avoid breaking inside figures */
  figure, .figure {{
    page-break-inside: avoid;
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

# Generate PDF
print(f"Generating PDF...")
HTML(string=html_full).write_pdf(str(OUTPUT_PDF))
print(f"✅ PDF saved to: {OUTPUT_PDF}")
