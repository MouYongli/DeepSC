#!/usr/bin/env python3
"""
Convert Markdown to PDF with embedded images
Handles HTML tables with embedded markdown images
"""

import os
from pathlib import Path

from weasyprint import CSS, HTML

import re
import sys


def process_markdown_with_images(md_content, base_dir):
    """Process markdown and convert image paths to HTML img tags"""

    # First, handle images in HTML tables by converting markdown syntax to HTML
    # Match: ![alt text](path)
    def replace_md_image(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        # Make path absolute if relative
        if not img_path.startswith("http"):
            full_path = os.path.join(base_dir, img_path)
            img_path = f"file://{os.path.abspath(full_path)}"
        return f'<img src="{img_path}" alt="{alt_text}" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px;">'

    # Replace markdown image syntax with HTML
    md_content = re.sub(r"!\[([^\]]*)\]\(([^\)]+)\)", replace_md_image, md_content)

    return md_content


def md_to_html(md_content):
    """Convert markdown to HTML, handling tables and basic formatting"""
    html_lines = []
    in_table = False
    in_code_block = False

    for line in md_content.split("\n"):
        # Handle code blocks
        if line.strip().startswith("```"):
            if in_code_block:
                html_lines.append("</pre></code>")
                in_code_block = False
            else:
                html_lines.append("<code><pre>")
                in_code_block = True
            continue

        if in_code_block:
            html_lines.append(line)
            continue

        # Handle headers
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("#### "):
            html_lines.append(f"<h4>{line[5:]}</h4>")

        # Handle horizontal rules
        elif line.strip() == "---":
            html_lines.append("<hr>")

        # Handle bold and inline formatting
        elif line.strip():
            # Bold
            line = re.sub(r"\*\*([^\*]+)\*\*", r"<strong>\1</strong>", line)
            # Inline code
            line = re.sub(r"`([^`]+)`", r"<code>\1</code>", line)
            # Check if it's a list item
            if line.strip().startswith("- "):
                if not html_lines or not html_lines[-1].startswith("<li>"):
                    html_lines.append("<ul>")
                html_lines.append(f"<li>{line.strip()[2:]}</li>")
            elif (
                html_lines
                and html_lines[-1].startswith("<li>")
                and not line.strip().startswith("- ")
            ):
                html_lines.append("</ul>")
                html_lines.append(f"<p>{line}</p>")
            # Check if it's a table
            elif "|" in line and not line.strip().startswith("<"):
                html_lines.append(line)  # Keep table syntax as-is
            else:
                html_lines.append(f"<p>{line}</p>")
        else:
            # Close open lists
            if html_lines and html_lines[-1].startswith("<li>"):
                html_lines.append("</ul>")
            html_lines.append("<br>")

    return "\n".join(html_lines)


def parse_table(md_content):
    """Parse markdown tables and convert to HTML"""
    lines = md_content.split("\n")
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this is a table line
        if "|" in line and line.strip().startswith("|"):
            # Start of table
            table_lines = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1

            # Convert table
            result.append("<table>")
            for idx, tline in enumerate(table_lines):
                if idx == 1 and re.match(r"\|[\s\-:]+\|", tline):
                    # This is the separator line, skip it
                    continue

                cells = [cell.strip() for cell in tline.split("|")[1:-1]]

                if idx == 0:
                    result.append("<thead><tr>")
                    for cell in cells:
                        result.append(f"<th>{cell}</th>")
                    result.append("</tr></thead><tbody>")
                else:
                    result.append("<tr>")
                    for cell in cells:
                        result.append(f"<td>{cell}</td>")
                    result.append("</tr>")

            result.append("</tbody></table>")
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


def md_to_pdf(md_file, output_pdf=None):
    """Convert markdown file to PDF"""

    if output_pdf is None:
        output_pdf = md_file.replace(".md", ".pdf")

    # Read markdown content
    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Get base directory for image paths
    base_dir = Path(md_file).parent

    # Process images first
    md_content = process_markdown_with_images(md_content, base_dir)

    # Parse tables
    md_content = parse_table(md_content)

    # Convert remaining markdown to HTML
    html_content = md_to_html(md_content)

    # Wrap in HTML template with styling
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                font-size: 11pt;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-top: 30px;
                page-break-before: auto;
                font-size: 24pt;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 2px solid #95a5a6;
                padding-bottom: 8px;
                margin-top: 25px;
                page-break-after: avoid;
                font-size: 18pt;
            }}
            h3 {{
                color: #7f8c8d;
                margin-top: 20px;
                page-break-after: avoid;
                font-size: 14pt;
            }}
            h4 {{
                color: #95a5a6;
                margin-top: 15px;
                font-size: 12pt;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                font-size: 10pt;
                page-break-inside: avoid;
            }}
            th {{
                background-color: #3498db;
                color: white;
                padding: 8px;
                text-align: left;
                font-weight: bold;
            }}
            td {{
                padding: 6px 8px;
                border: 1px solid #ddd;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin: 10px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
                display: block;
                page-break-inside: avoid;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 9pt;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
                border-left: 4px solid #3498db;
                page-break-inside: avoid;
                font-size: 9pt;
                white-space: pre-wrap;
            }}
            strong {{
                color: #2c3e50;
                font-weight: bold;
            }}
            hr {{
                border: none;
                border-top: 2px solid #ecf0f1;
                margin: 20px 0;
            }}
            ul {{
                margin: 10px 0;
                padding-left: 30px;
            }}
            li {{
                margin: 5px 0;
            }}
            p {{
                margin: 10px 0;
            }}
            /* For side-by-side image comparison */
            table td img {{
                width: 95%;
                margin: 5px auto;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Convert HTML to PDF
    HTML(string=html_template, base_url=str(base_dir)).write_pdf(
        output_pdf, stylesheets=[CSS(string="@page { size: A4; margin: 1.5cm; }")]
    )

    print(f"PDF created: {output_pdf}")
    return output_pdf


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python md_to_pdf_fixed.py <markdown_file> [output_pdf]")
        sys.exit(1)

    md_file = sys.argv[1]
    output_pdf = sys.argv[2] if len(sys.argv) > 2 else None

    md_to_pdf(md_file, output_pdf)
