from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
import textwrap

INPUT_MD = 'report/Yelp_User_Review_Simulation_Report.md'
OUTPUT_PDF = 'report/Yelp_User_Review_Simulation_Report.pdf'

def split_paragraphs(text):
    # Split by blank lines
    parts = [p.strip() for p in text.split('\n\n') if p.strip()]
    return parts


def make_pdf():
    with open(INPUT_MD, 'r', encoding='utf-8') as f:
        md = f.read()

    doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=LETTER,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    normal = styles['BodyText']
    heading = styles['Heading1']
    heading2 = styles['Heading2']

    story = []

    # Title: first line
    lines = md.strip().split('\n')
    if lines:
        title = lines[0].strip()
        story.append(Paragraph(title, ParagraphStyle('Title', parent=styles['Title'], alignment=1)))
        story.append(Spacer(1, 12))

    paragraphs = split_paragraphs(md)
    for p in paragraphs[1:]:  # skip first line already added as title
        # Treat lines that end with ':' or all-caps start as headings heuristically
        first_line = p.split('\n',1)[0].strip()
        if first_line.endswith(':') or (first_line.isupper() and len(first_line) < 80):
            # heading
            story.append(Paragraph(first_line.rstrip(':'), heading2))
            rest = p[len(first_line):].strip()
            if rest:
                for para in textwrap.wrap(rest, 100):
                    story.append(Paragraph(para, normal))
                story.append(Spacer(1, 6))
        elif p.startswith('1.') or p.startswith('2.') or p.startswith('3.') or p.startswith('4.') or p.startswith('5.') or p.startswith('6.'):
            # section header detection
            # split header number and rest
            sec_title = first_line
            story.append(Paragraph(sec_title, heading2))
            rest = p[len(first_line):].strip()
            if rest:
                for para in textwrap.wrap(rest, 100):
                    story.append(Paragraph(para, normal))
            story.append(Spacer(1, 6))
        else:
            # normal paragraph
            for para in textwrap.wrap(p, 100):
                story.append(Paragraph(para, normal))
            story.append(Spacer(1, 6))

    doc.build(story)
    print(f'PDF written to {OUTPUT_PDF}')

if __name__ == '__main__':
    make_pdf()
