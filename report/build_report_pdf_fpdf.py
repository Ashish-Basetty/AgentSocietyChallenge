from fpdf import FPDF

INPUT_MD = 'report/Yelp_User_Review_Simulation_Report.md'
OUTPUT_PDF = 'report/Yelp_User_Review_Simulation_Report.pdf'

class PDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def wrap_text(text, width=90):
    import textwrap
    lines = []
    for para in text.split('\n'):
        if not para.strip():
            lines.append('')
            continue
        wrapped = textwrap.wrap(para, width)
        if not wrapped:
            lines.append('')
        else:
            lines.extend(wrapped)
    return lines


def make_pdf():
    with open(INPUT_MD, 'r', encoding='utf-8') as f:
        md = f.read()

    pdf = PDF(format='letter')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    lines = md.strip().split('\n')
    if lines:
        title = lines[0].strip()
        pdf.set_font('Helvetica', 'B', 16)
        pdf.multi_cell(0, 8, title, align='C')
        pdf.ln(4)

    pdf.set_font('Helvetica', '', 11)

    paragraphs = [p.strip() for p in md.split('\n\n') if p.strip()]
    for p in paragraphs[1:]:
        # simple section detection: lines starting with a number and a dot
        first_line = p.split('\n',1)[0].strip()
        if first_line and (first_line[0].isdigit() and first_line[1:2] == '.'):
            # heading
            pdf.set_font('Helvetica', 'B', 12)
            pdf.multi_cell(0, 7, first_line)
            pdf.ln(1)
            rest = p[len(first_line):].strip()
            if rest:
                pdf.set_font('Helvetica', '', 11)
                for line in wrap_text(rest, 98):
                    pdf.multi_cell(0, 6, line)
                pdf.ln(2)
        else:
            # normal paragraph
            pdf.set_font('Helvetica', '', 11)
            for line in wrap_text(p, 98):
                pdf.multi_cell(0, 6, line)
            pdf.ln(2)

    pdf.output(OUTPUT_PDF)
    print(f'PDF written to {OUTPUT_PDF}')

if __name__ == '__main__':
    make_pdf()
