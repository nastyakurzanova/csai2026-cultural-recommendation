"""Russian single-column article populated from actual saved evaluation outputs."""
from pathlib import Path
import json,re
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH,WD_TAB_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
def mr(text):
    r=OxmlElement('m:r');t=OxmlElement('m:t');t.text=text;r.append(t);return r
def sub(a,b):
    s=OxmlElement('m:sSub');e=OxmlElement('m:e');e.append(mr(a));b1=OxmlElement('m:sub');b1.append(mr(b));s.extend([e,b1]);return s
def sup(a,b):
    s=OxmlElement('m:sSup');e=OxmlElement('m:e');e.append(mr(a));b1=OxmlElement('m:sup');b1.append(mr(b));s.extend([e,b1]);return s
def frac(a,b):
    f=OxmlElement('m:f');num=OxmlElement('m:num');den=OxmlElement('m:den')
    for x in a:num.append(x)
    for x in b:den.append(x)
    f.extend([num,den]);return f
doc=Document();sec=doc.sections[0];sec.page_width=Cm(21);sec.page_height=Cm(29.7)
sec.top_margin=sec.bottom_margin=Cm(1.8);sec.left_margin=sec.right_margin=Cm(2);sec.footer_distance=Cm(.8)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2']:
    st=doc.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0)
    f=st.element.get_or_add_rPr().rFonts
    for a in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:f.attrib.pop(qn('w:'+a),None)
    for a in ['ascii','hAnsi','eastAsia','cs']:f.set(qn('w:'+a),'Times New Roman')
for n in list(doc.styles.element.iter(qn('w:pBdr'))):n.getparent().remove(n)
normal=doc.styles['Normal'];normal.font.size=Pt(11.5);normal.paragraph_format.line_spacing=1.07;normal.paragraph_format.space_after=Pt(5);normal.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY;normal.paragraph_format.widow_control=True
doc.styles['Title'].font.size=Pt(18);doc.styles['Title'].font.bold=True;doc.styles['Title'].paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER
for name in ['Heading 1','Heading 2']:
    st=doc.styles[name];st.font.size=Pt(12);st.font.bold=name=='Heading 1';st.font.italic=name=='Heading 2';st.paragraph_format.space_before=Pt(8);st.paragraph_format.space_after=Pt(5)
def p(text,style=None):
    text=text.replace(' — ',' – ');text=re.sub(r'(?<=\d) (?=\d{3}(?:\D|$))','\u00a0',text)
    return doc.add_paragraph(text,style)
def h(t):doc.add_heading(t,1)
def page():doc.add_page_break()
def equation(parts,num):
    par=p('');par.paragraph_format.space_after=Pt(7);par.alignment=WD_ALIGN_PARAGRAPH.LEFT
    par.paragraph_format.tab_stops.add_tab_stop(Cm(8.5),WD_TAB_ALIGNMENT.CENTER);par.paragraph_format.tab_stops.add_tab_stop(Cm(17),WD_TAB_ALIGNMENT.RIGHT)
    par.add_run('\t');m=OxmlElement('m:oMath')
    for x in parts:m.append(x)
    par._p.append(m);par.add_run(f'\t({num})')
def table(cap,headers,rows,widths):
    q=p(cap);q.paragraph_format.keep_with_next=True
    for r in q.runs:r.font.size=Pt(10)
    t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    for i,w in enumerate(widths):t.columns[i].width=Cm(w)
    for i,x in enumerate(headers):t.rows[0].cells[i].text=str(x)
    for values in rows:
        for cell,x in zip(t.add_row().cells,values):cell.text=str(x)
    for ri,row in enumerate(t.rows):
        tr=row._tr.get_or_add_trPr();tr.append(OxmlElement('w:cantSplit'))
        if ri==0:tr.append(OxmlElement('w:tblHeader'))
        for ci,cell in enumerate(row.cells):
            cell.width=Cm(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tc=cell._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
            for edge in ['top','left','bottom','right']:
                x=OxmlElement('w:'+edge);x.set(qn('w:val'),'single' if edge=='bottom' else 'nil');x.set(qn('w:sz'),'4');x.set(qn('w:color'),'888888' if ri==0 else 'D9D9D9');b.append(x)
            tc.append(b)
            mar=OxmlElement('w:tcMar')
            for edge in ['top','left','bottom','right']:
                x=OxmlElement('w:'+edge);x.set(qn('w:w'),'60');x.set(qn('w:type'),'dxa');mar.append(x)
            tc.append(mar)
            for q in cell.paragraphs:
                q.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 else WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.space_after=Pt(0);q.paragraph_format.line_spacing=1
                for r in q.runs:r.font.size=Pt(10);r.bold=ri==0
    p('')
def inline(par,parts):
    m=OxmlElement('m:oMath')
    for x in parts:m.append(x)
    par._p.append(m)

refs=[
'[1] Konstantakis, M., Alexandridis, G., Caridakis, G.: “A Personalized Heritage-Oriented Recommender System Based on Extended Cultural Tourist Typologies”, Big Data and Cognitive Computing, 2020, 4 (2), 12. DOI: 10.3390/bdcc4020012.',
'[2] Gao, J., Peng, P., Lu, F., et al.: “Towards travel recommendation interpretability: Disentangling tourist decision-making process via knowledge graph”, Information Processing & Management, 2023, 60 (4), 103369. DOI: 10.1016/j.ipm.2023.103369.',
'[3] Fan, Z., Chen, C.: “CuPe-KG: Cultural perspective–based knowledge graph construction of tourism resources via pretrained language models”, Information Processing & Management, 2024, 61 (3), 103646. DOI: 10.1016/j.ipm.2024.103646.',
'[4] Cao, Y., Wang, X., He, X., et al.: “Unifying Knowledge Graph Learning and Recommendation: Towards a Better Understanding of User Preferences”. The World Wide Web Conference, 2019. DOI: 10.1145/3308558.3313705.',
'[5] Ferrari Dacrema, M., Cremonesi, P., Jannach, D.: “Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches”. Proc. ACM RecSys, 2019. DOI: 10.1145/3298689.3347058.',
'[6] Lim, K.H., Chan, J., Leckie, C., et al.: “Personalized Tour Recommendation Based on User Interests and Points of Interest Visit Durations”. Proc. IJCAI, 2015, pp. 1778–1783. https://ijcai.org/proceedings/2015/253.',
'[7] Vrandečić, D., Krötzsch, M.: “Wikidata: a free collaborative knowledgebase”, Communications of the ACM, 2014, 57 (10), pp. 78–85. DOI: 10.1145/2629489.',
'[8] Wikidata: WikiProject Built heritage Data structure. https://www.wikidata.org/wiki/Wikidata:WikiProject_Built_heritage/Data_structure (дата обращения: 06.11.2026).',
'[9] National Museums Scotland. Architecture trail. https://www.nms.ac.uk/national-museum-of-scotland/see-and-do/tours-and-trails/architecture-trail (дата обращения: 06.11.2026).',
'[10] Tong, H., Faloutsos, C., Pan, J.-Y.: “Fast Random Walk with Restart and Its Applications”. Proc. IEEE ICDM, 2006, pp. 613–622. DOI: 10.1109/ICDM.2006.70.',
'[11] Steck, H.: “Embarrassingly Shallow Autoencoders for Sparse Data”. The World Wide Web Conference, 2019, pp. 3251–3257. DOI: 10.1145/3308558.3313710.'
]
