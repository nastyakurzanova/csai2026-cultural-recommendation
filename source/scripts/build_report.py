from pathlib import Path
import json,pandas as pd
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports'
CITIES={'Buda':'Будапешт','Delh':'Дели','Edin':'Эдинбург','Glas':'Глазго','Osak':'Осака','Pert':'Перт','Toro':'Торонто','Vien':'Вена'}
audit=pd.read_csv(ROOT/'processed/city_audit.csv');ready=pd.read_csv(ROOT/'processed/data_readiness.csv')
summary=json.loads((OUT/'preparation_summary.json').read_text())
doc=Document();sec=doc.sections[0]
sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.top_margin=sec.bottom_margin=Inches(.65);sec.left_margin=sec.right_margin=Inches(.7)
for n in ['Normal','Title','Subtitle','Heading 1','Heading 2']:
    st=doc.styles[n];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0)
    st.element.rPr.rFonts.set(qn('w:eastAsia'),'Times New Roman')
doc.styles['Normal'].font.size=Pt(11)
doc.styles['Normal'].paragraph_format.space_after=Pt(5)
doc.styles['Normal'].paragraph_format.line_spacing=1.05
doc.styles['Title'].font.size=Pt(20);doc.styles['Title'].paragraph_format.space_after=Pt(8)
doc.styles['Heading 1'].font.size=Pt(14);doc.styles['Heading 1'].paragraph_format.space_before=Pt(9)
doc.styles['Heading 1'].paragraph_format.space_after=Pt(5)
doc.styles['Heading 2'].font.size=Pt(12)
def p(text,bold=False):
    x=doc.add_paragraph();x.add_run(text).bold=bold;return x
def table(headers,rows,widths):
    t=doc.add_table(rows=1, cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    for i,w in enumerate(widths):t.columns[i].width=Inches(w)
    for i,h in enumerate(headers):t.rows[0].cells[i].text=str(h)
    tr=t.rows[0]._tr.get_or_add_trPr();repeat=OxmlElement('w:tblHeader');tr.append(repeat)
    for row in rows:
        cells=t.add_row().cells
        for i,value in enumerate(row):cells[i].text=str(value)
    for ri,row in enumerate(t.rows):
        pr=row._tr.get_or_add_trPr();pr.append(OxmlElement('w:cantSplit'))
        for ci,cell in enumerate(row.cells):
            cell.width=Inches(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcPr=cell._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
            for edge in ['top','left','bottom','right']:
                e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
            tcPr.append(borders)
            shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E8EEF4' if ri==0 else ('F7F9FB' if ri%2==0 else 'FFFFFF'));tcPr.append(shade)
            mar=OxmlElement('w:tcMar')
            for e,v in [('top','75'),('bottom','75'),('left','85'),('right','85')]:
                x=OxmlElement('w:'+e);x.set(qn('w:w'),v);x.set(qn('w:type'),'dxa');mar.append(x)
            tcPr.append(mar)
            for par in cell.paragraphs:
                par.paragraph_format.space_after=Pt(0);par.paragraph_format.line_spacing=1
                par.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 else WD_ALIGN_PARAGRAPH.CENTER
                for run in par.runs:run.font.size=Pt(10.5);run.bold=ri==0
    doc.add_paragraph().paragraph_format.space_after=Pt(0)
    return t

doc.add_paragraph('Подготовка данных для исследования',style='Title')
p('Этап 6   |   Проверка Flickr и сопоставление с Wikidata   |   26 октября 2026 года')
p('Тема статьи: «Адаптивная рекомендация объектов культурного наследия на основе графа знаний и динамики пользовательских предпочтений».')
p('Решение по данным',True)
p('Для основного эксперимента предварительно выбраны Будапешт и Вена. Эдинбург подходит для дополнительной проверки после уточнения неоднозначных объектов. Подготовлены исходные снимки, таблицы посещений, соответствия Wikidata и журнал проблем. Итоговая культурная выборка ещё не утверждена; обучение и проверка гипотез не проводились.')
doc.add_heading('Исходный набор Flickr',1)
p('Загружен архив автора data-ijcai15.zip [1]. В нём восемь городов и 153 208 строк фотографий. Каталог содержит 236 исходных записей мест. Числа ниже рассчитаны по полученным файлам, а не перенесены из публикаций.')
table(['Город','Фото\nстрок','Пользо-\nватели','Места\nв каталоге','Последо-\nвательности'],[[CITIES[r.city],f'{r.raw_photo_rows:,}'.replace(',',' '),r.users,r.catalog_pois,r.sequences] for r in audit.itertuples()],[1.4,1.2,1.1,1.1,1.8])
p('В таблице пользователи и последовательности посчитаны после отделения 53 записей с отсутствующей либо непригодной временной меткой. Число фотографий и размер каталога указаны до очистки.')
doc.add_heading('Что обнаружено',1)
p('Одинаковые название и координаты имеют пары Осака 8 и 13 (Osaka Castle), Вена 7 и 12 (Belvedere). Для подготовки созданы канонические идентификаторы и сохранена таблица исходных соответствий. В этих двух парах фотографии фактически относятся только к одному исходному идентификатору каждой пары.')
p('У объекта Осака 26 координаты находятся примерно в 404 км от медианы каталога города. Встречаются невозможные для источника 2015 года даты, включая 4500 год. Ошибочные координаты и даты не заменялись догадками.')

doc.add_page_break();doc.add_heading('Правила подготовки посещений',1)
for title,text in [
('Сохранение происхождения.','Исходный ZIP сохранён без изменений; записаны размер, SHA-256 и время получения. Первичные таблицы доступны отдельно от производных.'),
('Обработка времени.','53 строки с метками вне диапазона 1970–2015 годов отделены в файлы quarantine. Это операционное правило подготовки поведенческой истории, а не утверждение о невозможности более ранних фотографий. Ещё 423 строки до 2000 года сохранены и отмечены для проверки.'),
('Единица взаимодействия.','Одно посещение — непрерывная серия фотографий одного объекта внутри исходной последовательности пользователя после сортировки по времени. Возвращение к объекту после другого места сохраняется как новое посещение. Получено 33 778 посещений из 153 155 оставшихся фотографий.'),
('Неопределённый порядок.','Если у одного пользователя в одной последовательности разные места имеют одинаковое время фотографии, порядок не восстанавливается догадкой. Для предварительного отбора исключена целиком такая последовательность. Всего обнаружено 127 последовательностей с такой неоднозначностью.'),
('Проверка истории.','Для предварительного отбора также исключены последовательности с датами до 2000 года, требующими проверки. Всего 114 таких последовательностей; они могут пересекаться с предыдущей группой. Диагностический отбор проходит 20 926 исходных последовательностей.'),
('Защита будущего эксперимента.','Готовый исходный показатель poiFreq не перенесён в очищенные фотографии. Популярность и любые поведенческие признаки будут рассчитываться только по обучающей истории. Обучающее и тестовое разделение на этом этапе не создавалось.')]:
    par=doc.add_paragraph();par.add_run(title+' ').bold=True;par.add_run(text)
doc.add_heading('Сопоставление с Wikidata',1)
p('Для всех 236 исходных названий получены сущности-кандидаты через английские статьи Wikipedia и официальные перенаправления. После объединения точных дубликатов остаётся 234 места. Для 230 из них координаты согласуются в пределах 2 км; этот порог служит только фильтром грубых ошибок и не доказывает тождество объектов.')
p('Три ссылки ведут к страницам неоднозначности: Sansad Bhavan, Gallery of Modern Art и Kyōbashi. Zepp сопоставляется с сетью площадок, а не с надёжно определённой исторической площадкой. Эти случаи не допущены в предварительную культурную выборку.')
p('National Museum of Scotland и Royal Museum в Эдинбурге ведут к одной сущности Wikidata. Записи не объединены автоматически: требуется различить музей, здание и историческое название. Обе временно исключены из диагностики культурной истории.')
p('Из Wikidata сохранены 1 239 утверждений по десяти свойствам вместе с идентификаторами утверждений, редакций, квалификаторами и числом ссылок на источники. Это сырьё для следующего этапа, а не готовый проверенный граф.')

doc.add_page_break();doc.add_heading('Достаточность истории для двух профилей',1)
p('Для диагностики использован консервативный предварительный отбор: надёжный порядок времени, согласованная сущность и исходная культурная категория. Парк, мост или здание другой категории могут иметь культурное значение; такие объекты нельзя окончательно исключать только по категории.')
p('Диагностическое условие H2: не менее трёх более ранних культурных посещений в предыдущих последовательностях и не менее трёх в текущей. После двух посещений текущей последовательности остаётся хотя бы одна позиция для предсказания. Это проверка наличия данных, а не утверждённый протокол оценки или признак изменения интересов.')
table(['Город','Пользователи\nдля диагностики','Последова-\nтельности','Позиции\nпредсказания'],[[CITIES[r.city],r.h2_diagnostic_users,r.h2_diagnostic_sequences,r.h2_diagnostic_prediction_positions] for r in ready.itertuples()],[1.5,1.8,1.6,1.7])
p('Для Будапешта и Вены доступно по 100 и 107 пользователей с такой историей. Эдинбург после исключения неоднозначной пары музеев даёт 33 пользователя. Значения изменятся после предметной проверки объектов, выбора временных границ и разбиения данных. Короткие истории сохраняются для отдельных сценариев.')
doc.add_heading('Что предстоит уточнить при построении графа',1)
p('Проверить культурную принадлежность каждого места, разделить организацию и здание, уточнить спорные соответствия. Выбрать содержательные связи: архитектурный стиль, архитектор, историческая личность, событие. Общая категория или охранный статус сами по себе не заменяют культурную связь.')
p('Снимок Wikidata получен в 2026 году, а посещения существенно старше. Перед экспериментом нужно отобрать сведения, применимые к историческому моменту: поздние события, переезды и современные статусы нельзя автоматически использовать как доступные в прошлом признаки. По последовательностям фотографий нельзя установить истинные мотивы путешественника.')
doc.add_heading('Материалы и воспроизведение',1)
p('Архив содержит исходные данные, очищенные таблицы, предварительную культурную выборку, соответствия Wikidata, исходные ответы API и скрипты. README описывает порядок запуска и ограничения. Сводные значения проверены на согласованность числа фотографий, посещений и уникальности канонических идентификаторов.')
p('[1] Kwan Hui Lim. Datasets. Flickr User–POI Visits. https://sites.google.com/site/limkwanhui/datacode')
p('[2] Wikidata. WikiProject Built heritage Data structure. https://www.wikidata.org/wiki/Wikidata:WikiProject_Built_heritage/Data_structure')
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
run=footer.add_run();fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');run._r.addnext(fld)
doc.core_properties.title='Подготовка данных для исследования культурных рекомендаций'
doc.core_properties.subject='Аудит Flickr и сопоставление с Wikidata'
doc.core_properties.author=''
for node in list(doc.styles.element.iter(qn('w:pBdr'))):node.getparent().remove(node)
for n in ['Normal','Title','Subtitle','Heading 1','Heading 2']:
    fonts=doc.styles[n].element.rPr.rFonts
    for attr in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
        fonts.attrib.pop(qn('w:'+attr),None)
    fonts.set(qn('w:ascii'),'Times New Roman');fonts.set(qn('w:hAnsi'),'Times New Roman')
out=OUT/'Etap6_Podgotovka_dannyh.docx';doc.save(out);print(out)
