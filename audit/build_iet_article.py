"""Russian manuscript using the supplied IET Word template and audited results."""
from pathlib import Path
import argparse
import json
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'results'
FOLLOW=ROOT/'followup/results'
TITLE='Адаптивная рекомендация объектов культурного наследия на основе графа знаний и динамики пользовательских предпочтений'
CITY={'Buda':'Будапешт','Vien':'Вена','Edin':'Эдинбург','Delh':'Дели','Glas':'Глазго','Osak':'Осака','Pert':'Перт','Toro':'Торонто'}

REFS=[
 'Савченко А.П., Никифорова Т.Д.: «Архитектура онтологического хранилища цифрового культурно-исторического наследия». Историческая информатика, 2026, № 3, с. 225–238. DOI: 10.7256/2585-7797.2026.3.81920.',
 'Антопольский А.Б.: «О создании русскоязычной онтологии по цифровой гуманитаристике». Научно-техническая информация. Сер. 2. Информационные процессы и системы, 2023, № 2, с. 22–32. DOI: 10.31857/S160578800025499-4.',
 'Атаева О.М., Серебряков В.А., Тучкова Н.П.: «О представлении результатов научного института в виде графа знаний в семантической библиотеке». Электронные библиотеки, 2024, 27 (6), с. 798–818. DOI: 10.26907/1562-5419-2024-27-6-798-818.',
 'Кутянин А.Р.: «Рекомендательные системы: обзор основных постановок и результатов». Интеллектуальные системы. Теория и приложения, 2017, 21 (4), с. 18–30.',
 'Коцюба И.Ю., Назаренко А.Е.: «Разработка рекомендательной системы для планирования туристических маршрутов в оптимизационной постановке». Моделирование, оптимизация и информационные технологии, 2020, 8 (2). DOI: 10.26102/2310-6018/2020.29.2.021.',
 'Цыплов А.М., Бухановский А.В.: «Комбинированная модель качества рекомендательных систем». Научно-технический вестник информационных технологий, механики и оптики, 2025, 25 (6), с. 1117–1124. DOI: 10.17586/2226-1494-2025-25-6-1117-1124.',
 'Ferrari Dacrema, M., Cremonesi, P., Jannach, D.: “Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches”. Proc. ACM RecSys, Copenhagen, Denmark, 2019, pp. 101–109. DOI: 10.1145/3298689.3347058.',
 'Lim, K.H., Chan, J., Leckie, C., et al.: “Personalized Tour Recommendation Based on User Interests and Points of Interest Visit Durations”. Proc. IJCAI, Buenos Aires, Argentina, 2015, pp. 1778–1783. https://ijcai.org/proceedings/2015/253.',
 'Vrandečić, D., Krötzsch, M.: “Wikidata: a free collaborative knowledgebase”. Communications of the ACM, 2014, 57 (10), pp. 78–85. DOI: 10.1145/2629489.',
 'Tong, H., Faloutsos, C., Pan, J.-Y.: “Fast Random Walk with Restart and Its Applications”. Proc. IEEE ICDM, Hong Kong, China, 2006, pp. 613–622. DOI: 10.1109/ICDM.2006.70.',
 'Steck, H.: “Embarrassingly Shallow Autoencoders for Sparse Data”. Proc. The Web Conference, San Francisco, USA, 2019, pp. 3251–3257. DOI: 10.1145/3308558.3313710.'
]


def fmt(v,d=4):
    return f'{float(v):.{d}f}'.replace('.',',').replace('-','−')


def style(doc,name,size,bold=False,italic=False):
    if name not in doc.styles:
        doc.styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH)
    st=doc.styles[name]
    st.font.name='Times New Roman';st.font.size=Pt(size)
    st.font.bold=bold;st.font.italic=italic;st.font.color.rgb=RGBColor(0,0,0)
    rf=st.element.get_or_add_rPr().rFonts
    for x in list(rf.attrib):rf.attrib.pop(x,None)
    for x in ['ascii','hAnsi','eastAsia','cs']:rf.set(qn('w:'+x),'Times New Roman')
    p=st.paragraph_format
    p.line_spacing=1.0;p.space_after=Pt(4);p.space_before=Pt(0)
    p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY;p.widow_control=True
    return st


def mr(text):
    r=OxmlElement('m:r');t=OxmlElement('m:t');t.text=text;r.append(t);return r


def script(text,subscript,sup=False):
    out=OxmlElement('m:sSup' if sup else 'm:sSub')
    e=OxmlElement('m:e');e.append(mr(text));out.append(e)
    s=OxmlElement('m:sup' if sup else 'm:sub');s.append(mr(subscript));out.append(s)
    return out


def configure(sec,cols):
    sec.page_width=Cm(21);sec.page_height=Cm(29.7)
    sec.left_margin=sec.right_margin=Cm(1.5)
    sec.top_margin=sec.bottom_margin=Cm(2.54)
    sec.header_distance=Cm(1.25);sec.footer_distance=Cm(1.25)
    old=sec._sectPr.find(qn('w:cols'))
    if old is not None:sec._sectPr.remove(old)
    c=OxmlElement('w:cols');c.set(qn('w:num'),str(cols));c.set(qn('w:space'),'227')
    c.set(qn('w:equalWidth'),'1');sec._sectPr.append(c)


def make_figure(C,H):
    e=C.loc['end_to_end'];g=C.loc['H1_cultural_links']
    rows=[(float(H['delta']),float(H['ci_low']),float(H['ci_high'])),
          (e.delta,e.ci_low,e.ci_high),(g.delta,g.ci_low,g.ci_high)]
    fig,ax=plt.subplots(figsize=(3.46,2.02))
    for j,(m,lo,hi) in enumerate(rows):
        ax.errorbar(m,j,xerr=[[m-lo],[hi-m]],fmt=['o','s','^'][j],color='#213a4a',
                    capsize=3,markersize=4.8,linewidth=1.15)
    ax.axvline(0,color='#606060',linestyle='--',linewidth=.7)
    ax.set_yticks([0,1,2],['H3','Гибрид','H1']);ax.invert_yaxis()
    ax.set_xlabel('Разность NDCG@5',fontsize=9)
    ax.set_xticks([-.05,0,.05,.10,.15,.20])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:.2f}'.replace('.',',').replace('-','−')))
    ax.tick_params(labelsize=8.4);ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x',alpha=.2);fig.tight_layout(pad=.45)
    f=OUT/'hypothesis_intervals_ru.png';fig.savefig(f,dpi=300);plt.close(fig);return f


def build(template,dest):
    doc=Document(template)
    body=doc._element.body
    for child in list(body):
        if child.tag!=qn('w:sectPr'):body.remove(child)
    for tag in ('w:headerReference','w:footerReference','w:titlePg','w:pgNumType'):
        for x in body.sectPr.findall(qn(tag)):body.sectPr.remove(x)
    base=doc.sections[0];base.start_type=WD_SECTION.NEW_PAGE;configure(base,1)
    style(doc,'Normal',11.5)
    tt=style(doc,'Title',18,True)
    tt.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER
    tt.paragraph_format.space_after=Pt(8)
    for name,size,b,i in [('Heading 1',12,True,False),('Heading 2',11,False,True),
                          ('IET Abstract RU',11,False,False),('IET Caption RU',10,False,False),
                          ('IET References RU',10,False,False),('IET Authors RU',10,False,False)]:
        st=style(doc,name,size,b,i)
        if name.startswith('Heading'):
            st.paragraph_format.keep_with_next=True
            st.paragraph_format.space_before=Pt(6)
        if name in ('IET References RU','IET Authors RU'):st.paragraph_format.space_after=Pt(3)
    for x in list(doc.styles.element.iter(qn('w:pBdr'))):x.getparent().remove(x)
    f=base.footer.paragraphs[0];f.clear();f.alignment=WD_ALIGN_PARAGRAPH.CENTER
    fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');f._p.append(fld)

    def p(s,sty=None):
        s=re.sub(r'(?<=\d) (?=\d{3}\b)','\u00a0',s)
        s=re.sub(r'(?<!\S)([ВвКкСсУуОоИиАа]) (?=\S)',lambda m:m[1]+'\u00a0',s)
        return doc.add_paragraph(s.replace(' — ',' – '),sty)
    def h(s,level=1):
        return doc.add_heading(s,level)
    def br(col=False):
        q=p('');q.paragraph_format.space_after=Pt(0);q.paragraph_format.line_spacing=Pt(1)
        q.add_run().add_break(WD_BREAK.COLUMN if col else WD_BREAK.PAGE)
    def eq(num,*parts):
        q=p('');q.alignment=WD_ALIGN_PARAGRAPH.LEFT
        q.paragraph_format.tab_stops.add_tab_stop(Cm(4.2),WD_TAB_ALIGNMENT.CENTER)
        q.paragraph_format.tab_stops.add_tab_stop(Cm(8.75),WD_TAB_ALIGNMENT.RIGHT)
        q.add_run('\t');math=OxmlElement('m:oMath')
        for x in parts:math.append(mr(x) if isinstance(x,str) else x)
        q._p.append(math);q.add_run('\t('+str(num)+')')
    def table(cap,headers,data,widths):
        p(cap,'IET Caption RU').paragraph_format.keep_with_next=True
        t=doc.add_table(rows=1,cols=len(headers));t.autofit=False;t.alignment=WD_TABLE_ALIGNMENT.CENTER
        for i,w in enumerate(widths):t.columns[i].width=Cm(w)
        for j,v in enumerate(headers):t.rows[0].cells[j].text=v
        for r in data:
            for c,v in zip(t.add_row().cells,r):c.text=str(v)
        for i,row in enumerate(t.rows):
            pr=row._tr.get_or_add_trPr();pr.append(OxmlElement('w:cantSplit'))
            if i==0:pr.append(OxmlElement('w:tblHeader'))
            for j,cell in enumerate(row.cells):
                cell.width=Cm(widths[j]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tc=cell._tc.get_or_add_tcPr();bb=OxmlElement('w:tcBorders')
                for edge in ['top','bottom','left','right']:
                    ed=OxmlElement('w:'+edge)
                    ed.set(qn('w:val'),'single' if edge=='bottom' and (i==0 or i==len(t.rows)-1) else 'nil')
                    ed.set(qn('w:sz'),'4');ed.set(qn('w:color'),'A0A0A0');bb.append(ed)
                tc.append(bb)
                margin=OxmlElement('w:tcMar')
                for edge in ['top','bottom','left','right']:
                    ed=OxmlElement('w:'+edge);ed.set(qn('w:w'),'45');ed.set(qn('w:type'),'dxa');margin.append(ed)
                tc.append(margin)
                for q in cell.paragraphs:
                    q.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                    q.paragraph_format.space_after=Pt(0);q.paragraph_format.line_spacing=1.0
                    for r in q.runs:r.font.size=Pt(10);r.bold=i==0
        q=p('');q.paragraph_format.line_spacing=Pt(3);q.paragraph_format.space_after=Pt(0)

    A=json.loads((OUT/'audit_summary.json').read_text())
    H=json.loads((OUT/'h3_summary.json').read_text())
    D=json.loads((OUT/'interpretation.json').read_text())
    C=pd.read_csv(FOLLOW/'contrasts.csv').query("evaluation=='transfer'").set_index('contrast')
    C1=pd.read_csv(ROOT/'results/metrics_all_variants.csv').query("city=='MAIN_MACRO' and split=='test' and subset=='warm'").set_index('model')
    C2=pd.read_csv(FOLLOW/'metrics.csv').query("evaluation=='transfer' and subset=='all' and city=='MACRO'").set_index('model')
    P=pd.read_csv(OUT/'coverage.csv').set_index('city')
    T=pd.read_csv(FOLLOW/'transfer_cohort.csv').set_index('city')
    S=pd.read_csv(FOLLOW/'nonzero_gate_contrasts.csv').query("evaluation=='transfer'").set_index('subset')
    timing=pd.read_csv(OUT/'common_date_sensitivity.csv').set_index('contrast').loc['H3']
    e=C.loc['end_to_end'];g=C.loc['H1_cultural_links'];flat=C.loc['matched_graph_vs_flat']
    figure=make_figure(C,H)

    # Page 1: full-width heading and abstract, then the two-column introduction.
    p(TITLE.upper(),'Title')
    p('[Имя и фамилия автора]¹','IET Authors RU').alignment=WD_ALIGN_PARAGRAPH.CENTER
    p('¹ [Подразделение, организация, город, страна]; [электронная почта]','IET Authors RU').alignment=WD_ALIGN_PARAGRAPH.CENTER
    p('Ключевые слова: культурное наследие, граф знаний, туристические рекомендации, история посещений, цифровая гуманитаристика.','IET Authors RU')
    h('Аннотация')
    abstract=(f'Что помогает предложить туристу следующее культурное место: история переходов, сведения об объектах или изменение веса интересов? '
              f'Мы проверили это на фотографиях Flickr и графах Wikidata для {A["objects"]} объектов восьми городов. '
              'Метод сочетает модель переходов Маркова и культурный граф, а профиль туриста разделяет на ранние посещения и текущий маршрут. '
              f'В тесте на пяти новых городах (81 прогноз, 44 пользователя) Марков получил NDCG@5 {fmt(H["Markov"])}, '
              f'гибрид с графом – {fmt(C2.loc["GraphPositive","NDCG5"])}. '
              'Польза культурных связей и изменения веса профилей не подтвердилась. '
              f'Дополнительная проверка показала преимущество учёта последнего места перед общей популярностью: '
              f'разность {fmt(H["delta"])}, 95%-й доверительный интервал [{fmt(H["ci_low"])}; {fmt(H["ci_high"])}]. '
              'Этот результат относится к изученному архиву. Мы отдельно проверили происхождение данных, разделение по времени и расчёт всех тестовых рангов второй серии.')
    assert len(abstract.split())<=200
    p(abstract,'IET Abstract RU')
    sec=doc.add_section(WD_SECTION.CONTINUOUS);configure(sec,2)
    h('1 Введение')
    p('Представим, что турист уже побывал в двух местах и выбирает третье. Сервис может предложить то, куда обычно идут после последней остановки. Есть и другой подход: учесть, что здания построены одним архитектором или относятся к одному стилю. Такие связи дают повод показать объект, который легко пропустить в обычном списке популярных мест.')
    p('Для цифровой гуманитаристики важен сам способ описания объектов. Хранилище может связывать памятники, людей и события, как в работе Савченко и Никифоровой [1]. Антопольский рассматривает роль русскоязычной онтологии [2], а Атаева, Серебряков и Тучкова показывают, как описывать связи в цифровой коллекции с помощью графа знаний [3]. В этой работе проверяется более узкий вопрос: улучшают ли такие связи порядок рекомендаций.')
    br(True)
    h('1.1 Вопросы исследования',2)
    p('Методы рекомендаций по посещениям и свойствам объектов описаны в обзоре [4]. Туристические маршруты можно также строить как задачу оптимизации с заданными ограничениями [5]. Здесь маршрут целиком не строится: мы предсказываем одно следующее место в известном городе.')
    p('H1: сведения об архитекторах, стилях и именовании помогают лучше, чем одни категории. H2: изменение веса ранних и текущих посещений помогает лучше постоянного веса. Качество измеряется по позиции реального следующего места; о мотивах туриста фотографии не сообщают.')
    p('Отдельно проверяем H3: учёт последнего места полезнее общей популярности. Это дополнительная гипотеза, сформулированная при аудите после первых сравнений. Она нужна, чтобы понять, есть ли в порядке посещений сигнал, с которым конкурирует культурный граф. Сопоставление моделей и учёт смещений важны при оценке рекомендательных систем [6, 7].')

    # Page 2: sources, lineage, chronology, scope.
    br();h('2 Данные')
    p('Последовательности получены из открытого исследовательского архива Flickr, описанного Лимом и соавторами [8]. Исходная строка – фотография с идентификатором пользователя, местом, временем съёмки и номером последовательности. Несколько снимков подряд в одном месте считаются одним посещением. Если человек ушёл и вернулся, возврат остаётся отдельным событием.')
    p('В архиве 153 208 строк. У 53 время выходит за допустимый диапазон, поэтому эти строки исключены. Остались 153 155 фотографий и 33 778 посещений после объединения одинаковых мест. Повторная обработка исходного архива дала те же результаты. Мы отдельно сверили каждую группу снимков с её местом, временем начала и конца.')
    p('Из общего списка оставили музеи, исторические здания, памятники, религиозные места и другие именованные культурные объекты. Большие районы и неоднозначно сопоставленные места исключили. Для Belvedere и Osaka Castle объединили дубли, а записи о National Museum of Scotland и Royal Museum отнесли к одному музейному комплексу. Решения по каждому объекту сохранены в данных.')
    p('Снимок Wikidata [9] получен 26 октября 2026 года. Для каждого места проверены идентификатор и координаты. Использованы три отношения: архитектор P84, архитектурный стиль P149 и именование P138. Категория служит отдельным типом связи. Все 164 культурные связи найдены в сохранённых ответах Wikidata; у 89 утверждений есть хотя бы одна ссылка.')
    p('Сверка подтверждает, что сведения перенесены из источника без подмены. Она не заменяет экспертной проверки исторических атрибуций. В частности, наличие ссылки в Wikidata само по себе не удостоверяет правильность утверждения.')
    br(True);h('2.1 Состав выборки и время',2)
    p('В таблице 1 показаны размеры каталогов и тесты второй серии. В исходных трёх городах приведены контексты с тремя ранними посещениями и двумя текущими. В пяти новых городах тест включает все контексты с наблюдаемым последним местом после исключения пользователей исходных городов.')
    table('Таблица 1 Каталоги и тесты второй серии',['Город','Места','Связи','Прогнозы'],
          [[CITY[c],int(P.loc[c,'places']),int(P.loc[c,'edges']),
            {'Buda':158,'Vien':114,'Edin':225}.get(c,int(T.loc[c,'test_contexts']) if c in T.index else 0)]
           for c in CITY],[3.55,1.55,1.6,2.1])
    p('Последовательности делятся по времени внутри каждого города примерно в пропорции 60/20/20. Отсекаются последовательности с неразрешимым порядком снимков, перекрывающиеся интервалы одного пользователя и пограничные последовательности в новых городах. В основном анализе рассматриваются последовательности, начавшиеся с 2000 года.')
    p('Профиль прогноза включает только уже наблюдавшиеся места и последовательности, завершённые до начала текущей. Частоты мест и переходов вычисляются только на обучении. Среди кандидатов весь культурный каталог города, кроме последнего места; раньше посещённые места остаются допустимыми. Независимая проверка восстановила все 8 721 контекст и подтвердила эти границы.')
    p(f'Городские отсечки приходятся на разные даты. В переносе {D["transfer_before_final_dev_cutoff"]} целевых событий произошли до последней даты валидации исходных городов. Поэтому оценка переноса характеризует перенос настроек между городами в историческом архиве, а не единый запуск сервиса в одну дату.')

    # Page 3: model and tuning.
    br();h('3 Как работает метод')
    p('У метода два источника оценок. Первый – порядок посещений. Марковская модель запоминает, сколько раз из каждого места переходили в каждое другое. Например, если после собора чаще шли в соседний музей, этот музей получает более высокий балл. Когда для последнего места нет обучающих переходов, модель использует популярность в городе.')
    p('Второй источник – граф культурных связей. Каждое место соединено со своей категорией и доступными культурными признаками. Если у объекта несколько архитекторов, общий вес отношения делится между ними поровну. Вариант с одними категориями даёт сравнение для H1. Ещё один вариант использует те же сведения как обычные признаки, без распространения оценок по графу.')
    p('Близость мест вычисляется случайным блужданием с возвратом [10]. Из текущей вершины можно пройти по связи либо с вероятностью λ вернуться к началу. Если P – нормированная матрица переходов по графу, то распределения достижимости для всех начальных вершин задаёт выражение')
    eq(1,'R = λ ',script('[I − (1 − λ)P]','−1',True))
    p('Из R берём только строки и столбцы мест, исключаем переход к самому себе и нормируем строки. Получаем K. Когда в строке нет доступных соседей, используем обучающую популярность. Для контроля без графа считаем косинусное сходство векторов из тех же признаков. Это позволяет отдельно проверить, даёт ли что-либо распространение по связям.')
    p('Ранний профиль L – доли посещений разных мест в завершённых последовательностях. Текущий профиль S – доли мест в уже известной части маршрута. С помощью K оба профиля получают баллы для ещё возможных следующих мест. Ранний профиль называется так по времени наблюдения: он не всегда описывает устойчивый интерес в течение месяцев.')
    br(True);h('3.1 Как объединяются баллы',2)
    p('В первой серии поведенческая часть – ItemKNN. Объекты сравниваются по общим посетителям обучающей выборки. Косинусное сходство умножается на m/(m + 5), где m – число общих посетителей. Поведенческое ядро C и графовое K смешиваются с весом η; к полученному ядру применяются профили L и S. Адаптивная доля S равна a + (b − a) × d × n/(n + c), где d – расхождение профилей, n – длина текущей истории, а a, b и c выбираются по валидации.')
    p('Во второй серии более сильная основа – Марков. Обозначим распределение Маркова через Mₜ, а графовую оценку через Qₜ. Она получается из смеси (1 − αₜ)L и αₜS после умножения на K. Оба распределения нормируются по одинаковым кандидатам. Итоговый балл:')
    eq(2,script('s','t'),' = (1 − η)',script('M','t'),' + η',script('Q','t'))
    p('При η = 0 остаётся Марков. При ненулевом η граф может изменить порядок. Для адаптации берём постоянную долю α₀ и меняем её на β × dₜ × nₜ/(nₜ + 3), ограничивая итог числом от 0 до 1. Расхождение dₜ равно половине суммы модулей разностей культурных долей двух профилей. Если данных о культуре одного профиля нет, поправка равна нулю; если старых посещений нет, используется только текущий профиль.')
    p('На валидации сравнивались веса графа от 0 до 0,50, пять долей текущего профиля от 0 до 1 и λ = 0,15, 0,30 или 0,60. Для β проверены 0, ±0,25 и ±0,50. Настройка идёт в два шага: сначала выбираются параметры постоянной смеси, затем при них подбирается β. Полные сетки и результаты подбора сохранены.')
    p('Модель загружается из сохранённых матриц. Вход – список текущих и ранних мест, выход – упорядоченные рекомендации. Во время выдачи обращение к Wikidata не требуется.')

    # Page 4: protocols and model comparisons.
    br();h('4 Сравнение моделей')
    p('Сначала модели настроены на Будапеште и Вене и проверены на их поздних посещениях. Основной тест первой серии содержит 272 предсказания для 47 пользователей. Эдинбург использован для переноса настроек с локальным обучением. Сравнивались популярность, Марков, ItemKNN, EASE [11], расстояние и графовые варианты.')
    p('Одна цель считается релевантной: место, куда человек действительно пришёл следующим. NDCG@5 даёт больше баллов за более высокую позицию в первой пятёрке. HR@5 считает попадания в пятёрку. MRR – среднее значение 1/ранг цели. Внутри города сначала считаем среднее для каждого человека, затем среднее по людям. Города получают одинаковый вес.')
    labels=[('Popularity','Популярность'),('Markov','Марков'),('ItemKNN','ItemKNN'),('EASE','EASE'),
            ('GeoFixed','Географическая модель'),('CulturalFixed','Гибрид с ItemKNN'),('CulturalAdaptive','Гибрид с адаптацией')]
    table('Таблица 2 Первая серия, Будапешт и Вена',['Метод','NDCG@5','HR@5','MRR'],
          [[name]+[fmt(C1.loc[k,m]) for m in ('NDCG5','HR5','MRR')] for k,name in labels],[4.45,1.75,1.35,1.25])
    p('В таблице 2 Марков опережает все варианты первой серии. Валидация гибрида с ItemKNN выбрала вес графа 0. Его оценка совпала с ItemKNN именно по этой причине. При принудительном ненулевом вкладе графа NDCG@5 составила 0,2572. Это дополнительное сравнение, которое проведено после выбора нулевого веса.')
    p('Чтобы проверить граф на более сильной основе, во второй серии его добавили непосредственно к Маркову. Сравнения на прежних трёх городах считаются повторными, потому что их результаты уже были известны при выборе этой архитектуры.')
    br(True);h('4.1 Перенос в другие города',2)
    p('Гиперпараметры второй серии подбирались на 263 валидационных контекстах Будапешта и Вены. Затем их зафиксировали. Для Дели, Глазго, Осаки, Перта и Торонто заново рассчитали только локальные обучающие частоты. Из теста исключили людей, встречавшихся в любом исходном городе. Осталось 81 предсказание для 44 пользователей.')
    p('Для полного графа выбраны λ = 0,15, η = 0,10 и α₀ = 0,25. Валидация предпочла β = 0, то есть постоянный вес. Категориальный и плоский варианты в таблице 3 используют те же веса, чтобы различие отражало описание мест, а не отдельный подбор смеси.')
    h3pred=pd.read_csv(OUT/'h3_predictions.csv').query("evaluation=='transfer' and model=='Popularity'")
    h3hr=h3pred.groupby(['city','user']).HR5.mean().groupby('city').mean().mean()
    rows=[['Популярность (H3)',fmt(H['Popularity']),fmt(h3hr)]]
    for k,name in [('Markov','Марков'),('CategoryMatched','Марков и категории'),
                   ('FlatMatched','Марков и плоские признаки'),('GraphPositive','Марков и культурный граф'),
                   ('AdaptiveFree','Граф и выбранная адаптация')]:
        rows.append([name,fmt(C2.loc[k,'NDCG5']),fmt(C2.loc[k,'HR5'])])
    table('Таблица 3 Пять городов переноса',['Метод','NDCG@5','HR@5'],rows,[5.4,1.82,1.58])
    p('Для Маркова и гибрида с графом разность NDCG@5 равна '+fmt(e.delta)+
      ', 95%-й интервал ['+fmt(e.ci_low)+'; '+fmt(e.ci_high)+']. Точечная оценка ниже у гибрида, но интервал включает ноль. '
      'В тесте с графом цель поднялась в списке в 5 случаях, опустилась в 15 и сохранила позицию в 61.')
    p('Парные интервалы получены повторными выборками пользователей, а не отдельных посещений. На один глобальный Flickr ID приходится один вес во всех городах. Проведено 10 000 попыток бутстрэпа и перестановок знаков. Для трёх запланированных сравнений переноса применена поправка Холма; H3 показана отдельно.')

    # Page 5: honest interpretation of the three hypotheses.
    br();h('5 Что показали гипотезы')
    h('5.1 Культурные связи',2)
    p(f'При равных весах полный граф и граф категорий дали практически одинаковое качество: разность {fmt(g.delta)}, '
      f'95%-й интервал [{fmt(g.ci_low)}; {fmt(g.ci_high)}], скорректированное p = {fmt(g.p_holm,3)}. '
      f'По сравнению с теми же сведениями без диффузии разность {fmt(flat.delta)}, интервал [{fmt(flat.ci_low)}; {fmt(flat.ci_high)}]. '
      'Данных для подтверждения H1 нет; возможный небольшой эффект остаётся неопределённым.')
    p(f'Одно из объяснений видно в самих данных. В новых городах только {D["transfer_linked_values"]} из {D["transfer_feature_values"]} '
      'значений культурных признаков повторяются у нескольких мест одного города. Если архитектор указан лишь у одного здания, '
      'по этой связи трудно подсказать другое здание. Во всех восьми городах культурные признаки есть у 83 из 135 мест.')
    p('Ещё одна возможная причина – выбор места часто связан с маршрутом. В тесте переноса 52 из 81 цели находятся в пределах километра от последнего места, а 40 уже встречались в истории. Марков видит, куда люди обычно идут дальше. Общий стиль или архитектор сам по себе ничего не говорит о расстоянии или времени работы музея. Эти числа помогают объяснить результат, но не устанавливают причину каждого выбора.')
    p('В отдельной проверке 20 раз переставляли культурные рёбра, сохраняя число связей каждого типа у места и у признака. При неизменных параметрах NDCG@5 менялась от 0,6551 до 0,6860 против 0,6837 у исходного графа. Такая проверка показывает чувствительность метода к структуре. Она не даёт основания считать исходный граф лучше всех перестановок.')
    br(True);h('5.2 Адаптация и порядок посещений',2)
    p('В первой серии при большом изменении категорий прибавка от адаптации составила 0,0018 на 59 прогнозах; после поправки p = 1. В переносе для H2 доступны 16 прогнозов восьми пользователей с двумя профилями. Только у 12 прогнозов культурные признаки есть в обоих. При β = 0 варианты совпали: валидация предпочла постоянный вес.')
    p(f'Чтобы проверить, что произойдёт при реальном изменении веса, отдельно взяли лучший ненулевой β = {fmt(S.loc["two_profiles","beta"],2)} из уже сохранённой валидационной сетки. '
      f'Вес изменился в {int(S.loc["two_profiles","contexts_with_changed_weight"])} из 16 случаев, но NDCG@5 подгруппы не изменилась. '
      'Это разведочная проверка. При восьми пользователях разумный вывод состоит в том, что прибавка не установлена.')
    p(f'H3 получила поддержку в дополнительном сравнении на том же тесте: Марков {fmt(H["Markov"])} против популярности '
      f'{fmt(H["Popularity"])}. Разность {fmt(H["delta"])}, интервал [{fmt(H["ci_low"])}; {fmt(H["ci_high"])}], p = {fmt(H["p_two_sided"],4)}. '
      'Разность положительна в каждом из пяти городов и сохраняется при исключении любого одного города. Этот результат показывает пользу учёта порядка на данном архиве.')
    figp=p('');figp.alignment=WD_ALIGN_PARAGRAPH.CENTER;figp.paragraph_format.keep_with_next=True
    figp.add_run().add_picture(str(figure),width=Cm(8.75))
    p('Рисунок 1 Парные разности с 95%-ми интервалами. H3: Марков минус популярность; гибрид: полный метод минус Марков; H1: полный граф минус категории.','IET Caption RU')
    p(f'При общей календарной границе осталось {int(timing.contexts)} прогнозов. Разность по H3 равна {fmt(timing.delta)}, интервал [{fmt(timing.ci_low)}; {fmt(timing.ci_high)}]. H3 следует перепроверить на другом корпусе.')

    # Page 6: scope of conclusions and numbered bibliography.
    br();h('6 Что ограничивает выводы')
    p('В тесте переноса размеры городов неравны: в Осаке всего два прогноза, а в Торонто 33. При пяти кандидатах HR@5 не различает модели в Осаке; NDCG@5 всё ещё учитывает позицию цели. Город с малым числом наблюдений получает такой же вес, как крупный, поэтому парные интервалы и проверка с исключением города особенно важны.')
    p('Flickr фиксирует посещения, во время которых люди фотографировали. Не все фактические остановки видны, и иногда более ранняя последовательность относится к той же поездке. Снимок Wikidata от 2026 года используется как описание мест задним числом. Граф и частоты построены для известного каталога; работа не проверяет добавление нового объекта, которого в этом каталоге не было.')
    p('Повторная обработка исходных фотографий и запуск обеих серий дали 16 побайтно совпавших файлов. Независимо пересчитаны 92 316 строк метрик и 18 238 рангов второй серии. При округлении почти равных баллов местами изменился порядок внутри отдельных списков, но NDCG@5 в сравнениях сохранилась.')
    p('К статье приложены исходные данные, каталог объектов и происхождение связей, временные разбиения, параметры, ранги и код. Проверки и таблица позволяют отследить результат до конкретного тестового события.')
    h('7 Заключение')
    p('Мы реализовали метод, который объединяет порядок посещений, культурный граф и два профиля. На изученных данных последовательность оказалась полезной: учёт последнего места улучшил ранжирование по сравнению с популярностью. При этом добавление графа к Маркову не показало устойчивой прибавки, а динамический вес профилей не помог в малой проверенной подгруппе.')
    p('Следующая проверка должна включать более полные культурные описания и пользователей с несколькими поездками, разделёнными во времени. Это позволит понять, помогают ли культурные связи и ранняя история там, где их действительно достаточно.')
    br(True);h('8 Литература')
    for i,s in enumerate(REFS,1):
        q=p(f'[{i}] {s}','IET References RU')
        q.paragraph_format.left_indent=Cm(.55);q.paragraph_format.first_line_indent=Cm(-.55)
        q.paragraph_format.keep_together=True
    doc.core_properties.title=TITLE
    doc.core_properties.author=''
    doc.core_properties.subject='Русская версия по шаблону IET'
    dest.parent.mkdir(parents=True,exist_ok=True);doc.save(dest)
    print(dest)


if __name__=='__main__':
    p0=argparse.ArgumentParser(description=__doc__)
    p0.add_argument('--template',type=Path,required=True)
    p0.add_argument('--out',type=Path,default=HERE/'Article_Cultural_Heritage_RU_Draft.docx')
    a=p0.parse_args();build(a.template,a.out)
