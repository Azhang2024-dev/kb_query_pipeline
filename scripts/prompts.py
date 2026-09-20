# -*- coding: utf-8 -*-
"""语言无关 prompt 模板：认列 + 指定/质量判定 + 语义去重。

判定口径（严格版）：
- 指定 query = 指向单一特定对象且不可泛化，含指示词指向与具名实体指向两类；
- 具名实体（人/机构/案件/项目/系统/考试/日期/表格/代码/金额/书）只要问的是"那个具体对象"
  即判指定，即使句式是通用疑问；例外是"问通用规则/知识"，即使提到具名机构。
"""

SCHEMA_PROMPT = """You are a data schema mapper. Given a table file, figure out which column holds which semantic role.

The table columns (with dtype, non-null ratio, and sample values) are:

{columns}

Map the actual column names to these canonical roles:
- query: the user's question text (REQUIRED, one column)
- answer: the LLM's generated answer text (REQUIRED, one column)
- search_result: the retrieved chunk/snippet text (REQUIRED, one column)
- score: the retrieval similarity score (REQUIRED, one column)
- filename: source document filename (optional)
- request_id: unique id that groups rows belonging to the same query (optional but strongly recommended if present)
- reference: the 标注/gold/citation annotation column (optional)
- is_cited: whether the answer cites the retrieved content (optional)
- citation_no: citation sequence number (optional)

Rules:
- A column must be mapped to at most one role; one role to exactly one column.
- Column names may be in ANY language (Chinese, English, French, Arabic, Russian, Indonesian...) and may be abbreviated, misspelled, or have no meaningful name. Judge by dtype, sample values, and content pattern, not just the header text.
- score columns contain numeric values like 0.5-1.0 or percentage-like numbers.
- search_result columns contain long text snippets of retrieved documents.
- query columns contain question-like text (may end with ? ? ؟ ? or be phrased as a question).
- answer columns contain complete, fluent answers (usually longer, citation marks like <citation:1> or [citation] may appear).
- If a column clearly matches a role, map it. Do not invent roles that do not exist in the file.
- If you are unsure, set "confidence" below 0.6 and explain in "reason".

Respond with ONLY a JSON object like:
{{
  "query": "<actual column name>",
  "answer": "<actual column name>",
  "search_result": "<actual column name>",
  "score": "<actual column name>",
  "filename": "<actual column name or null>",
  "request_id": "<actual column name or null>",
  "reference": "<actual column name or null>",
  "is_cited": "<actual column name or null>",
  "citation_no": "<actual column name or null>",
  "confidence": {{"<role>": 0.0-1.0, ...}},
  "reason": "<short explanation of mapping decisions>"
}}
"""

# ---------------------------------------------------------------------------
# 判定标准（严格口径）——单条与批量 prompt 共用
# ---------------------------------------------------------------------------

CLASSIFY_RULES = """DEFINITION of a specified query (指定 query):

A question is "specified" when it POINTS AT ONE SPECIFIC OBJECT that cannot be generalized away — it can only be answered by looking at THAT particular item. There are TWO families:

【Family A】 Demonstrative/determiner pointing (指示词指向):
- this/that, this contract, that document, this TV, 这个/那个/此合同/该文件/本协议/这款
- French: ce/cet/cette/ces (ce projet, cette convention, cet accord)
- Russian: этот/эта/это/эти/тот/та/то/те + данный/настоящий/указанный, incl. all case forms (этой теме, этому взысканию, этим жилым помещением, этого сборника, в этом наборе, у того)
- Arabic: هذا/هذه/ذلك/تلك + المذكور/المذكورة (the aforementioned/the said)
- Indonesian: ini/itu (kontrak ini, dokumen itu)
The demonstrative ALONE makes it point at a specific thing — specified even if the rest sounds general.

【Family B】 Named-entity pointing (具名实体指向) — NEW STRICT RULE:
Even WITHOUT any demonstrative word, a question is specified if it asks about ONE specific identifiable object that only makes sense within a particular document/context. Examples that MUST be judged "specified" (指定):
- A specific person / specific email: "Comment contacter Sarah Estelle Touvoli par e-mail ?", "Comment contacter Haytham Ouertani par courriel ?" (contacting THAT person)
- A specific case / specific company: "Pourquoi le recours en annulation a-t-il été déclaré irrecevable pour forclusion ?", "Quelles mesures bancaires sont prévues pour corriger le déséquilibre financier de l'entreprise ?" (Cas DEMBA SARL)
- A specific project / specific site: "Quel est le pourcentage global de fermeture des actions HAZOP sur les sites de Gamba-Ivinga et Atora ?"
- A specific system / platform / server: "Quel est le format du SMS de configuration pour connecter une montre en protocole watch à un serveur Traccar ?"
- A specific exam / specific year: "Combien de sujets le candidat peut-il choisir pour l'épreuve de philosophie du baccalauréat général 2024 ?", "متى يتم فتح باب التظلمات على نتيجة الدور الثاني للثانوية العامة؟", "كيف أستعلم عن نتيجة الثانوية الأزهرية 2021 باستخدام رقم الجلوس؟"
- A specific date / date range: "Quel est le thème du stage de recherche effectué du 02 décembre au 18 janvier 2024 ?"
- A specific form / specific document / specific code: "ما الخطوات اللازمة لاستكمال نموذج توصية طالب الماستر وتوقيعه وختمه؟", "Как расшифровывается иное код 31 в больничном листе?", "Quelle est la signification du code de sécurité H335 ?"
- A specific amount / specific priced item: "Quel est le poids de la portion de frite vendue à 1000 ?"
- A specific book / specific textbook: "كيف ينقسم محتوى كتاب كفاية الأصول إلى مباحث الألفاظ والأصول العملية والعقلية؟", "ما موضوع تركيب الذرة في كتاب العلوم للصف الثالث المتوسط؟"
- A specific specific regulated/legal instrument by number/title: "Федеральный закон № 178-ФЗ", "закон № 135-ФЗ", "в постановлении" (ONLY if multi-file retrieval; a single-file retrieval is answerable and NOT specified)
- A specific CONTRACT/AGREEMENT CLAUSE (合同条款) — even if phrased like a general question, asking what THE contract/agreement stipulates (amount, ratio, deadline, condition, liability, penalty, warranty, service term) is specified; it can only be answered by reading that particular agreement. Indonesian examples that MUST be "指定": "Kapan pembayaran 40% DP, 40% berikutnya, dan 20% pelunasan harus dilakukan?", "Apakah penggantian suku cadang di atas Rp 1.900.000 per unit harus mendapat persetujuan Pihak Pertama sebelum pengerjaan?", "Berapa lama kewajiban menjaga kerahasiaan tetap berlaku setelah masa penjajakan berakhir?", "Berapa lama waktu respons untuk kecatatan kategori Kritis, Major, dan Minor pada masa pemeliharaan?", "Bagaimana ketentuan tanda jadi jika pembeli terlambat atau tidak melunasi pembayaran tepat waktu?", "Kapan perjanjian yang mengikat firma kepada pihak ketiga harus disetujui oleh kedua sekutu?", "Apakah imbalan jasa tetap dibayar jika pengurusan tidak berhasil karena alasan di luar kendali Pihak Kedua?", "Apa rekomendasi batas maksimal pencatatan saldo akhir pada prosedur penutupan?" Trigger words (Indonesian): Pihak Pertama/Kedua/Ketiga, Terjamin, cidera janji, perjanjian, DP + number, uang muka, tanda jadi, pelunasan, cicilan, garansi, retur, SLA, Masa Pemeliharaan, penjajakan, kerahasiaan, imbalan jasa, suku cadang, saldo akhir, "Rp <amount>". EXCEPTION: a general definition/rule question about the CONCEPT is NOT specified (e.g. "apa itu garansi?", "bagaimana prinsip umum perjanjian?").
- A specific EXAM-QUESTION REFERENCE (试卷题号/指标类) — asking WHICH question number / which indicator / which question format covers a topic IN A SPECIFIC test paper is specified; only that test's blueprint can answer it. Indonesian examples: "Nomor soal mana yang mencakup indikator merumuskan langkah takhrij?", "Apa bentuk soal dan nomor soal untuk materi perjalanan dakwah Nabi Muhammad SAW?", "Apa indikator soal untuk materi struktur teks laporan hasil observasi?", "Soal bentuk partisipasi politik warga negara muncul pada nomor berapa?" Trigger words: nomor soal, indikator soal, bentuk soal, muncul pada nomor, diuji pada nomor, diujikan dalam bentuk.

EXCEPTIONS — NOT specified even with a named entity (these ask about GENERAL RULES / general knowledge, not about "that one object"):
- General-rule questions mentioning a NAMED INSTITUTION: "À quels étudiants la bourse d'encouragement de l'Ambassade de Chine au Togo est-elle destinée ?" (asks the general rule of a scholarship — NOT specified), "Можно ли подать документы в Академию через Госуслуги или только лично ?" (general rule)
- General knowledge questions: "Quel type de régime politique a la Côte d'Ivoire ?", "Comment expliquer le concept de l'économie circulaire ?", "Как расшифровывается иное код 31" is specified (a specific form field) BUT "Какое сечение нейтрального провода принимают в трехфазных четырехпроводных сетях?" is NOT (general engineering rule)
- Generic products/software/parts as general subjects: "Microsoft Word", "Mi Router", "Калькулятор 3д печати", "деталь «Стойка»" — general subject, NOT specified
- Indonesian general-knowledge/after-sales/licensing questions (validated by human review): "Di mana saya bisa memperbaiki bor listrik tangan Surya Pratama yang rusak?" (generic repair service for a product — NOT specified), "Kapan perawat harus mengajukan perpanjangan SIP sebelum masa berlakunya habis?" (general license-renewal rule — NOT specified), "Untuk apa ASEAN membentuk forum ADMM pada 9 Mei 2006 di Kuala Lumpur?" (general institutional knowledge — NOT specified), "Berapa jarak kosong minimal di sekitar produk saat pemasangan agar sirkulasi udara cukup?" (generic product-installation rule — NOT specified)
- Time expressions that only mean "this year/today": 今年/今天/tahun ini/hari ini — NOT specified
- Definition questions: "X itu apa", "Qu'est-ce que X", "что такое X", "ما المقصود بـ X" — NOT specified
- Classification questions: "это банковский счет или нет?" — NOT specified
- Legal terms without a specific instrument: "персональных данных", "ТО" abbreviation — NOT specified
- Historical/cultural facts asked generally: "причины пакта Молотова-Риббентропа" — NOT specified

TIE-BREAKER RULE: ask yourself — could this question be answered without reading ANY specific document, from general knowledge alone? If YES → not specified. If it REQUIRES looking at ONE specific document/context to know WHICH object is meant → specified. When in doubt between 是 and 否, prefer 存疑 (uncertain) — do NOT force 是. Uncertain rows are fail-closed into the gray list for human verdict; forcing 是 silently enriches the removed pool and hides difficulty shift from eval readers.

QUALITY ISSUES to check (may be multiple, or none):
- "empty_answer": the answer text is empty/blank/missing
- "mixed_language": the question text is written in a language DIFFERENT from the evaluation language ({language_hint}). E.g. a German question in a French evaluation set, or a question containing Chinese characters in an Indonesian evaluation set. Only flag it if the question is clearly not in the evaluation language; ignore occasional loanwords.
- "incomplete_question": the question is truncated, cut off, or unparseable
- "retrieval_failed": the search results / chunks are all empty or missing
"""


def _single_prompt_body():
    return f"""You are classifying questions for a knowledge-base QA evaluation pipeline. Decide two things about the question:
1. Is it a "specified/specific question" (指定 query)?
2. Does it have quality issues?

{CLASSIFY_RULES}

The retrieval info for this question: {{nfile_text}}

QUESTION:
{{query}}

ANSWER:
{{answer}}

Respond with ONLY a JSON object:
{{
  "is_specified": "是" | "否" | "存疑",
  "quality_issues": ["empty_answer" | "mixed_language" | "incomplete_question" | "retrieval_failed"],
  "reason": "<one or two sentences in the same language as the question>",
  "confidence": 0.0-1.0
}}

When unsure between 是 and 否, you MUST use "存疑" (uncertain). Do not force 是. Prefer a clear 是/否 only when confident."""


CLASSIFY_PROMPT = _single_prompt_body()


CLASSIFY_BATCH_PROMPT = (
    "You are classifying questions for a knowledge-base QA evaluation pipeline. Decide two things about EACH question in the list:\n"
    "1. Is it a \"specified/specific question\" (指定 query)?\n"
    "2. Does it have quality issues?\n\n"
    + CLASSIFY_RULES
    + """
Each question is given as a JSON object: {{"index": i, "query": ..., "answer": ..., "nfile": ...}} where nfile = number of retrieved files.

Input questions:
{questions}

Respond with ONLY a JSON object:
{{
  "results": [
    {{
      "index": <same index as input>,
      "is_specified": "是" | "否" | "存疑",
      "quality_issues": ["empty_answer" | "mixed_language" | "incomplete_question" | "retrieval_failed"],
      "reason": "<one or two sentences in the same language as the question>",
      "confidence": 0.0-1.0
    }},
    ...
  ]
}}

Rules:
- Return EXACTLY one result per input question, with matching "index" values.
- When unsure between 是 and 否, you MUST use "存疑" (uncertain). Do not force 是. Prefer a clear 是/否 only when confident.
"""
)


# ---------------------------------------------------------------------------
# 语义去重 prompt（差额补抽/复核时对边界对使用）
# ---------------------------------------------------------------------------

DEDUP_PROMPT = """You are checking whether two questions for a knowledge-base QA evaluation set are semantically duplicates (意思相近/重复).

Question A: {q1}
Question B: {q2}

Decide whether B asks essentially the SAME thing as A — same meaning expressed in different words (同义反复), or asks for the same fact/rule in a different phrasing. 

NOT duplicates if they differ in what they actually ask (different aspect, different scope, one asks start and the other end, different subjects), even if the topic domain overlaps.

Respond with ONLY a JSON object:
{{
  "is_duplicate": true | false,
  "reason": "<one sentence, in the same language as the questions>"
}}
"""
