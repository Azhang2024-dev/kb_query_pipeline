# -*- coding: utf-8 -*-
"""合成测试数据生成：长格式检索结果（证据文件结构）。"""
import pandas as pd

COLUMNS = ["query", "query_id", "query_type", "标注", "request_id", "doc_id",
           "filename", "搜索结果", "score", "label", "问答结果", "是否引用", "引用序号"]


def make_df(records):
    """records: list of (request_id, query, score, answer, search_result)。"""
    rows = []
    for i, (rid, q, score, ans, src) in enumerate(records, 1):
        rows.append({
            "query": q,
            "query_id": i,
            "query_type": "question",
            "标注": "",
            "request_id": rid,
            "doc_id": f"doc_{i}",
            "filename": f"file_{i}.pdf",
            "搜索结果": src,
            "score": score,
            "label": 1,
            "问答结果": ans,
            "是否引用": "Y" if i % 2 else "N",
            "引用序号": i,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def golden_specified():
    """俄语变格指示词漏筛黄金用例（必须判为指定）。"""
    return [
        "Сколько длится лекционное занятие по этой теме?",
        "Какой УИН указан для перечисления задолженности по этому взысканию?",
        "Может ли собственник квартиры выселить проживающего в ней гражданина, если у того нет права пользования жилым помещением?",
        "Какова сумма комиссии по этой безналичной оплате услуг?",
        "Какова главная особенность этого сборника зелий?",
        "От чего зависят свойства каждого зелья в этом наборе?",
    ]


def golden_not_specified():
    """非指定黄金用例（必须判为否）。"""
    return [
        "Как подать документы в Академию через Госуслуги?",
        "Quels sont les trois critères d'une bonne formulation d'un sujet de recherche ?",
        "Какой объем персональных данных можно передавать третьим лицам для цели обработки?",
    ]


def make_golden_df():
    """俄语变格黄金用例 → 长格式 df（每条 1-2 行召回）。"""
    records = []
    for i, q in enumerate(golden_specified() + golden_not_specified(), 1):
        records.append((f"rid_golden_{i:02d}", q, 0.7, "Ответ: да.", "检索片段内容"))
        records.append((f"rid_golden_{i:02d}", q, 0.5, "Ответ: да.", "第二个召回片段"))
    return make_df(records)


if __name__ == "__main__":
    df = make_golden_df()
    print(df.to_string())
