"""从课程电影数据集中构建并校验冻结的 40 题 benchmark v2。

题目规格只声明查询对象，答案、最小证据文档、证据字符区间和关系路径均由
films.jsonl 的结构化字段重新推导。这样可以人工阅读最终 YAML，同时用本脚本
检测语料更新造成的真值漂移。
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.import_wikipedia_films import build_person_document, fold_variants, source_to_text

SOURCE_DIR = PROJECT_ROOT / "data" / "source" / "wikipedia_300_films_final"
OUTPUT_FILE = PROJECT_ROOT / "eval" / "benchmark_v2" / "questions.yaml"
NOISE = re.compile(r"[{}\[\]|=<>]|^\s*$")


class NoAliasDumper(yaml.SafeDumper):
    """让冻结文件逐项展开，避免 YAML 锚点降低人工审阅可读性。"""

    def ignore_aliases(self, data: object) -> bool:
        return True


# 规格经人工挑选，兼顾题型、答案规模、跨文档跳数和主题多样性。不要根据一次
# 模型跑分自动替换这些对象，否则会把 test 集变成调参集。
SPECS: Dict[str, Sequence[Any]] = {
    "multi_director": (
        "无间道", "少林足球", "抓娃娃 (电影)", "爆款好人",
    ),
    "cofilm_set": (
        ("乔瑟夫·高登-李维", "米高·肯恩"),
        ("安妮·海瑟薇", "米高·肯恩"),
        ("汤姆·哈迪", "基利安·墨菲"),
        ("梁朝伟", "张曼玉"),
        ("刘德华", "梁朝伟"),
        ("威廉·达佛", "爱德华·诺顿"),
        ("乌玛·瑟曼", "刘玉玲"),
        ("森姆·积逊", "尼古拉斯·霍尔特"),
    ),
    "director_overlap": (
        ("克里斯托弗·诺兰", "昆汀·塔伦蒂诺"),
        ("克里斯托弗·诺兰", "大卫·芬奇"),
        ("宫崎骏", "王家卫"),
        ("昆汀·塔伦蒂诺", "大卫·芬奇"),
        ("斯蒂芬·斯皮尔伯格", "魏斯·安德森"),
        ("昆汀·塔伦蒂诺", "魏斯·安德森"),
        ("宁浩", "管虎"),
        ("张艺谋", "王家卫"),
    ),
    "repeated_cast": (
        "雷利·史考特", "昆汀·塔伦蒂诺", "宫崎骏", "王家卫",
        "宁浩", "大卫·芬奇", "魏斯·安德森", "陈咏燊",
    ),
    "director_actor_films": (
        ("克里斯托弗·诺兰", "汤姆·哈迪"),
        ("昆汀·塔伦蒂诺", "森姆·积逊"),
        ("王家卫", "梁朝伟"),
        ("大卫·芬奇", "毕·彼特"),
        ("宁浩", "黄渤"),
        ("斯蒂芬·斯皮尔伯格", "雷夫·范恩斯"),
    ),
    "hard_negative": (
        ("克里斯托弗·诺兰", "摩根·弗里曼", "莱昂纳多·迪卡普里奥"),
        ("昆汀·塔伦蒂诺", "森姆·积逊", "毕·彼特"),
        ("宫崎骏", "小林薰", "木村拓哉"),
        ("宁浩", "葛优", "黄渤"),
        ("大卫·芬奇", "摩根·弗里曼", "爱德华·诺顿"),
        ("张艺谋", "巩俐", "马丽"),
    ),
}

DEV_IDS = {
    "v2-multi-director-01",
    "v2-cofilm-01", "v2-cofilm-05",
    "v2-director-overlap-03", "v2-director-overlap-05",
    "v2-repeated-cast-03",
    "v2-director-actor-04",
    "v2-hard-negative-06",
}


def norm(text: object) -> str:
    value = fold_variants(str(text or ""))
    return re.sub(r"[\s\W_]+", "", unicodedata.normalize("NFKC", value).casefold())


def valid_name(name: object) -> bool:
    token = str(name or "").strip()
    return bool(token and len(token) <= 80 and not NOISE.search(token))


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        root = self.parent[item]
        if root != item:
            self.parent[item] = self.find(root)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def preferred_id(ids: Sequence[str]) -> str:
    rank = lambda item: (0 if item.startswith("Q") else 1 if item.startswith("WP:") else 2, item)
    return sorted(ids, key=rank)[0]


class Dataset:
    def __init__(self, source_dir: Path) -> None:
        self.source_dir = source_dir
        self.records = list(iter_jsonl(source_dir / "films.jsonl"))
        self.uf = UnionFind()
        self.raw_surfaces: Dict[str, set[str]] = defaultdict(set)
        self.raw_primary: Dict[str, Counter[str]] = defaultdict(Counter)
        self.person_rows: List[Tuple[dict, str]] = []
        self._collect_people()
        self.raw_to_canonical, self.entities = self._canonical_people()
        self.person_docs: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        self._build_person_docs()
        self.films: Dict[str, dict] = {}
        self.film_by_name: Dict[str, str] = {}
        self.film_directors: Dict[str, set[str]] = defaultdict(set)
        self.film_actors: Dict[str, set[str]] = defaultdict(set)
        self.director_films: Dict[str, set[str]] = defaultdict(set)
        self.actor_films: Dict[str, set[str]] = defaultdict(set)
        self._build_graph()

    def _add_person(self, person: dict) -> None:
        entity_id = str(person.get("id") or "").strip()
        name = str(person.get("name") or "").strip()
        if not entity_id or not valid_name(name):
            return
        self.uf.add(entity_id)
        self.raw_primary[entity_id][fold_variants(name)] += 1
        for surface in [name, *(person.get("aliases") or [])]:
            if valid_name(surface):
                self.raw_surfaces[entity_id].add(fold_variants(str(surface).strip()))

    def _collect_people(self) -> None:
        for file_name in ("actors.jsonl", "directors.jsonl"):
            role = "Actor" if file_name == "actors.jsonl" else "Director"
            for row in iter_jsonl(self.source_dir / file_name):
                self._add_person(row.get("person") or {})
                self.person_rows.append((row, role))
        for row in self.records:
            for key in ("actors", "directors"):
                for person in row.get(key) or []:
                    self._add_person(person)

        owner: Dict[str, str] = {}
        for entity_id, surfaces in self.raw_surfaces.items():
            for surface in surfaces:
                token = norm(surface)
                if not token:
                    continue
                if token in owner:
                    self.uf.union(entity_id, owner[token])
                else:
                    owner[token] = entity_id

    def _canonical_people(self) -> Tuple[Dict[str, str], Dict[str, dict]]:
        groups: Dict[str, List[str]] = defaultdict(list)
        for entity_id in self.uf.parent:
            groups[self.uf.find(entity_id)].append(entity_id)
        raw_to_canonical: Dict[str, str] = {}
        entities: Dict[str, dict] = {}
        for ids in groups.values():
            canonical = preferred_id(ids)
            surfaces = sorted(
                {surface for item in ids for surface in self.raw_surfaces[item]},
                key=lambda value: (-len(value), value),
            )
            # 优先采用 canonical ID 本身记录的名字，避免把别名当展示名。
            own = self.raw_primary.get(canonical, Counter())
            name = sorted(own, key=lambda value: (-own[value], value))[0] if own else surfaces[-1]
            aliases = [surface for surface in surfaces if surface != name]
            entities[canonical] = {"id": canonical, "name": name, "type": "Person", "aliases": aliases}
            for item in ids:
                raw_to_canonical[item] = canonical
        return raw_to_canonical, entities

    def _build_person_docs(self) -> None:
        for row, role in self.person_rows:
            raw_id = str((row.get("person") or {}).get("id") or "")
            canonical = self.raw_to_canonical.get(raw_id)
            document = build_person_document(row, role)
            if canonical and document:
                document = dict(document)
                document.pop("file_name", None)
                self.person_docs[(canonical, role)].append(document)

    def _build_graph(self) -> None:
        for row in self.records:
            film = row.get("film") or {}
            film_id = str(film.get("id") or "").strip()
            title = fold_variants(str(film.get("name") or "").strip())
            if not film_id or not valid_name(title):
                continue
            aliases = []
            short = re.sub(r"\s*[（(](?:\d{4}年)?电影[)）]\s*$", "", title).strip()
            if short and short != title:
                aliases.append(short)
            self.entities[film_id] = {"id": film_id, "name": title, "type": "Movie", "aliases": aliases}
            self.films[film_id] = {"record": row, "title": title, "text": source_to_text(row.get("source_document") or {})}
            self.film_by_name[norm(title)] = film_id
            for person in row.get("directors") or []:
                raw_id = str(person.get("id") or "")
                if raw_id not in self.raw_to_canonical:
                    continue
                director_id = self.raw_to_canonical[raw_id]
                self.film_directors[film_id].add(director_id)
                self.director_films[director_id].add(film_id)
            for person in row.get("actors") or []:
                raw_id = str(person.get("id") or "")
                if raw_id not in self.raw_to_canonical:
                    continue
                actor_id = self.raw_to_canonical[raw_id]
                self.film_actors[film_id].add(actor_id)
                self.actor_films[actor_id].add(film_id)

    def person(self, name: str) -> str:
        token = norm(name)
        matches = [entity_id for entity_id, item in self.entities.items()
                   if item["type"] == "Person" and token in {norm(item["name"]), *(norm(x) for x in item["aliases"])}]
        if len(matches) != 1:
            raise ValueError(f"人物 {name!r} 应唯一解析，实际为 {matches}")
        return matches[0]

    def film(self, title: str) -> str:
        film_id = self.film_by_name.get(norm(title))
        if not film_id:
            raise ValueError(f"找不到影片 {title!r}")
        return film_id

    def item(self, entity_id: str) -> dict:
        return self.entities[entity_id]

    def edge(self, person_id: str, relation: str, film_id: str) -> dict:
        return {"head_id": person_id, "relation": relation, "tail_id": film_id}

    def evidence_for_edges(self, edges: Sequence[dict]) -> List[dict]:
        result: List[dict] = []
        seen: set[Tuple[str, int, int]] = set()
        for edge in edges:
            film_id = edge["tail_id"]
            person = self.item(edge["head_id"])
            text = self.films[film_id]["text"]
            match = None
            for surface in [person["name"], *person["aliases"]]:
                start = text.find(surface)
                if start >= 0:
                    match = (start, start + len(surface), surface)
                    break
            if match is None:
                role = "Actor" if edge["relation"] == "acted_in" else "Director"
                title = self.films[film_id]["title"]
                for document in self.person_docs.get((person["id"], role), []):
                    person_text = str(document.get("text") or "")
                    start = person_text.find(title)
                    if start >= 0:
                        text = person_text
                        match = (start, start + len(title), title)
                        doc_id = str(document["doc_id"])
                        break
            else:
                doc_id = f"film_{film_id}"
            if match is None:
                raise ValueError(
                    f"影片与人物文档中都找不到关系证据：{self.films[film_id]['title']} / {person['name']}"
                )
            start, end, surface = match
            key = (doc_id, start, end)
            if key in seen:
                continue
            seen.add(key)
            result.append({
                "doc_id": doc_id, "char_start": start, "char_end": end,
                "text": surface, "supports": edge,
            })
        return result


def answer_items(dataset: Dataset, ids: Iterable[str]) -> List[dict]:
    return [dataset.item(entity_id) for entity_id in sorted(set(ids), key=lambda x: dataset.item(x)["name"])]


def make_question(
    dataset: Dataset,
    question_id: str,
    kind: str,
    question: str,
    subjects: Sequence[str],
    answers: Sequence[str],
    paths: Sequence[dict],
    *,
    answer_type: str = "entity_set",
    derivation: str,
) -> dict:
    edges = [edge for path in paths for edge in path.get("edges", [])]
    evidence = dataset.evidence_for_edges(edges)
    return {
        "id": question_id,
        "kind": kind,
        "difficulty": "calibration" if kind == "multi_director" else "hard",
        "question": question,
        "answer_type": answer_type,
        "target_entity_type": "none" if answer_type == "no_answer" else dataset.item(answers[0])["type"],
        "subjects": answer_items(dataset, subjects),
        "gold_answers": answer_items(dataset, answers),
        "gold_documents": sorted({item["doc_id"] for item in evidence}),
        "gold_evidence": evidence,
        "gold_paths": list(paths),
        "derivation": derivation,
    }


def build_questions(dataset: Dataset) -> List[dict]:
    questions: List[dict] = []

    for index, title in enumerate(SPECS["multi_director"], 1):
        film = dataset.film(title)
        directors = dataset.film_directors[film]
        paths = [{"answer_id": director, "edges": [dataset.edge(director, "directed", film)]}
                 for director in directors]
        questions.append(make_question(
            dataset, f"v2-multi-director-{index:02d}", "multi_director",
            f"严格依据当前语料，《{dataset.item(film)['name']}》的全部导演是谁？请给出完整名单，不要多答。",
            [film], list(directors), paths,
            derivation="影片记录中去重后的全部导演集合。",
        ))

    for index, pair in enumerate(SPECS["cofilm_set"], 1):
        left, right = (dataset.person(name) for name in pair)
        films = dataset.actor_films[left] & dataset.actor_films[right]
        paths = [{"answer_id": film, "edges": [
            dataset.edge(left, "acted_in", film), dataset.edge(right, "acted_in", film),
        ]} for film in films]
        questions.append(make_question(
            dataset, f"v2-cofilm-{index:02d}", "cofilm_set",
            f"在当前语料收录的影片中，{dataset.item(left)['name']}与{dataset.item(right)['name']}共同出演了哪些影片？请列出完整交集，不要多答。",
            [left, right], list(films), paths,
            derivation="两位演员去重片单的集合交集。",
        ))

    for index, pair in enumerate(SPECS["director_overlap"], 1):
        left, right = (dataset.person(name) for name in pair)
        left_actors = {actor for film in dataset.director_films[left] for actor in dataset.film_actors[film]}
        right_actors = {actor for film in dataset.director_films[right] for actor in dataset.film_actors[film]}
        actors = left_actors & right_actors
        paths = []
        for actor in actors:
            left_film = sorted(dataset.director_films[left] & dataset.actor_films[actor])[0]
            right_film = sorted(dataset.director_films[right] & dataset.actor_films[actor])[0]
            paths.append({"answer_id": actor, "edges": [
                dataset.edge(left, "directed", left_film), dataset.edge(actor, "acted_in", left_film),
                dataset.edge(right, "directed", right_film), dataset.edge(actor, "acted_in", right_film),
            ]})
        questions.append(make_question(
            dataset, f"v2-director-overlap-{index:02d}", "director_overlap",
            f"哪些演员既出演过{dataset.item(left)['name']}执导的影片，也出演过{dataset.item(right)['name']}执导的影片？请给出当前语料中的完整交集，不要多答。",
            [left, right], list(actors), paths,
            derivation="两位导演的去重演员集合交集；每个答案至少由两部导演作品支持。",
        ))

    for index, name in enumerate(SPECS["repeated_cast"], 1):
        director = dataset.person(name)
        counts = Counter(actor for film in dataset.director_films[director]
                         for actor in dataset.film_actors[film])
        actors = {actor for actor, count in counts.items() if count >= 2}
        paths = []
        for actor in actors:
            films = sorted(dataset.director_films[director] & dataset.actor_films[actor])[:2]
            edges = []
            for film in films:
                edges.extend([dataset.edge(director, "directed", film), dataset.edge(actor, "acted_in", film)])
            paths.append({"answer_id": actor, "edges": edges})
        questions.append(make_question(
            dataset, f"v2-repeated-cast-{index:02d}", "repeated_cast",
            f"按影片去重后，哪些演员在{dataset.item(director)['name']}执导的不同影片中至少出现过两次？请给出完整集合，不要多答。",
            [director], list(actors), paths,
            derivation="先对每部影片演员去重，再按导演汇总演员出现的不同影片数，阈值为 2。",
        ))

    for index, pair in enumerate(SPECS["director_actor_films"], 1):
        director, actor = (dataset.person(name) for name in pair)
        films = dataset.director_films[director] & dataset.actor_films[actor]
        paths = [{"answer_id": film, "edges": [
            dataset.edge(director, "directed", film), dataset.edge(actor, "acted_in", film),
        ]} for film in films]
        questions.append(make_question(
            dataset, f"v2-director-actor-{index:02d}", "director_actor_films",
            f"在当前语料中，{dataset.item(actor)['name']}出演过哪些由{dataset.item(director)['name']}执导的影片？请列出完整集合，不要多答。",
            [director, actor], list(films), paths,
            derivation="导演片单与演员片单的集合交集。",
        ))

    for index, triple in enumerate(SPECS["hard_negative"], 1):
        director, left, right = (dataset.person(name) for name in triple)
        common = dataset.actor_films[left] & dataset.actor_films[right]
        if common:
            raise ValueError(f"困难负例不成立：{triple} 共同出演 {common}")
        # 两位演员都与同一导演有关，构造强干扰证据；但没有共同影片。
        left_film = sorted(dataset.director_films[director] & dataset.actor_films[left])[0]
        right_film = sorted(dataset.director_films[director] & dataset.actor_films[right])[0]
        support = [{"answer_id": None, "edges": [
            dataset.edge(director, "directed", left_film), dataset.edge(left, "acted_in", left_film),
            dataset.edge(director, "directed", right_film), dataset.edge(right, "acted_in", right_film),
        ]}]
        negative = make_question(
            dataset, f"v2-hard-negative-{index:02d}", "hard_negative",
            f"在当前语料中，{dataset.item(left)['name']}与{dataset.item(right)['name']}是否共同出演过同一部影片？只回答“有”或“无”，不要列出各自出演的影片。",
            [left, right, director], [], support, answer_type="no_answer",
            derivation="两位演员分别出演过同一导演的不同影片，但去重片单交集为空。",
        )
        negative["gold_support_paths"] = negative["gold_paths"]
        negative["gold_paths"] = []
        questions.append(negative)

    add_graph_perturbations(dataset, questions)
    return questions


def add_graph_perturbations(dataset: Dataset, questions: Sequence[dict]) -> None:
    """给正例构造能切断所有 gold 解答路径、但正文证据仍保留的查询视图。"""
    for question in questions:
        kind = question["kind"]
        masked: List[dict] = []
        if kind in {"multi_director", "hard_negative"}:
            condition = "complete_control" if kind == "multi_director" else "negative_control"
        elif kind in {"cofilm_set", "director_overlap", "director_actor_films"}:
            # 每条答案路径隐藏最后一条 acted_in 边，确保该答案在可见图中失去证明。
            masked = [dict(path["edges"][-1]) for path in question["gold_paths"]]
            condition = "critical_edge_missing"
        elif kind == "repeated_cast":
            director = question["subjects"][0]["id"]
            for answer in question["gold_answers"]:
                actor = answer["id"]
                films = sorted(dataset.director_films[director] & dataset.actor_films[actor])
                # 只保留一次可见出演；其余出演边全部隐藏，使“至少两次”不再能由图推出。
                masked.extend(dataset.edge(actor, "acted_in", film) for film in films[1:])
            condition = "count_support_missing"
        else:
            raise ValueError(f"未定义缺边策略：{kind}")

        unique = {
            (edge["head_id"], edge["relation"], edge["tail_id"]): edge for edge in masked
        }
        masked = [unique[key] for key in sorted(unique)]
        evidence = dataset.evidence_for_edges(masked) if masked else []
        oracle_queries = [
            f"{dataset.item(edge['head_id'])['name']} {dataset.item(edge['tail_id'])['name']} "
            f"{'出演' if edge['relation'] == 'acted_in' else '执导'}"
            for edge in masked
        ]
        question["graph_perturbation"] = {
            "condition": condition,
            "expected_gap": bool(masked),
            "masked_edges": masked,
            "masked_edge_count": len(masked),
            "compensation_gold_documents": sorted({item["doc_id"] for item in evidence}),
            "compensation_gold_evidence": evidence,
            "oracle_queries": oracle_queries,
            "recoverable_from_text": bool(masked),
        }


def source_digest(source_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("films.jsonl", "actors.jsonl", "directors.jsonl", "relations.jsonl"):
        digest.update(name.encode())
        digest.update((source_dir / name).read_bytes())
    return digest.hexdigest()


def validate(questions: Sequence[dict]) -> None:
    expected = {"multi_director": 4, "cofilm_set": 8, "director_overlap": 8,
                "repeated_cast": 8, "director_actor_films": 6, "hard_negative": 6}
    counts = Counter(item["kind"] for item in questions)
    if len(questions) != 40 or counts != Counter(expected):
        raise ValueError(f"题量分布错误：{len(questions)} / {dict(counts)}")
    ids = [item["id"] for item in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("问题 ID 重复")
    splits = Counter(item["split"] for item in questions)
    if splits != Counter({"dev": 8, "test": 32}):
        raise ValueError(f"dev/test 分布错误：{dict(splits)}")
    conditions = Counter(
        item.get("graph_perturbation", {}).get("condition") for item in questions
    )
    expected_conditions = Counter({
        "critical_edge_missing": 22,
        "count_support_missing": 8,
        "complete_control": 4,
        "negative_control": 6,
    })
    if conditions != expected_conditions:
        raise ValueError(f"缺边场景分布错误：{dict(conditions)}")
    for item in questions:
        if item["answer_type"] != "no_answer" and not item["gold_answers"]:
            raise ValueError(f"{item['id']} 缺少答案")
        if not item["gold_documents"] or not item["gold_evidence"]:
            raise ValueError(f"{item['id']} 缺少可审计证据")
        perturbation = item.get("graph_perturbation") or {}
        if perturbation.get("expected_gap"):
            if not perturbation.get("masked_edges") or not perturbation.get("compensation_gold_documents"):
                raise ValueError(f"{item['id']} 的缺边场景不可从文本恢复")
            masked = {
                (edge["head_id"], edge["relation"], edge["tail_id"])
                for edge in perturbation["masked_edges"]
            }
            for path in item.get("gold_paths") or []:
                path_edges = {
                    (edge["head_id"], edge["relation"], edge["tail_id"])
                    for edge in path.get("edges") or []
                }
                if not (masked & path_edges):
                    raise ValueError(f"{item['id']} 存在未被切断的 gold 路径")


def main() -> int:
    dataset = Dataset(SOURCE_DIR)
    questions = build_questions(dataset)
    for question in questions:
        question["split"] = "dev" if question["id"] in DEV_IDS else "test"
    validate(questions)
    # 全量 Movie/Person 目录供严格 scorer 识别“多答”的数据集内实体，而不只识别 gold。
    catalog = {entity_id: dataset.item(entity_id) for entity_id in sorted(dataset.entities)}
    payload = {
        "benchmark": {
            "id": "film-graph-rag-gap-v2-40",
            "version": 2,
            "status": "frozen",
            "question_count": len(questions),
            "split_counts": {"dev": 8, "test": 32},
            "source_sha256": source_digest(SOURCE_DIR),
            "construction": "固定查询规格；完整真值图推导答案，并按题隐藏关键边，Chunk 证据保持不变",
        },
        "entity_catalog": catalog,
        "questions": questions,
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        yaml.dump(payload, Dumper=NoAliasDumper, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"已构建并冻结 {len(questions)} 题 -> {OUTPUT_FILE}")
    print("题型分布：", dict(Counter(item["kind"] for item in questions)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
