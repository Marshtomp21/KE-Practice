"""生成一份规模达标、连通性良好的影视语料，写入 data/raw。

真实百科正文受版权与抓取限制，无法随仓库分发；本脚本按固定随机种子合成
一份**虚构但结构自洽**的影视语料：若干位高产导演各自带有稳定班底，作品之间
存在改编、续作关系，人物之间通过影片形成多跳路径。因为生成时的结构是已知的，
它同时充当评测的标准答案（写入 eval/ground_truth.json）。

正文一律写成自然语言叙事段落，不使用表格式罗列，抽取任务才有意义。
替换成真实语料时，只需保证 data/raw 下 JSON 字段一致，下游无需改动。

用法：python scripts/make_sample_corpus.py [--movies 140] [--out data/raw]
"""
from __future__ import annotations

import argparse
import json
import hashlib
import random
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SURNAMES = list("李王张刘陈杨黄赵吴周徐孙马朱胡林郭何高罗郑梁谢宋唐许韩冯邓曹彭曾萧田董袁潘蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方邹熊白孟秦邱侯江尹薛闫段雷黎史陶贺毛郝顾龚邵万钱严覃武戴莫孔向汤")
GIVEN_CHARS = list("嘉明志清远航书文思宁婉云舒烨澜然瀚辰逸澄泽衡砚池茗琮予洛珩兮亦知临见白青苔霁初宜和安顺祥恒方直朴真淳")

TITLE_HEAD = ["南方", "北岸", "长夜", "暮色", "浮城", "旧海", "青玉", "山雨", "灯塔", "候鸟",
              "雾港", "麦浪", "边镇", "银鱼", "潮汐", "孤岛", "长街", "白塔", "寒枝", "夏至",
              "野火", "河灯", "石桥", "落雪", "远山", "晚风", "尘光", "空巷", "月台", "槐树"]
TITLE_TAIL = ["来信", "列车", "旅人", "手记", "回声", "长夏", "余晖", "证词", "背面", "尽头",
              "之外", "以南", "纪事", "遗事", "秘密", "候场", "归途", "断章", "沉默", "呼吸"]

GENRES = ["剧情", "喜剧", "犯罪", "悬疑", "爱情", "武侠", "科幻", "历史", "动作", "家庭"]
AWARDS = ["金鹿奖", "白鹭奖", "东亚电影节", "南岭国际电影节", "青云电影奖", "海鸥奖"]
AWARD_CATEGORY = {
    "金鹿奖": ["最佳影片", "最佳导演", "最佳男主角", "最佳女主角", "最佳编剧"],
    "白鹭奖": ["最佳影片", "最佳导演", "最佳新人"],
    "东亚电影节": ["评审团大奖", "最佳导演"],
    "南岭国际电影节": ["最佳影片", "最佳男主角"],
    "青云电影奖": ["最佳编剧", "最佳女主角"],
    "海鸥奖": ["最佳影片", "最佳摄影"],
}
COMPANY_HEAD = ["长风", "海石", "云岭", "北渡", "青禾", "白露", "远方", "沧浪", "明夷", "钧天"]
REGIONS = ["中国内地", "中国香港", "中国台湾"]
NOVELISTS_NOTE = ["同名小说", "短篇小说集中的一篇", "长篇纪实文学"]


def stable_id(prefix: str, token: str) -> str:
    """文档 ID 必须跨进程稳定，不能用内置 hash（受 PYTHONHASHSEED 影响）。"""
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}"


def build_name_pool(rng: random.Random, size: int) -> List[str]:
    pool: List[str] = []
    seen = set()
    while len(pool) < size:
        name = rng.choice(SURNAMES) + "".join(rng.choice(GIVEN_CHARS) for _ in range(rng.choice([1, 2])))
        if name in seen:
            continue
        seen.add(name)
        pool.append(name)
    return pool


def build_title_pool(rng: random.Random, size: int) -> List[str]:
    pool: List[str] = []
    seen = set()
    while len(pool) < size:
        title = rng.choice(TITLE_HEAD) + rng.choice(TITLE_TAIL)
        if title in seen:
            continue
        seen.add(title)
        pool.append(title)
    return pool


def synthesize(movie_count: int, seed: int) -> Dict[str, object]:
    """按导演圈层生成世界设定，返回结构化事实。"""
    rng = random.Random(seed)

    director_count = 12
    people = build_name_pool(rng, 460)
    directors = people[:director_count]
    writers = people[director_count:director_count + 60]
    actors = people[director_count + 60:]
    titles = build_title_pool(rng, movie_count + 20)
    companies = [head + "影业" for head in COMPANY_HEAD]

    # 每位导演有稳定班底：4 位常用演员 + 2 位常用编剧 + 1 家常合作公司
    troupe = {
        d: {
            "actors": rng.sample(actors, 4),
            "writers": rng.sample(writers, 2),
            "company": rng.choice(companies),
        }
        for d in directors
    }

    movies: List[Dict[str, object]] = []
    for index in range(movie_count):
        director = directors[index % director_count]
        crew = troupe[director]
        year = 1992 + (index // director_count) * 2 + rng.randint(0, 1)
        cast = rng.sample(crew["actors"], rng.randint(2, 3)) + rng.sample(actors, rng.randint(2, 4))
        cast = list(dict.fromkeys(cast))
        movie = {
            "title": titles[index],
            "year": year,
            "region": rng.choice(REGIONS),
            "director": director,
            "writers": rng.sample(crew["writers"], 1) + ([director] if rng.random() < 0.3 else []),
            "company": crew["company"] if rng.random() < 0.75 else rng.choice(companies),
            "genres": rng.sample(GENRES, rng.randint(1, 2)),
            "cast": cast,
            "characters": {},
            "awards": [],
            "adapted_from": None,
            "sequel_of": None,
        }
        for actor in cast[:2]:
            movie["characters"][actor] = rng.choice(SURNAMES) + rng.choice(GIVEN_CHARS)
        movies.append(movie)

    # 改编与续作：让影片之间也有边，多跳路径不止经由人物
    for index, movie in enumerate(movies):
        if index >= director_count and rng.random() < 0.12:
            earlier = movies[rng.randrange(0, index)]
            if earlier["director"] == movie["director"] and earlier["year"] < movie["year"]:
                movie["sequel_of"] = earlier["title"]
        if rng.random() < 0.18:
            movie["adapted_from_note"] = rng.choice(NOVELISTS_NOTE)

    # 奖项：约三成影片有获奖或提名记录
    for movie in movies:
        if rng.random() < 0.34:
            award = rng.choice(AWARDS)
            category = rng.choice(AWARD_CATEGORY[award])
            winner = movie["director"] if "导演" in category else (
                movie["cast"][0] if "男主角" in category or "女主角" in category or "新人" in category
                else movie["title"]
            )
            movie["awards"].append(
                {
                    "award": award,
                    "category": category,
                    "year": movie["year"] + 1,
                    "subject": winner,
                    "won": rng.random() < 0.55,
                }
            )

    return {"directors": directors, "troupe": troupe, "movies": movies, "companies": companies}


def movie_document(movie: Dict[str, object], world: Dict[str, object]) -> Dict[str, object]:
    title = movie["title"]
    year = movie["year"]
    lines: List[str] = []
    lines.append(
        f"《{title}》是 {year} 年上映的{movie['region']}{'、'.join(movie['genres'])}电影，"
        f"由{movie['director']}执导，{movie['company']}出品[1]。"
    )
    writers = "、".join(movie["writers"])
    lines.append(f"影片剧本由{writers}撰写。")
    if movie.get("adapted_from_note"):
        lines.append(f"故事改编自{movie['adapted_from_note']}。")
    if movie["sequel_of"]:
        lines.append(f"本片是{movie['director']}此前作品《{movie['sequel_of']}》的续作。")

    cast_bits = []
    for actor in movie["cast"]:
        role = movie["characters"].get(actor)
        cast_bits.append(f"{actor}在片中饰演{role}" if role else f"{actor}参与了本片的演出")
    lines.append("，".join(cast_bits) + "。")

    for record in movie["awards"]:
        verb = "获得" if record["won"] else "获得了提名"
        lines.append(
            f"{record['year']} 年，{record['subject']}凭借本片在{record['award']}上"
            f"{verb}{record['category']}。"
        )

    body = "\n\n".join(lines)
    return {
        "doc_id": stable_id("movie", title),
        "title": title,
        "url": f"https://example.org/entry/movie/{title}",
        "entity_type": "Movie",
        "text": body,
        "infobox": {
            "上映年份": str(year),
            "导演": movie["director"],
            "出品公司": movie["company"],
        },
    }


def person_document(name: str, world: Dict[str, object]) -> Dict[str, object]:
    movies = world["movies"]
    directed = [m for m in movies if m["director"] == name]
    acted = [m for m in movies if name in m["cast"]]
    written = [m for m in movies if name in m["writers"]]
    if not (directed or acted or written):
        return {}

    first_year = min(m["year"] for m in directed + acted + written)
    roles = []
    if directed:
        roles.append("导演")
    if acted:
        roles.append("演员")
    if written:
        roles.append("编剧")

    lines: List[str] = [
        f"{name}，{first_year - 30} 年出生，{'、'.join(roles)}，"
        f"{first_year} 年起活跃于影坛[1]。"
    ]

    if directed:
        head = directed[0]
        lines.append(
            f"{head['year']} 年，{name}执导了个人作品《{head['title']}》，影片由"
            f"{head['company']}出品。"
        )
        if len(directed) > 1:
            rest = "、".join(f"《{m['title']}》（{m['year']}）" for m in directed[1:6])
            lines.append(f"此后他先后执导了{rest}等影片。")
        regulars = world["troupe"][name]["actors"]
        lines.append(
            f"{name}的作品长期启用固定班底，{'、'.join(regulars[:3])}等演员多次出现在他的影片中。"
        )

    if acted:
        sample = acted[:5]
        for movie in sample:
            role = movie["characters"].get(name)
            if role:
                lines.append(
                    f"{movie['year']} 年，{name}出演{movie['director']}执导的《{movie['title']}》，"
                    f"在片中饰演{role}。"
                )
            else:
                lines.append(
                    f"{movie['year']} 年，{name}参演了{movie['director']}执导的《{movie['title']}》。"
                )
        if len(acted) > 5:
            lines.append(f"除此之外，{name}还参与过另外 {len(acted) - 5} 部影片的演出。")

    if written:
        titles = "、".join(f"《{m['title']}》" for m in written[:4])
        lines.append(f"作为编剧，{name}撰写了{titles}等影片的剧本。")

    for movie in directed + acted:
        for record in movie["awards"]:
            if record["subject"] == name:
                verb = "获得" if record["won"] else "获得提名"
                lines.append(
                    f"{record['year']} 年，{name}凭借《{movie['title']}》在"
                    f"{record['award']}{verb}{record['category']}。"
                )

    return {
        "doc_id": stable_id("person", name),
        "title": name,
        "url": f"https://example.org/entry/person/{name}",
        "entity_type": "Person",
        "text": "\n\n".join(lines),
        "infobox": {"职业": "、".join(roles)},
    }


def company_document(company: str, world: Dict[str, object]) -> Dict[str, object]:
    produced = [m for m in world["movies"] if m["company"] == company]
    if not produced:
        return {}
    first = min(m["year"] for m in produced)
    titles = "、".join(f"《{m['title']}》（{m['year']}）" for m in produced[:8])
    directors = sorted({m["director"] for m in produced})
    text = (
        f"{company}是一家成立于 {first - 3} 年的影视制作机构[1]。\n\n"
        f"公司先后出品了{titles}等影片。\n\n"
        f"{'、'.join(directors[:4])}等导演的作品由该公司出品。"
    )
    return {
        "doc_id": stable_id("company", company),
        "title": company,
        "url": f"https://example.org/entry/company/{company}",
        "entity_type": "Company",
        "text": text,
        "infobox": {"成立年份": str(first - 3)},
    }


def ground_truth(world: Dict[str, object]) -> Dict[str, object]:
    """把生成时已知的结构导出为标准答案，评测脚本据此判分。"""
    movies = world["movies"]
    return {
        "movies": [
            {
                "title": m["title"],
                "year": m["year"],
                "director": m["director"],
                "cast": m["cast"],
                "writers": m["writers"],
                "company": m["company"],
                "genres": m["genres"],
                "awards": m["awards"],
                "sequel_of": m["sequel_of"],
            }
            for m in movies
        ],
        "directors": world["directors"],
        "troupe": {d: v["actors"] for d, v in world["troupe"].items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成影视领域样例语料")
    parser.add_argument("--movies", type=int, default=160, help="影片条目数量")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--clean", action="store_true", help="生成前清空输出目录")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for stale in out_dir.glob("*.json"):
            stale.unlink()

    world = synthesize(args.movies, args.seed)

    written = 0
    for movie in world["movies"]:
        doc = movie_document(movie, world)
        (out_dir / f"{doc['doc_id']}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written += 1

    people = set(world["directors"])
    for movie in world["movies"]:
        people.update(movie["cast"])
        people.update(movie["writers"])
    person_docs = 0
    for name in sorted(people):
        doc = person_document(name, world)
        if not doc:
            continue
        (out_dir / f"{doc['doc_id']}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        person_docs += 1
        written += 1

    for company in world["companies"]:
        doc = company_document(company, world)
        if not doc:
            continue
        (out_dir / f"{doc['doc_id']}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written += 1

    truth_file = PROJECT_ROOT / "eval" / "ground_truth.json"
    truth_file.parent.mkdir(parents=True, exist_ok=True)
    truth_file.write_text(
        json.dumps(ground_truth(world), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"影片条目 {len(world['movies'])}，影人条目 {person_docs}，共写出 {written} 个文件 -> {out_dir}")
    print(f"标准答案 -> {truth_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
