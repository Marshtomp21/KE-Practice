# L2 Film 120 v2 标注复核记录

复核日期：2026-09-03  
状态：冻结（修改题目、答案、证据或实体映射时必须升级版本）

## 复核规则

- 逐题核对问题语义、规范实体 ID、别名、gold documents、证据原文及字符范围。
- 路径题必须存在“两名演员 → 同一影片”的两条 `acted_in` 边。
- 聚合题先对单部影片演员去重；每个答案必须拥有至少两条指向不同影片的路径。
- 清除模板残片，且同一题候选实体的归一化别名不能属于不同 canonical ID。
- 无答案题中的虚构片名或人名已在 120 篇冻结文档中核对为不存在。

## Dev（允许调参，不可并入最终分数）

| ID | 类型 | Gold answer | Gold documents |
|---|---|---|---|
| dev-fact-01 | 事实 | 西恩·贝克 | film_Q123185887 |
| dev-fact-02 | 事实 | 宫崎骏 | film_Q155653 |
| dev-path-01 | 路径 | 饭戏攻心 | film_Q108003049 |
| dev-aggregate-01 | 集合 | 比尔·莫瑞；爱德华·诺顿；蒂达·史云顿；詹森·舒瓦兹曼 | film_Q217112, film_Q3521099 |
| dev-no-answer-01 | 无答案 | 无法根据冻结语料确定 | — |

## Test（最终冻结评测）

| ID | 类型 | Gold answer | Gold documents |
|---|---|---|---|
| test-fact-01 | 事实 | 哈维尔·根斯 | film_Q124291916 |
| test-fact-02 | 事实 | 陶德·菲利普斯 | film_Q108628759 |
| test-fact-03 | 事实 | 张栾 | film_Q131690947 |
| test-fact-04 | 事实 | 昆汀·杜皮尔 | film_Q124472773 |
| test-fact-05 | 事实 | 李耀燮 | film_Q126209959 |
| test-fact-06 | 事实 | 陈咏燊 | film_Q108003049 |
| test-fact-07 | 事实 | 陆川（导演） | film_Q109345157 |
| test-fact-08 | 事实 | 大卫·雷奇 | film_Q113671585 |
| test-path-01 | 路径 | 749局 | film_Q109345157 |
| test-path-02 | 路径 | 璀璨女人梦 | film_Q123928072 |
| test-path-03 | 路径 | 我谈的那场恋爱 | film_Q130288104 |
| test-path-04 | 路径 | 看我今天怎么说 | film_Q131311712 |
| test-path-05 | 路径 | 公开试当真 | film_Q129333360 |
| test-aggregate-01 | 集合 | 张继聪；林明祯；王菀之；邓丽欣；陈湛文 | film_Q108003049, film_Q123330658 |
| test-aggregate-02 | 集合 | 神木隆之介；美轮明宏 | film_Q155653, film_Q186572, film_Q29011 |
| test-aggregate-03 | 集合 | 徐峥；黄渤 | film_Q11085495, film_Q855284 |
| test-aggregate-04 | 集合 | 毕·彼特 | film_Q190050, film_Q190908 |
| test-aggregate-05 | 集合 | 张曼玉；张震；梁朝伟；章子怡 | film_Q1056853, film_Q1155695, film_Q164702 |
| test-no-answer-01 | 无答案 | 无法根据冻结语料确定 | — |
| test-no-answer-02 | 无答案 | 无法根据冻结语料确定 | — |

## 人工实体修正

- `张震`、`张震 (演员)` 及源数据冲突 ID 统一为 `Q717432`。
- `毕·彼特`、`布拉德·皮特`、`布莱德·彼特` 统一为 `Q35332`。
- `詹森·舒瓦兹曼`、`杰森·薛兹曼`、`积逊·舒华沙曼` 统一为 `Q313705`。
- `陆川 (导演)` 保留规范名与 `陆川` 别名，canonical ID 为 `Q1856626`。

机器完整性检查见 `shared/tests/test_benchmark.py`；逐题完整字段以 `dev.jsonl` 和
`test.jsonl` 为准，文件散列见 `freeze_manifest.json`。
