from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from llm_client import LLMClient
from prompts import (
    COT_PROMPT,
    FEW_SHOT_PROMPT,
    GRID_PROMPT_VARIANTS,
    JSON_PROMPT,
    JUDGE_PROMPT,
    SIMPLE_PROMPT,
    SYSTEM_ROLE,
)


@dataclass
class MovieComment:
    title: str
    comment: str


MOVIE_COMMENTS_2026 = [
    MovieComment("沙丘3", "镜头语言依然顶级，但中段节奏太慢，结尾救回来了。"),
    MovieComment("疯狂动物城2", "梗密集又温暖，小孩大人都能看得很开心。"),
    MovieComment("阿凡达3", "特效无可挑剔，可人物关系写得有点套路。"),
    MovieComment("蜘蛛侠：新篇", "动作戏爆炸，情绪线也完整，是今年最惊喜的商业片。"),
    MovieComment("挽救计划", "设定很大，剧情却东一榔头西一棒槌，看完只剩疲惫。"),
]

PLOT_FOR_COT = """在未来都市中，主角林岸是一名记忆修复师，专门帮人找回被删除的记忆。
他接到一位失踪少女母亲的委托，发现少女最后一段记忆指向一家慈善基金会。
调查深入后，林岸发现基金会其实在用“创伤治疗”名义筛选可控人格。
第一重反转：少女并未失踪，而是主动潜入基金会内部试图曝光真相。
第二重反转：林岸发现自己的童年记忆也被篡改，他可能曾是基金会实验体。
结局中，林岸公开证据，但城市主系统宣布证据为伪造，民众转而质疑林岸。
最终镜头显示：林岸看到自己正在播放一段“早已录好的忏悔视频”。"""

PLOT_FOR_GRID = """一个落魄编剧为了还债，接下神秘投资人的任务：
在30天内写出一部能让观众“自愿遗忘痛苦”的电影。
他采访不同人群，发现每个人都想忘记不同的东西：失败、背叛、死亡。
剧本不断修改，编剧逐渐把自己的真实经历写进去。
首映当天观众泪流满面，但散场后多数人说不清电影讲了什么。
编剧拿到尾款后发现合同附加条款：项目成功即默认出售个人记忆样本。
最后他翻开旧日记，发现自己已经第三次参与同一项目。"""


def extract_json(text: str) -> Dict[str, Any]:
    """Parse pure JSON text or JSON enclosed in markdown fences."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\\s*", "", raw)
        raw = re.sub(r"\\s*```$", "", raw)
    return json.loads(raw)


def task1_zero_vs_few_shot(client: LLMClient) -> None:
    print("=" * 20, "任务1：Zero-shot vs Few-shot", "=" * 20)
    for item in MOVIE_COMMENTS_2026:
        zero = client.chat([
            {"role": "user", "content": SIMPLE_PROMPT.format(comment=item.comment)}
        ])
        few = client.chat([
            {"role": "user", "content": FEW_SHOT_PROMPT.format(comment=item.comment)}
        ])
        print(f"电影: {item.title}")
        print(f"短评: {item.comment}")
        print(f"Zero-shot => {zero}")
        print(f"Few-shot  => {few}")
        print("-" * 70)


def task2_force_json(client: LLMClient) -> None:
    print("=" * 20, "任务2：强制 JSON 输出", "=" * 20)
    for item in MOVIE_COMMENTS_2026:
        prompt = JSON_PROMPT.format(comment=item.comment, title=item.title)
        text = client.chat([{"role": "user", "content": prompt}], temperature=0)
        try:
            parsed = extract_json(text)
            print(f"[{item.title}] JSON解析成功: {parsed}")
        except json.JSONDecodeError as err:
            print(f"[{item.title}] JSON解析失败: {err}")
            print("原始输出:")
            print(text)
        print("-" * 70)


def task3_cot(client: LLMClient) -> None:
    print("=" * 20, "任务3：思维链（CoT）", "=" * 20)

    no_cot = client.chat([
        {
            "role": "user",
            "content": "请分析下面剧情反转是否合理，并给出1-10评分：\n" + PLOT_FOR_COT,
        }
    ])
    with_cot = client.chat([
        {"role": "user", "content": COT_PROMPT.format(plot=PLOT_FOR_COT)}
    ])

    print("不加 CoT 指令:")
    print(no_cot)
    print("-" * 70)
    print("加 CoT 指令:")
    print(with_cot)


def task4_system_role_chat(client: LLMClient) -> None:
    print("=" * 20, "任务4：角色扮演与 System Prompt", "=" * 20)
    print("输入 exit 退出。")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_ROLE}]
    while True:
        user_input = input("你: ").strip()
        if user_input.lower() == "exit":
            print("聊天结束。")
            break

        messages.append({"role": "user", "content": user_input})
        reply = client.chat(messages)
        print(f"影评家: {reply}")
        messages.append({"role": "assistant", "content": reply})


def evaluate_prompt(client: LLMClient, target_prompt: str) -> Dict[str, Any]:
    """Call an LLM judge to score prompt quality."""
    text = client.chat([
        {"role": "user", "content": JUDGE_PROMPT.format(target_prompt=target_prompt)}
    ])
    return extract_json(text)


def info_density(text: str) -> float:
    tokens = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]+", text)
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    length_penalty = 1.0 / (1.0 + abs(len(tokens) - 45) / 45)
    return unique_ratio * length_penalty


def task6_grid_search(client: LLMClient) -> None:
    print("=" * 20, "任务6：提示词迭代优化（Grid Search）", "=" * 20)

    candidates = []
    for idx, tmpl in enumerate(GRID_PROMPT_VARIANTS, start=1):
        prompt = tmpl.format(plot=PLOT_FOR_GRID)
        output = client.chat([{"role": "user", "content": prompt}], temperature=0.3)
        score = info_density(output)
        candidates.append({
            "id": idx,
            "prompt": tmpl,
            "output": output,
            "length": len(output),
            "density": round(score, 4),
        })

    for item in candidates:
        print(f"方案{item['id']} | length={item['length']} | density={item['density']}")
        print(item["output"])
        print("-" * 70)

    best = max(candidates, key=lambda x: x["density"])
    print("最佳方案:")
    print(
        json.dumps(
            {
                "best_id": best["id"],
                "best_density": best["density"],
                "reason": "信息密度得分最高",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    client = LLMClient()

    task1_zero_vs_few_shot(client)
    task2_force_json(client)
    task3_cot(client)

    print("=" * 20, "任务5：提示词打分器", "=" * 20)
    report = evaluate_prompt(client, SIMPLE_PROMPT)
    print("SIMPLE_PROMPT 评分:")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    task6_grid_search(client)

    # 任务4是交互式聊天，默认不自动进入，避免批处理运行被阻塞
    print("\n如需体验任务4（角色扮演聊天），请手动调用: task4_system_role_chat(client)")


if __name__ == "__main__":
    main()
