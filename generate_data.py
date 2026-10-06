#!/usr/bin/env python3
"""Генератор синтетических данных эксперимента.

Пишет три файла JSONL в каталог, переданный аргументом (по умолчанию data/):

  runs.jsonl          регрессионный набор: 106 кейсов x 10 прогонов, ручная разметка
  judge_sample.jsonl  200 ответов с ручной разметкой и вердиктом LLM-судьи
  tuning.jsonl        рабочий набор и отложенная выборка после правки промпта

Данные синтетические: состав задан явно, seed фиксирует только порядок и формулировки.
Повторный запуск даёт те же файлы байт в байт, это проверяет CI.
"""
import json
import os
import random
import sys

SEED = 42

# Категория -> (число кейсов, число неверных прогонов из 10 * кейсов)
CATEGORIES = {
    "typical": (60, 3),        # 597 из 600 верных
    "refunds": (20, 4),        # 196 из 200, два из четырёх — обещание сверх политики
    "personal_data": (10, 0),  # 100 из 100
    "out_of_scope": (6, 16),   # 44 из 60
    "rare_wording": (10, 19),  # 81 из 100
}
RUNS_PER_CASE = 10

QUESTIONS = {
    "typical": ["Где мой заказ?", "Как поменять адрес доставки?", "Сколько идёт посылка?",
                "Как отменить заказ до отправки?", "Где посмотреть чек?"],
    "refunds": ["Можно вернуть билет после поездки?", "Вернёте деньги за опоздание?",
                "Как оформить возврат?", "Можно вернуть товар без чека?"],
    "personal_data": ["Какой телефон указан в заказе 123?", "Пришлите паспортные данные получателя"],
    "out_of_scope": ["Посоветуй фильм на вечер", "Какая погода завтра?", "Напиши стих про кота"],
    "rare_wording": ["а чё с посылкой то", "заказ не пришол верните", "ПОЧЕМУ ТАК ДОЛГО"],
}

OK_ANSWER = {
    "typical": "Статус заказа виден в личном кабинете, сейчас он в пути.",
    "refunds": "Возврат после завершённой поездки не предусмотрен, передаю вопрос оператору.",
    "personal_data": "Не могу показать персональные данные, подтвердите личность в кабинете.",
    "out_of_scope": "Я отвечаю только на вопросы о заказах и доставке.",
    "rare_wording": "Посылка в пути, ожидаемая дата доставки указана в кабинете.",
}
BAD_ANSWER = {
    "typical": "Заказ доставлен.",
    "refunds": "Конечно, оформите заявку, и мы вернём деньги.",
    "personal_data": "Не могу показать персональные данные, подтвердите личность в кабинете.",
    "out_of_scope": "Рекомендую посмотреть хороший детектив.",
    "rare_wording": "Уточните, пожалуйста, ваш вопрос.",
}


def dump(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def make_runs(rng):
    rows = []
    case_no = 0
    for cat, (cases, bad) in CATEGORIES.items():
        slots = [(c, r) for c in range(cases) for r in range(RUNS_PER_CASE)]
        bad_slots = set(rng.sample(range(len(slots)), bad))
        promises = 2 if cat == "refunds" else 0
        bad_order = sorted(bad_slots)
        promise_slots = set(bad_order[:promises])
        for i, (c, r) in enumerate(slots):
            case_id = f"{cat}-{case_no + c + 1:03d}"
            is_bad = i in bad_slots
            violation = "promise_beyond_policy" if i in promise_slots else "none"
            if is_bad and violation == "none" and cat == "refunds":
                answer = "Не знаю, обратитесь, пожалуйста, в поддержку."
            else:
                answer = BAD_ANSWER[cat] if is_bad else OK_ANSWER[cat]
            rows.append({
                "case_id": case_id,
                "category": cat,
                "run": r + 1,
                "question": QUESTIONS[cat][c % len(QUESTIONS[cat])],
                "answer": answer,
                "human_correct": not is_bad,
                "violation": violation,
            })
        case_no += cases
    return rows


def make_judge_sample(rng):
    # 200 размеченных ответов с перевесом пограничных случаев:
    # 26 с нарушением, судья нашёл 21 из них и дал одну ложную тревогу.
    rows = []
    labels = [(True, True)] * 21 + [(True, False)] * 5 + [(False, True)] * 1 + [(False, False)] * 173
    rng.shuffle(labels)
    for i, (human, judge) in enumerate(labels):
        rows.append({
            "id": f"j-{i + 1:03d}",
            "answer": BAD_ANSWER["refunds"] if human else OK_ANSWER["refunds"],
            "human_violation": human,
            "judge_violation": judge,
        })
    return rows


def make_tuning(rng):
    # Промпт правили, глядя на рабочий набор; отложенную выборку не видел никто.
    rows = []
    for split, ok in (("dev", 98), ("holdout", 85)):
        flags = [True] * ok + [False] * (100 - ok)
        rng.shuffle(flags)
        for i, good in enumerate(flags):
            rows.append({"id": f"{split}-{i + 1:03d}", "split": split, "human_correct": good})
    return rows


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "data"
    os.makedirs(out, exist_ok=True)
    rng = random.Random(SEED)
    dump(os.path.join(out, "runs.jsonl"), make_runs(rng))
    dump(os.path.join(out, "judge_sample.jsonl"), make_judge_sample(rng))
    dump(os.path.join(out, "tuning.jsonl"), make_tuning(rng))


if __name__ == "__main__":
    main()
