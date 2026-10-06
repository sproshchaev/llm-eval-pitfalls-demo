#!/usr/bin/env bash
# Пересобирает данные из генератора, печатает четыре отчёта и сверяет вывод с expected.txt.
# Нужен только python3, сторонние пакеты не используются.
set -euo pipefail
cd "$(dirname "$0")"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

python3 generate_data.py "$tmp"
if ! diff -r "$tmp" data > /dev/null; then
  echo "РАСХОЖДЕНИЕ: data/ не совпадает с выводом generate_data.py"
  exit 1
fi

out=$(python3 eval.py)
echo "$out"

if [ "$out" = "$(cat expected.txt)" ]; then
  echo "OK: результаты совпадают с expected.txt"
else
  echo "РАСХОЖДЕНИЕ с expected.txt:"
  diff <(echo "$out") expected.txt || true
  exit 1
fi
