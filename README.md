<p align="center"><img src="./assets/logo-wordmark.svg" alt="MindStream AI" width="460"/></p>

<h1 align="center">Soup Kernel Challenge · свои Triton-кернелы для Soup</h1>

<p align="center">
  <em>Быстрее и легче то же самое обучение Qwen2.5-1.5B на бесплатной Kaggle T4 — без изменения результата</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Decentrathon-6.0-7C3AED?style=for-the-badge" alt="Decentrathon 6.0" />
  <img src="https://img.shields.io/badge/Case-Soup_Kernel_Challenge-0EA5E9?style=for-the-badge" alt="Soup Kernel Challenge" />
  <img src="https://img.shields.io/badge/Triton-3.6-4B32C3?style=for-the-badge" alt="Triton 3.6" />
  <img src="https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.11" />
  <img src="https://img.shields.io/badge/GPU-Kaggle_T4_(fp16)-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle T4" />
  <img src="https://img.shields.io/badge/soup--cli-0.77.0-F59E0B?style=for-the-badge" alt="soup-cli 0.77.0" />
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.13" />
  <a href="./solution/test_head_loss.py"><img src="https://img.shields.io/badge/Kernel_tests-pytest-brightgreen?style=for-the-badge&logo=pytest&logoColor=white" alt="Kernel tests" /></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge" alt="Apache-2.0" /></a>
</p>

<p align="center">
  <a href="#-kaggle">📓 Kaggle-ноутбук</a> |
  <a href="#-результаты">📊 Результаты</a> |
  <a href="#kernels">⚙️ Кернелы</a> |
  <a href="#-как-запустить">🚀 Запуск</a>
</p>

<p align="center">
  <b>🇷🇺 Русский</b> · <a href="#kk">🇰🇿 Қазақша</a> · <a href="#en">🇬🇧 English</a>
</p>

---

## 💡 Коротко

`soup train` дообучает **Qwen2.5-1.5B-Instruct** (веса NF4, LoRA r=16) на 256 казахских инструкциях на одной Kaggle T4. В базе это около 80 с и около 14,35 ГБ видеопамяти. Наши Triton-кернелы подключаются к Soup одной функцией `solution.apply()`: та же задача, тот же адаптер того же качества — но меньше памяти и времени.

> **Статус:** идёт разработка (30–31 октября 2026). Цифры ниже обновляются после каждого прогона на Kaggle T4.

---

## 📊 Результаты

Замер — официальные скрипты `harness/run_task.py` и `harness/check_rules.py` на Kaggle T4.

| | База Soup 0.77.0 | Наше решение | Отношение |
|---|---|---|---|
| Время `soup train` | ⏳ | ⏳ | s = ⏳ |
| Пик памяти GPU (nvidia-smi) | ⏳ | ⏳ | m = ⏳ |
| Eval loss адаптера | ⏳ | ⏳ | Δ = ⏳ |
| Доля наших кернелов во времени обучения | — | ⏳ | нужно ≥ 5% |

Проверки правил (`check_rules.py`): ⏳ после прогона.

---

<a id="kernels"></a>

## ⚙️ Что сделано

### 1. Слитая cross-entropy по словарю из 151 936 токенов — [`solution/head_loss.py`](./solution/head_loss.py)

**Проблема.** База держит в памяти полные логиты `(batch, seq, 151 936)`: fp16 после LM head, fp32-копию для loss, их градиент, `.contiguous()`-копию для точности в TRL и ещё раз считает cross-entropy по строкам для журнала `rewind.jsonl` Soup. Это несколько гигабайт на шаг.

**Решение.** Токены идут кусками по 512:

```text
 hidden (fp32) ──► кусок 512 токенов ──► logits = h16 @ W16ᵀ   (тот же fp16-matmul, что в базе)
                                            │
                     ┌──────────────────────┴───────────────────────┐
                     │ Triton: head_ce_stats_grad_kernel (1 строка = 1 программа)
                     │  проход 1: online log-sum-exp, логит цели, Σp·z (энтропия), argmax
                     │  проход 2: на месте пишет softmax − one_hot (градиент)
                     └──────────────────────┬───────────────────────┘
                                            ▼
                     dH = grad @ W16  (только строки с метками) → хранится до backward
```

- Полные логиты **никогда не существуют**: живёт только кусок (~155 МБ) и градиент скрытых состояний `(токены × 1536)`.
- Один проход даёт всё, что нужно потребителям: loss для обучения, энтропию и точность для лога TRL, loss каждой строки для `rewind.jsonl` Soup. Формулы взяты из исходников TRL 0.29.1 и Soup 0.77.0 без изменений.
- Градиент считается ещё в прямом проходе, поэтому backward не пересчитывает логиты.
- Подключение: [`solution/patches.py`](./solution/patches.py) — патч `Qwen2ForCausalLM.forward` (только обучающий шаг на CUDA) и `trl.SFTTrainer.compute_loss` (логирование метрик). Во всех остальных случаях работает оригинальный код.

### 2. Пример организаторов SwiGLU — [`solution/swiglu.py`](./solution/swiglu.py)

Оставлен включённым (минус ~2 ГБ). **В наши кернелы не засчитывается** — так требуют правила.

### ⏳ Следующие кернелы

Выбираются по профилю `--profile` с T4: распаковка NF4, слитый RMSNorm, поэлементные операции (44% времени шага по профилю организаторов).

---

## 🚀 Как запустить

<a id="-kaggle"></a>

**Kaggle (так проверяет жюри):**

1. Kaggle → **Create → New Notebook → File → Import Notebook** → [`templates/kaggle_notebook.ipynb`](./templates/kaggle_notebook.ipynb).
2. Справа: **Accelerator: GPU T4 x2**, **Internet: On** (аккаунт с подтверждённым телефоном).
3. В первой ячейке: `REPO_URL = "https://github.com/BAITC-Hacks/hack-65172d00-mindstream-ai.git"`, `COMMIT = "⏳ финальный коммит"`.
4. **Run All** (~10 минут): база и решение, `runs/result.json`, таблица проверок.

📓 Публичный ноутбук с результатами: ⏳ ссылка появится перед сдачей.

**Локально (Linux + CUDA GPU):**

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py                        # данные
python -m pytest solution -q                       # наши кернелы против PyTorch
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q    # 79 тестов Soup с нашими патчами
python harness/run_task.py                         # база и решение → runs/result.json
python harness/check_rules.py                      # проверки → runs/checks.json
```

---

## 🧪 Тесты

| Набор | Что проверяет |
|---|---|
| [`solution/test_head_loss.py`](./solution/test_head_loss.py) | loss, градиент, энтропия, loss по токенам, точность — против PyTorch; разные формы, паддинг, первый максимум в argmax |
| [`solution/test_kernels.py`](./solution/test_kernels.py) | пример SwiGLU от организаторов |
| [`tests/`](./tests) | 79 регрессионных тестов Soup с нашими патчами |

---

## ⚠️ Ограничения (честно)

- ⏳ Заполняется по итогам прогонов: где решение слабее, что не получилось.
- Метрики TRL и `rewind.jsonl` считаются из fp32-статистик кернела, а база — из fp16-логитов: возможны расхождения в десятитысячных (допуск правил — 3%).

---

## 📚 Источники идей

- E. Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models* (2024) — не материализовать логиты большого словаря.
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel): fused linear cross-entropy по кускам — идея, код не использовался.
- M. Milakov, N. Gimelshein, *Online normalizer calculation for softmax* (2018) — online log-sum-exp.

Код кернелов написан с нуля. Готовые кернел-библиотеки не используются (проверка 2).

---

## 🗂 Структура

| Путь | Что это |
|---|---|
| `solution/` | наши кернелы, `apply()` и тесты |
| `harness/`, `tests/`, `task/` | скрипты организаторов — не изменяются |
| `templates/` | Kaggle-ноутбук и `SUBMISSION.md` |
| `assets/` | логотип команды |

---

## 👤 Команда

**MindStream AI** — соло-участник: **Sonia Akhsen**, капитан команды · Актау

Разработка велась с AI-ассистентом [Claude Code](https://claude.com/claude-code).

## 📄 Лицензия

Apache-2.0, см. [`LICENSE`](./LICENSE) и [`NOTICE`](./NOTICE). Шаблон, скрипты `harness/` и тесты — от организаторов кейса (тесты из Soup v0.77.0, Apache-2.0). Данные: [AmanMussa/kazakh-instruction-v2](https://huggingface.co/datasets/AmanMussa/kazakh-instruction-v2) (MIT), перевод [Stanford Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) (CC BY-NC 4.0). Модель: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache-2.0).

---

<a id="kk"></a>
<details>
<summary><b>🇰🇿 Қазақша</b> — ашу үшін басыңыз</summary>

### 💡 Қысқаша

`soup train` бір Kaggle T4-те **Qwen2.5-1.5B-Instruct** моделін (NF4 салмақтары, LoRA r=16) 256 қазақша нұсқаулықта оқытады: базада шамамен 80 с және 14,35 ГБ бейнежад. Біздің Triton-кернелдеріміз Soup-қа бір `solution.apply()` функциясымен қосылады: сол тапсырма, сол сападағы сол адаптер — бірақ жад пен уақыт азырақ.

### 📊 Нәтижелер

Кесте [жоғарыда](#-результаты), әр Kaggle T4 іске қосуынан кейін жаңартылады.

### ⚙️ Не істелді

1. **151 936 токендік сөздік бойынша біріктірілген cross-entropy** ([`solution/head_loss.py`](./solution/head_loss.py)). Толық логиттер ешқашан жадта сақталмайды: токендер 512-ден бөлікпен өңделеді, бір Triton өтімі log-sum-exp, мақсат логиті, энтропия мен argmax есептейді және градиентті сол жерде жазады. Loss, TRL энтропиясы мен дәлдігі және Soup-тың `rewind.jsonl` журналы бір өтімнен алынады.
2. **Ұйымдастырушылардың SwiGLU мысалы** қосулы қалды (~2 ГБ үнемдейді), бірақ біздің кернел болып саналмайды.

### 🚀 Іске қосу

Kaggle: [`templates/kaggle_notebook.ipynb`](./templates/kaggle_notebook.ipynb) импорттаңыз → GPU T4 x2, Internet On → бірінші ұяшыққа `REPO_URL` → **Run All** (~10 минут). Жергілікті командалар [жоғарыда](#-как-запустить).

### 📚 Идеялар көздері

Cut Cross-Entropy (Wijmans et al., 2024), Liger-Kernel fused linear cross-entropy идеясы, online softmax (Milakov & Gimelshein, 2018). Кернел коды нөлден жазылған.

### 👤 Команда

**MindStream AI** — жеке қатысушы: **Sonia Akhsen**, команда капитаны · Ақтау

</details>

<a id="en"></a>
<details>
<summary><b>🇬🇧 English</b> — click to expand</summary>

### 💡 Summary

`soup train` fine-tunes **Qwen2.5-1.5B-Instruct** (NF4 weights, LoRA r=16) on 256 Kazakh instructions on one Kaggle T4: about 80 s and 14.35 GB of GPU memory in the base run. Our Triton kernels plug into Soup through a single `solution.apply()`: the same task, the same adapter of the same quality, in less memory and time.

### 📊 Results

See the [table above](#-результаты); it is updated after every Kaggle T4 run.

### ⚙️ What we built

1. **Fused cross-entropy over the 151,936-token vocabulary** ([`solution/head_loss.py`](./solution/head_loss.py)). The full logits are never kept: tokens go in chunks of 512, one Triton pass computes log-sum-exp, the target logit, entropy and argmax and writes the gradient in place. The loss, TRL's entropy and token accuracy, and Soup's `rewind.jsonl` per-row losses all come from that single pass, with TRL 0.29.1 and Soup 0.77.0 formulas unchanged.
2. **The organizers' SwiGLU example** stays on (~2 GB saved); it does not count as our kernel.

### 🚀 How to run

Kaggle: import [`templates/kaggle_notebook.ipynb`](./templates/kaggle_notebook.ipynb) → GPU T4 x2, Internet On → set `REPO_URL` in the first cell → **Run All** (~10 minutes). Local commands are [above](#-как-запустить).

### 📚 Sources of ideas

Cut Cross-Entropy (Wijmans et al., 2024), the chunked fused linear cross-entropy idea of Liger-Kernel, online softmax (Milakov & Gimelshein, 2018). The kernel code is written from scratch.

### 👤 Team

**MindStream AI** — solo participant: **Sonia Akhsen**, team captain · Aktau, Kazakhstan

</details>
