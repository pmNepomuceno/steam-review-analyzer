# Aspect labeling guide

These are the rules for filling in `gold_aspect` in `aspect_labels.csv`. The file comes from `backend/scripts/build_eval_sample.py` and is scored by `backend/scripts/evaluate_aspects.py`. The point of the rules is to make labeling consistent, so every ambiguous sentence is decided by a rule written down here rather than case by case.

## Workflow

- **Label blind.** Label only `aspect_labels.csv` and don't open `aspect_sample_key.csv` (the model's predictions) until you're done.
- **Columns to fill:**
  - `gold_aspect`: exactly one of `performance`, `price`, `bugs`, `story`, `gameplay`, `none`. Capitalization and surrounding spaces don't matter.
  - `ambiguous`: `y` when rule 3 below decided the label. Otherwise leave it blank.
  - `notes`: free text. For `none` sentences about something with no aspect of its own, put that topic's tag here (see "Topics with no aspect").
- **Every row needs a `gold_aspect`.** The evaluation refuses to run while any row is blank.
- **When you're done**, run `python scripts/evaluate_aspects.py` from `backend/` (see the repo `CLAUDE.md` for the venv). It lists every problem at once (blank rows, typos in labels, ids that don't match the key) and writes the report to `docs/eval/aspect_report.txt`.
- **If you use a spreadsheet, save as "CSV UTF-8".** Either `,` or `;` as the separator works, and so does changing the column order. Excel may turn sentences that start with `-`, `+` or `=` into formulas. That's harmless: the evaluation joins on `id` and takes the sentence text from the key file, so only `id`, `gold_aspect`, `ambiguous` and `notes` need to survive.

## Aspects

Label the **topic** of the sentence, never its sentiment. "Runs great" and "runs terribly" are both `performance`.

| label | covers | boundary cases |
|---|---|---|
| `performance` | frame rate, stutter, loading times, optimization, hardware requirements, Steam Deck/laptop performance | A crash is `bugs`, not `performance`. "Unplayable" alone, with no stated cause, is `none` |
| `bugs` | crashes, glitches, broken quests or scripts, save corruption, server and connection problems, disconnects | Server issues are `bugs` (matches the anchors). "Needs polish" with no specific defect is `none` |
| `price` | price, value for money, discounts and sales, refunds requested over value, microtransactions, DLC pricing | "Refunded" with no reason given is `none`. "Worth it" is `price` only when it clearly means money ("worth $70", "worth every penny"), not "worth your time" |
| `story` | plot, characters, writing, dialogue, voice acting, lore, quests *as narrative*, the ending | A quest that's broken is `bugs`. A quest that's boring as a task is `gameplay` |
| `gameplay` | mechanics, combat, controls, puzzles, level design, progression, difficulty, game modes, amount of content, replayability, UI/UX of playing | "Too short" or "not much content" is `gameplay`, but "too short for the price" is `price` (rule 2) |
| `none` | generic verdicts ("great game", "10/10"), recommendations, hours played with no value claim, jokes, memes, meta text, developer/publisher talk, and topics with no aspect | See below |

### Topics with no aspect

Graphics, art style, music, sound design and similar topics have no aspect of their own. Label them `none` and put the topic in `notes`, one short lowercase tag: `graphics`, `audio`, `art`, `multiplayer`, `modding`, `developer`, and so on. The evaluation counts these tags, which shows whether a missing aspect is a real gap.

## Tie-breaking rules

Apply them in order:

1. **Topic, not sentiment.** Decide what the sentence talks about before anything else.
2. **Primary clause wins.** When one clause carries the judgement and the other only justifies or qualifies it, label the judgement. Examples:
   - "worth every penny, been playing for 200 hours" → `price`. The claim is value; the hours are the evidence.
   - "the combat is great because the controls are so tight" → `gameplay`.
   - "can't recommend it until they fix the crashes" → `bugs`. The recommendation hinges on crashes.
3. **Equal-weight clauses: first mentioned wins, and set `ambiguous=y`.** When two aspects are mentioned with equal weight and neither justifies the other, label the first one. Examples:
   - "the story is great and the combat is fun" → `story`, `ambiguous=y`.
   - "performance is bad and it crashes a lot" → `performance`, `ambiguous=y`.

   The evaluation reports metrics both with and without these rows.
4. **Label the unit as written.** Units are fragments of longer reviews. Don't guess what the rest of the review said: "It" or "this" with no referent is `none` unless the unit itself names the topic.
5. **Sarcasm:** label the topic being mocked. "Love paying $70 to beta test" → `price`, since the complaint is being charged for an unfinished product. It is not `ambiguous` because the price complaint is the primary clause.
6. **Non-English or unreadable units** are `none`, with `notes=non-english` or `notes=garbled`.
