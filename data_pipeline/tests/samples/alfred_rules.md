# Rules: macroeconomic data vintages (ALFRED style)

Evidence: one line per published estimate of one quarter's value, with its release time. The same quarter is estimated several times (advance, second, third, annual update).

1. Latest estimate of a quarter = the most recently published estimate for that quarter at the as-of time.
2. First print (advance estimate) = the earliest published estimate for that quarter.
3. Total revision = latest estimate minus first print, in percentage points.
4. Revision count = number of estimates published after the first print for that quarter (each later release counts, even if the value is unchanged).
5. Published quarters: a quarter is published once its advance estimate is released. A quarter that has ended but has no released estimate yet cannot be answered.
6. Annual updates can revise old quarters, including quarters whose third estimate looked final.
7. As-of rule: the cutoff is a timestamp. A release later on the same day does not exist yet.

## Fact schema (used by the knowledge graph and code-writing methods)

| field | value |
| --- | --- |
| subject | `<ENTITY> \| <series name>` (e.g. `ZETA_GDP \| real GDP growth % SAAR`) |
| attribute | `estimate` |
| period | quarter label, e.g. `2026Q1` |
| value | number (percent, annual rate) |
| effective_at | empty |
| kind | `new` for the advance estimate, `revision` for every later estimate of the same quarter |
