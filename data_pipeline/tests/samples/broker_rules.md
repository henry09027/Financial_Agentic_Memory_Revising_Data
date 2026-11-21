# Rules: broker actions (price targets and ratings)

Evidence: one line per broker note, stock split or broker rename, each with the time it became public.

1. Active broker: has published a note on the stock within 12 months (365 days) before the as-of time. Ignore stale brokers.
2. Split adjustment: express every price target in post-split dollars. A target published before a split is divided by the split ratio, but only if the split is known by the as-of time.
3. Renamed broker: a broker that changes its name is the same broker (for example, Cobalt & Co. = TD Cobalt once the rename is announced).
4. Buy-equivalent ratings: Buy, Strong Buy, Outperform, Overweight, Positive, Market Outperform. Everything else is not Buy-equivalent.
5. Count rule: a target change is a note that raises or lowers the price target. Reiterations, unchanged targets, splits, renames and inferred revisions (rule 6) do not count.
6. Missed-revision rule: if a note says the broker moved its target "from $X" and $X differs from the broker's last recorded target (after split adjustment), the broker changed its target to $X immediately after its previous note; we never received that note. This is only known from the moment the revealing note is published.
7. As-of rule: the cutoff is a timestamp. A note published later on the same day does not exist yet.
8. Consensus = median of the latest split-adjusted target of each active broker.

## Fact schema (used by the knowledge graph and code-writing methods)

| field | broker notes | split | rename |
| --- | --- | --- | --- |
| subject | `<TICKER> \| <broker name, canonical = first name used>` | `<TICKER>` | `<TICKER> \| <old broker name>` |
| attribute | `price_target_usd` (as stated, not split-adjusted) and `rating` | `split_ratio` | `alias` |
| period | empty | empty | empty |
| value | number / rating label | ratio (e.g. 2) | new name |
| effective_at | note time (inferred revisions: previous note time + 1 second) | split time | rename time |
| kind | `new`, `inferred`, `corroborate` (unchanged reiteration) | `event` | `event` |
