# Rules: broker actions (price targets and ratings)

Evidence: one line per broker note, stock split or broker rename, each with the time it became public.

1. Active broker: has published a note on the stock within 12 months (365 days) before the as-of time. Ignore stale brokers.
2. Split adjustment: express every price target in post-split dollars. A target published before a split is divided by the split ratio, but only if the split is known by the as-of time.
3. Renamed broker: a broker that changes its name is the same broker (for example, Cowen & Co. = TD Cowen once the rename is announced).
4. Buy-equivalent ratings: Buy, Strong Buy, Outperform, Overweight, Positive, Market Outperform. Everything else is not Buy-equivalent.
5. Count rule: a target change is a note that raises or lowers the price target. Reiterations, unchanged targets, splits, renames and inferred revisions (rule 6) do not count.
6. Missed-revision rule: if a note says the broker moved its target "from $X" and $X differs from the broker's last recorded target (after split adjustment), the broker changed its target to $X immediately after its previous note; we never received that note. This is only known from the moment the revealing note is published.
7. As-of rule: the cutoff is a timestamp. A note published later on the same day does not exist yet.
8. Consensus = median of the latest split-adjusted target of each active broker.

## Fact schema (used by the knowledge graph and code-writing methods)

Every fact has six fields: subject, attribute, period, value, effective_at, kind.
Inside a subject, " | " (space, vertical bar, space) is an ordinary character separating the ticker from the broker name. Never escape it.

- Broker note: up to two facts, both with subject = ticker | broker name (canonical name = the first name the broker used), e.g. AUR | Atlas Securities
  - attribute = price_target_usd, value = the target as stated in the note (NOT split-adjusted)
  - attribute = rating, value = the rating label
  - period = empty; effective_at = the note time; kind = new, or corroborate for an unchanged reiteration, or inferred for a missed revision (rule 6; its effective_at = the broker's previous note time + 1 second)
- Split: subject = the ticker alone, e.g. AUR; attribute = split_ratio; value = the ratio (e.g. 2); effective_at = split time; kind = event
- Rename: subject = ticker | old broker name; attribute = alias; value = the new name; effective_at = rename time; kind = event

Note for this dataset: a stated "from $X" within 0.5% of the broker's last recorded target (rounding after a split) is not a missed revision.
