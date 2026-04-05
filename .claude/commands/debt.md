---
name: debt
description: Show prioritised open technical debt and architecture recommendations
---

Read TECHNICAL_DEBT.md and show a prioritised summary of open items.

If $ARGUMENTS is empty:
1. Read TECHNICAL_DEBT.md
2. Extract all items marked ⏳ Open
3. Also extract any Architecture Recommendations that have not been acted on
4. Present as two sections:

**Open Technical Debt** (table: #, Item, Priority)
**Architecture Recommendations** (table: Topic, Summary, Effort estimate)

Keep descriptions to one line each. Do not show resolved items.

If $ARGUMENTS is a debt item ID or keyword, show full details for that item only:
- Full description
- Why it matters / risk if left open
- Suggested fix approach
