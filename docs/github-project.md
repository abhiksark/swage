# GitHub project board setup

Labels, milestones M1–M10, and the 25 roadmap issues already exist in
`abhiksark/swage` (created 2026-08-18). The project board could not be
created from the setup machine (its `gh` predates `gh project`, and
Projects v2 needs the `project` token scope). Run the following once with
a current `gh` (`gh auth refresh -s project`):

```bash
# Create the board
gh project create --owner abhiksark --title "Swage Roadmap"
# Note the printed number, then set N below
N=<number>

# Status columns: Backlog / Ready / In progress / Review / Blocked / Done
# (Projects v2 ships Todo/In Progress/Done; rename and extend the
# built-in Status field in the web UI, which is faster than the API.)

# Custom fields
gh project field-create $N --owner abhiksark --name "Area" \
  --data-type SINGLE_SELECT \
  --single-select-options "python,mlir,planner,runtime,benchmarks,docs"
gh project field-create $N --owner abhiksark --name "Risk" \
  --data-type SINGLE_SELECT --single-select-options "low,medium,high"
gh project field-create $N --owner abhiksark --name "Requires GPU" \
  --data-type SINGLE_SELECT --single-select-options "yes,no"
gh project field-create $N --owner abhiksark --name "Research uncertainty" \
  --data-type SINGLE_SELECT --single-select-options "none,some,high"
# Milestone is already a native field once issues are added.

# Add every open issue to the board
gh issue list --repo abhiksark/swage --state open --limit 100 \
  --json url -q '.[].url' | while read -r url; do
    gh project item-add $N --owner abhiksark --url "$url"
  done
```

After creation, place M1 issues in **Ready** and everything else in
**Backlog**.
