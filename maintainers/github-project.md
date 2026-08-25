<!-- maintainers/github-project.md -->

# GitHub project board setup

Labels, milestones M1 to M10, and the roadmap issues exist in
`abhiksark/swage`. Creating or changing the project board requires a current
GitHub CLI and the `project` token scope:

```bash
gh auth refresh -s project
gh project create --owner abhiksark --title "Swage Roadmap"
```

Record the printed project number as `N`. Configure the Status field with
Backlog, Ready, In progress, Review, Blocked, and Done, then create these
fields:

```bash
gh project field-create "$N" --owner abhiksark --name "Area" \
  --data-type SINGLE_SELECT \
  --single-select-options "python,mlir,planner,runtime,benchmarks,docs"
gh project field-create "$N" --owner abhiksark --name "Risk" \
  --data-type SINGLE_SELECT --single-select-options "low,medium,high"
gh project field-create "$N" --owner abhiksark --name "Requires GPU" \
  --data-type SINGLE_SELECT --single-select-options "yes,no"
gh project field-create "$N" --owner abhiksark \
  --name "Research uncertainty" --data-type SINGLE_SELECT \
  --single-select-options "none,some,high"
```

Add at most 100 open issues:

```bash
gh issue list --repo abhiksark/swage --state open --limit 100 \
  --json url -q '.[].url' | while read -r issue_url; do
    gh project item-add "$N" --owner abhiksark --url "$issue_url"
  done
```

After creation, place current-milestone issues in Ready and later work in
Backlog. This is maintainer administration, not reader documentation.
