# Issue Tracker: GitHub Issues

Issues for this repository are tracked in GitHub Issues at:

```
https://github.com/mseijse01/coffee-text-analytics/issues
```

## Integration with Engineering Skills

The following skills read from and write to this issue tracker:

- **`to-tickets`**: Converts research findings, bugs, and TODOs into GitHub issues
- **`to-spec`**: Splits issues into executable tasks with acceptance criteria

## Workflow

When creating issues:

1. **Bugs**: Describe reproduction steps, expected vs. actual behavior, and environment
2. **Features/Enhancements**: Explain the use case and acceptance criteria
3. **Chores/Refactoring**: Link to related code locations and explain the motivation

Use the `gh issue create` CLI to create issues programmatically:

```bash
gh issue create --title "Title" --body "Description" --label "label1,label2"
```

## External PRs

PRs as a request surface: **off**. External pull requests are not treated as part of the triage queue. Change this to **on** here if that should change.
