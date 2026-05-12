#!/usr/bin/env bash
printf "Enter prompt: "
IFS= read -r USER_PROMPT

codex --search exec --json \
  -c 'mcp_servers.zoo.command="uvx"' \
  -c 'mcp_servers.zoo.args=["zoo-mcp"]' \
  -c mcp_servers.zoo.env.ZOO_API_TOKEN="$ZOO_API_TOKEN" \
  "you have access to the zoo mcp server, you can use this to take snapshots, export files, web search etc. Write the KCL yourself in main.kcl (which is currently empty) for the following request, and run a snapshot, 'zoo_format_kcl' and 'zoo_lint_and_fix_kcl' at the end to ensure there are no errors produced. The request is: ${USER_PROMPT}." \
  | jq -c --unbuffered 'now as $t | . + {timestamp: ($t|todateiso8601)}' \
  > "codex-run-$(date +%Y%m%d-%H%M%S).jsonl"
