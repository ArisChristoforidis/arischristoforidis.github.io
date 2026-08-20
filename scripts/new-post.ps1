[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Title
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$slug = $Title.ToLowerInvariant() -replace "[^a-z0-9]+", "-" -replace "^-|-$", ""

if ([string]::IsNullOrWhiteSpace($slug)) {
    throw "The title must contain at least one letter or number."
}

$target = Join-Path $repoRoot ("src\posts\{0}.md" -f $slug)
if (Test-Path -LiteralPath $target) {
    throw "A post already exists at $target"
}

$titleForYaml = $Title.Replace("'", "''")
$date = Get-Date -Format "yyyy-MM-dd"
$content = @"
---
layout: "post.njk"
title: '$titleForYaml'
date: $date
tags: ["posts"]
summary: ""
draft: true
---

Write here.
"@

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
[System.IO.File]::WriteAllText($target, $content.TrimStart(), [System.Text.UTF8Encoding]::new($false))
Write-Host "Created draft: $target"
