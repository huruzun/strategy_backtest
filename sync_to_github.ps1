Param(
  [string]$GithubUser,
  [string]$Email,
  [string]$RepoName = "strategy_backtest",
  [string]$Token = ""
)
$ErrorActionPreference = "Stop"
$repoPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoPath
try {
  git --version | Out-Null
} catch {
  Write-Error "git not installed"
  exit 1
}
git config --global user.name $GithubUser
git config --global user.email $Email
if (!(Test-Path (Join-Path $repoPath ".git"))) {
  git init
}
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne "main") {
  git branch -M main
}
git add -A
$hasChanges = $false
$diff = git diff --cached --name-only
if ($diff) {
  $hasChanges = $true
}
if ($hasChanges) {
  git commit -m "sync"
}
$remoteList = (& git remote)
$remoteExists = $false
if ($remoteList) {
  $remoteExists = ($remoteList -split "`n") -contains "origin"
}
  $remoteUrl = "https://github.com/$GithubUser/$RepoName.git"
  if (-not $remoteExists) {
    if ($Token -ne "") {
      $headers = @{ Authorization = "token $Token"; "User-Agent" = "TraeSync" }
      $body = @{ name = $RepoName; private = $false } | ConvertTo-Json
      try {
        Invoke-RestMethod -Method Post -Uri "https://api.github.com/user/repos" -Headers $headers -Body $body
      } catch {
      }
    $authUrl = "https://" + $GithubUser + ":" + $Token + "@github.com/" + $GithubUser + "/" + $RepoName + ".git"
    git remote add origin $authUrl
    } else {
      git remote add origin $remoteUrl
    }
  } else {
    if ($Token -ne "") {
      $authUrl = "https://" + $GithubUser + ":" + $Token + "@github.com/" + $GithubUser + "/" + $RepoName + ".git"
      git remote set-url origin $authUrl
    } else {
      git remote set-url origin $remoteUrl
    }
  }
try {
  git push -u origin main
} catch {
}
if ($Token -ne "") {
  git remote set-url origin $remoteUrl
}
Write-Output "done"
