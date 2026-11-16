param(
    [Parameter(Position=0, Mandatory)]
    [ValidateSet('sync','download','robotwin-prepare','robotwin-submit','robotwin-run','robotwin-status','robotwin-logs','robotwin-report')][string]$Command,
    [Parameter(Position=1)][string]$Value='',
    [ValidateSet('robotwin')][string]$Model='robotwin',
    [ValidatePattern('^[0-9a-f]{40}$')][string]$RobotCommit,
    [ValidateSet('easy','hard','both')][string]$RobotLane='both',
    [ValidateSet('4090','3090')][string]$GpuModel='4090',
    [ValidatePattern('^[a-z0-9_]+$')][string]$RobotTask,
    [ValidateRange(1,72)][int]$RobotHours=48,
    [switch]$RootOnly,
    [string]$SettingsFile
)
$ErrorActionPreference='Stop'
$OutputEncoding=[Text.UTF8Encoding]::new($false)
$repo=(Resolve-Path "$PSScriptRoot/../..").Path
if (-not $SettingsFile) { $SettingsFile="$repo/tools/idac/settings.local.psd1" }
if (-not (Test-Path -LiteralPath $SettingsFile)) { throw 'Copy tools/idac/settings.example.psd1 to settings.local.psd1 and fill in the IDAC paths' }
$cfg=Import-PowerShellDataFile -LiteralPath $SettingsFile
if (-not $cfg.InputRoot) { $cfg.InputRoot=$cfg.DataRoot }
if (-not $cfg.QualifiedDataRoot) { $cfg.QualifiedDataRoot=$cfg.DataRoot }
foreach ($key in @('CodeRoot','RunCodeRoot','DataRoot','InputRoot','QualifiedDataRoot','CondaRoot','EnvPrefix','SimulatorCodeRoot')) {
    if ($cfg[$key] -notmatch '^(/[A-Za-z0-9_.-]+)+$' -or $cfg[$key] -match '/\.\.?(/|$)') { throw "Invalid absolute path in $key" }
}
if ($cfg.SshHost -notmatch '^[A-Za-z0-9_][A-Za-z0-9_.-]*$' -or $cfg.Partition -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Invalid SSH alias or Slurm partition' }
if (-not $cfg.EnvPrefix.StartsWith("$($cfg.CondaRoot)/envs/")) { throw 'EnvPrefix must name a Conda environment under the configured CondaRoot' }
$remoteRepository='https://github.com/MoliaiELS/RLinf-VLA.git'
$branch='codex/xvla'
$sha=(& git -C $repo rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $sha -notmatch '^[0-9a-f]{40}$') { throw 'Could not resolve the local commit' }
if (-not $RobotCommit) { $RobotCommit=$sha }
$snapshot="$($cfg.RunCodeRoot)/$RobotCommit"
$toolDir="$snapshot/tools/xvla/robotwin"
$python="$($cfg.EnvPrefix)/bin/python"

function Invoke-Remote([string]$Script) {
    $envs="export FYP_ROBOTWIN_DATA='$($cfg.DataRoot)' FYP_ROBOTWIN_INPUT_ROOT='$($cfg.InputRoot)' FYP_ROBOTWIN_QUALIFIED_DATA='$($cfg.QualifiedDataRoot)' FYP_ROBOTWIN_ENV='$($cfg.EnvPrefix)' FYP_ROBOTWIN_CODE='$($cfg.SimulatorCodeRoot)' FYP_CONDA_ROOT='$($cfg.CondaRoot)'`n"
    ($envs+$Script).Replace("`r",'') | & ssh -T -o BatchMode=yes -o ConnectTimeout=15 $cfg.SshHost "tr -d '\r' | bash -se"
    if ($LASTEXITCODE -ne 0) { throw "IDAC command failed ($LASTEXITCODE)" }
}

switch ($Command) {
    'sync' {
        if ($RobotCommit -ne $sha) { throw 'Sync deploys the current commit. Use -RobotCommit for job commands.' }
        if ((& git -C $repo branch --show-current) -ne $branch) { throw "Switch to $branch before syncing" }
        if (& git -C $repo status --porcelain) { throw 'Commit local changes before syncing' }
        if ((& git -C $repo remote get-url origin) -ne $remoteRepository) { throw 'The local origin must point to RLinf-VLA' }
        Invoke-Remote @"
code='$($cfg.CodeRoot)'
if [[ -e "`$code" ]]; then
    test -d "`$code/.git"
    if [[ -n `$(git -C "`$code" status --porcelain) ]]; then
        echo 'Commit or stash changes in the server checkout before syncing' >&2
        exit 1
    fi
fi
"@
        & git -C $repo push origin "HEAD:refs/heads/$branch"
        if ($LASTEXITCODE -ne 0) { throw 'Git push failed' }
        $createSnapshot=if ($RootOnly) { 'no' } else { 'yes' }
        Invoke-Remote @"
code='$($cfg.CodeRoot)'
exec 9>"`$code.sync.lock"
flock 9
if [[ ! -e "`$code" ]]; then
    git clone --quiet --single-branch --branch '$branch' '$remoteRepository' "`$code"
fi
test "`$(git -C "`$code" remote get-url origin)" = '$remoteRepository'
test "`$(git -C "`$code" branch --show-current)" = '$branch'
test -z "`$(git -C "`$code" status --porcelain)"
git -C "`$code" fetch origin '$branch'
test "`$(git -C "`$code" rev-parse FETCH_HEAD)" = '$sha'
if ! git -C "`$code" remote get-url upstream >/dev/null 2>&1; then
    git -C "`$code" remote add upstream https://github.com/RLinf/RLinf.git
fi
git -C "`$code" merge --ff-only '$sha'
test "`$(git -C "`$code" rev-parse HEAD)" = '$sha'
if [[ '$createSnapshot' == yes ]]; then
    mkdir -p '$($cfg.RunCodeRoot)'
    test -e '$snapshot' || git -C "`$code" worktree add --quiet --detach '$snapshot' '$sha'
    test "`$(git -C '$snapshot' rev-parse HEAD)" = '$sha'
fi
echo 'RLINF_VLA_SYNC $sha $($cfg.CodeRoot)'
"@
    }
    'download' {
        if ($cfg.InputRoot -ne $cfg.DataRoot) { throw 'Shared InputRoot is read-only. Download assets and weights from its owning checkout.' }
        Invoke-Remote @"
test -x '$python'
test -f '$snapshot/tools/xvla/download.py'
mkdir -p '$($cfg.DataRoot)/logs'
nohup flock -n '$($cfg.DataRoot)/download-robotwin.lock' '$python' '$snapshot/tools/xvla/download.py' '$($cfg.DataRoot)/models' robotwin >> '$($cfg.DataRoot)/logs/download-robotwin.log' 2>&1 < /dev/null &
echo ROBOTWIN_DOWNLOAD_PID=`$!
"@
    }
    'robotwin-prepare' {
        if ($Value -notin @('env','assets','link','easy-preflight')) { throw 'Requires env, assets, link, or easy-preflight' }
        if ($Value -eq 'assets' -and $cfg.InputRoot -ne $cfg.DataRoot) { throw 'Shared InputRoot is read-only. Download assets from its owning checkout.' }
        Invoke-Remote @"
test -f '$toolDir/prepare.sh'
mkdir -p '$($cfg.DataRoot)/logs'
nohup flock -n '$($cfg.DataRoot)/prepare-$Value.lock' bash '$toolDir/prepare.sh' '$Value' >> '$($cfg.DataRoot)/logs/prepare-$Value.log' 2>&1 < /dev/null &
echo ROBOTWIN_PREPARE_PID=`$!
"@
    }
    'robotwin-submit' {
        if ($Value -notin @('probe','easy-preflight','smoke','coverage','formal','extension')) { throw 'Requires probe, easy-preflight, smoke, coverage, formal, or extension' }
        if ($Value -eq 'easy-preflight' -and $RobotLane -ne 'easy') { throw 'Easy preflight requires -RobotLane easy' }
        Invoke-Remote "bash '$toolDir/submit.sh' '$Value' '$($cfg.Partition)' '$GpuModel' '$RobotLane' '$RobotHours' '$RobotTask'"
    }
    'robotwin-run' {
        if ($Value -and $Value -notmatch '^\d+$') { throw 'Build job ID must be numeric' }
        $build=if ($Value) { "--build-job '$Value'" } else { '' }
        Invoke-Remote @"
test -f '$toolDir/dispatch.py'
mkdir -p '$($cfg.DataRoot)/dispatch/$RobotCommit'
nohup flock -n '$($cfg.DataRoot)/dispatch.lock' '$python' -u '$toolDir/dispatch.py' $build --partition '$($cfg.Partition)' --gpu '$GpuModel' --hours '$RobotHours' >> '$($cfg.DataRoot)/dispatch/$RobotCommit/dispatch.log' 2>&1 < /dev/null &
echo ROBOTWIN_DISPATCH_PID=`$!
"@
    }
    'robotwin-status' {
        Invoke-Remote @'
squeue -u "$(id -un)" -o '%i|%j|%T|%M|%b|%R'
for log in "$FYP_ROBOTWIN_DATA"/evidence/jobs.tsv "$FYP_ROBOTWIN_DATA"/dispatch/*/dispatch.log "$FYP_ROBOTWIN_DATA"/logs/{prepare-*,download-robotwin}.log; do
    if [[ -f "$log" ]]; then printf '%s\n' "$log"; tail -n 4 "$log"; fi
done
'@
    }
    'robotwin-logs' {
        if ($Value -notmatch '^(\d+|prepare-(env|assets|link|easy-preflight)|download-robotwin)$') { throw 'Requires a job ID or preparation log name' }
        Invoke-Remote "for ext in log out err; do if test -f '$($cfg.DataRoot)/logs/$Value'.`$ext; then tail -n 45 '$($cfg.DataRoot)/logs/$Value'.`$ext; fi; done"
    }
    'robotwin-report' {
        if ($Value -notin @('smoke','formal')) { throw 'Requires smoke or formal' }
        Invoke-Remote "'$python' '$toolDir/summarize.py' --input '$($cfg.DataRoot)/panels/$RobotCommit' --output '$($cfg.DataRoot)/reports/$RobotCommit/$Value' --phase '$Value'"
    }
}
