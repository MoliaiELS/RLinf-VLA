#!/usr/bin/env bash
set -euo pipefail
: "${FYP_CONDA_ROOT:?}" "${FYP_ROBOTWIN_DATA:?}" "${FYP_ROBOTWIN_ENV:?}" "${FYP_ROBOTWIN_CODE:?}"
tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$tool_dir/runtime.env"
data="$FYP_ROBOTWIN_DATA"
input_root="${FYP_ROBOTWIN_INPUT_ROOT:-$data}"
prefix="$FYP_ROBOTWIN_ENV"
code="$FYP_ROBOTWIN_CODE"
python="$prefix/bin/python"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
if [[ ! -x "$python" ]]; then
    echo "Set FYP_ROBOTWIN_ENV to the qualified RoboTwin environment; missing $python" >&2
    exit 1
fi
mkdir -p "$data"/{cache/pip,evidence,logs} "$code"
data="$(realpath "$data")"
input_root="$(realpath -m "$input_root")"
export FYP_ROBOTWIN_DATA="$data" FYP_ROBOTWIN_INPUT_ROOT="$input_root"

checkout() {
    local url="$1" rev="$2" path="$3" ref="${4:-$2}"
    if [[ ! -d "$path/.git" ]]; then git init -q "$path"; fi
    git -C "$path" cat-file -e "$rev^{commit}" 2>/dev/null || git -C "$path" fetch --depth=1 "$url" "$ref"
    if [[ "$(git -C "$path" rev-parse --verify HEAD 2>/dev/null || true)" != "$rev" ]]; then
        git -C "$path" checkout -q --detach "$rev"
    fi
    test "$(git -C "$path" rev-parse HEAD)" = "$rev"
    git -C "$path" diff --quiet HEAD
}

case "${1:?prepare mode}" in
    assets)
        if [[ "$input_root" != "$data" ]]; then
            # Shared inputs belong to the existing run. Wait for its writer to
            # finish, and verify the source archives without writing there.
            "$python" "$tool_dir/job_runtime.py" verify-assets
            exit
        fi
        archives="$data/archives/$ASSETS_REV"
        tree="$data/sim-assets/$ASSETS_REV"
        "$python" -u "$tool_dir/download_assets.py" "$archives" "$ASSETS_REV"
        if [[ ! -f "$tree/.extracted" ]]; then
            mkdir -p "$tree"
            for asset in embodiments objects background_texture; do
                unzip -q -o "$archives/$asset.zip" -d "$tree"
                test -d "$tree/$asset"
            done
            touch "$tree/.extracted"
        fi
        echo ROBOTWIN_ASSETS_READY
        ;;
    env)
        checkout https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_REV" "$code/RoboTwin-$ROBOTWIN_REV"
        checkout https://github.com/2toinf/X-VLA.git "$XVLA_REV" "$code/X-VLA-$XVLA_REV"
        checkout https://github.com/NVlabs/curobo.git "$CUROBO_REV" "$code/curobo-$CUROBO_REV" refs/tags/v0.7.8
        "$python" "$tool_dir/job_runtime.py" prepare-environment
        echo ROBOTWIN_SOURCES_READY
        ;;
    easy-preflight)
        exec 9>"$data/easy-preflight-prepare.lock"
        flock 9
        source_checkout="$code/RoboTwin-$ROBOTWIN_REV"
        robot="$code/RoboTwin-easy-preflight-$ROBOTWIN_REV"
        checkout "$source_checkout" "$ROBOTWIN_REV" "$robot"
        FYP_ROBOTWIN_ROOT="$robot" "$python" "$tool_dir/job_runtime.py" prepare-easy-assets
        ;;
    link)
        robot="$(realpath "$code/RoboTwin-$ROBOTWIN_REV")"
        tree="$input_root/sim-assets/$ASSETS_REV"
        test -f "$tree/.extracted"
        exec 9>"$data/asset-links.lock"
        flock 9
        embodiment_root="$tree/embodiments"
        if [[ "$input_root" != "$data" ]]; then
            # The official setup script writes generated YAMLs into the
            # embodiment tree. Extract a private copy from the verified ZIP.
            if [[ ! -s "$data/evidence/shared-assets-verified.json" ]]; then
                "$python" "$tool_dir/job_runtime.py" verify-assets
            fi
            private_tree="$data/sim-assets/$ASSETS_REV"
            case "$(realpath -m "$private_tree")" in
                "$data"/*) ;;
                *) echo "Private simulator assets must remain under $data" >&2; exit 1 ;;
            esac
            embodiment_root="$private_tree/embodiments"
            mkdir -p "$private_tree"
            if [[ ! -f "$embodiment_root/.archive-revision" ]]; then
                if [[ -e "$embodiment_root" || -L "$embodiment_root" ]]; then
                    echo "Existing embodiment tree lacks qualification: $embodiment_root" >&2
                    exit 1
                fi
                archive="$input_root/archives/$ASSETS_REV/embodiments.zip"
                echo "6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6  $archive" | sha256sum --check --status
                staging="$(mktemp -d "$private_tree/.embodiments.XXXXXXXX")"
                unzip -q "$archive" 'embodiments/*' -d "$staging"
                test -d "$staging/embodiments"
                printf '%s\n' "$ASSETS_REV" > "$staging/embodiments/.archive-revision"
                mv -T -- "$staging/embodiments" "$embodiment_root"
                rmdir -- "$staging"
            fi
            test "$(cat "$embodiment_root/.archive-revision")" = "$ASSETS_REV"
            test ! -L "$embodiment_root"
        fi
        for asset in embodiments objects background_texture; do
            target="$tree/$asset"
            if [[ "$asset" == embodiments ]]; then target="$embodiment_root"; fi
            test -d "$target"
            link="$robot/assets/$asset"
            if [[ ! -e "$link" && ! -L "$link" ]]; then ln -s "$target" "$link"; fi
            test "$(realpath "$link")" = "$(realpath "$target")"
        done
        if [[ -f "$embodiment_root/.config-root" ]]; then
            test "$(cat "$embodiment_root/.config-root")" = "$robot"
        else
            (cd "$robot" && "$python" script/update_embodiment_config_path.py)
        fi
        for arm in left right; do
            test -s "$robot/assets/embodiments/aloha-agilex/curobo_${arm}.yml"
        done
        printf '%s\n' "$robot" > "$embodiment_root/.config-root"
        echo ROBOTWIN_LINK_READY
        ;;
    *) exit 2 ;;
esac
