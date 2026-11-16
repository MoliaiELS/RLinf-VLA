"""Record the actual GPU, inputs, dependency identities and immutable commits."""
import csv
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys


QUALIFIED_VERSIONS = {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "transformers": "4.51.3",
    "sapien": "3.0.0b1",
    "mplib": "0.2.1",
    "nvidia-curobo": "0.7.8",
    "warp-lang": "1.6.0",
    "numpy": "1.26.3",
    "scipy": "1.15.0",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def commit(path):
    subprocess.run(["git", "-C", str(path), "diff", "--quiet", "HEAD"], check=True)
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def repository_identity(repo, upstream):
    head = commit(repo)
    ancestry = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", upstream, head],
        check=False,
    )
    if ancestry.returncode:
        raise ValueError(f"Worktree {head} does not descend from pinned RLinf {upstream}")
    return {"path": str(repo), "commit": head, "upstream_commit": upstream}


def qualified_environment(data, *, require_native=True):
    prefix = Path(os.environ["FYP_ROBOTWIN_ENV"]).resolve()
    if Path(sys.prefix).resolve() != prefix or sys.version_info[:2] != (3, 11):
        raise ValueError(f"Expected the Python 3.11 RoboTwin environment at {prefix}; found {sys.prefix}")
    evidence = data / "evidence"
    if not (evidence / "build.complete").is_file():
        raise ValueError(f"CuRobo build receipt is missing in {evidence}")
    versions = {name: metadata.version(name) for name in QUALIFIED_VERSIONS}
    for name, expected in QUALIFIED_VERSIONS.items():
        if versions[name] != expected:
            raise ValueError(f"RoboTwin requires {name}=={expected}; found {versions[name]}")
    patches = json.loads((evidence / "install-patches.json").read_text())
    expected_paths = {"sapien/wrapper/urdf_loader.py", "mplib/planner.py"}
    found_paths = set()
    for patch in patches:
        path = Path(patch["path"]).resolve()
        if not path.is_relative_to(prefix) or digest(path) != patch["sha256"]:
            raise ValueError(f"Install patch receipt does not verify inside {prefix}: {path}")
        found_paths.add(path.as_posix().split("/site-packages/", 1)[-1])
    if found_paths != expected_paths:
        raise ValueError(f"Unexpected RoboTwin install patch files: {sorted(found_paths)}")
    result = {
        "environment": versions,
        "pip_freeze_sha256": digest(evidence / "requirements.freeze.txt"),
        "install_patches": patches,
    }
    if require_native:
        qualification = json.loads((evidence / "environment-qualification.json").read_text())
        extension = Path(qualification["geom_cu"]["path"]).resolve()
        if (qualification["prefix"] != str(prefix) or qualification["environment"] != versions
                or qualification["pip_freeze_sha256"] != result["pip_freeze_sha256"]
                or not extension.is_relative_to(prefix)
                or digest(extension) != qualification["geom_cu"]["sha256"]):
            raise ValueError(f"RoboTwin environment differs from its native qualification: {prefix}")
        result["qualification"] = qualification
    return result


def write_text_atomic(path, content):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def prepare_environment(data):
    source = Path(os.environ.get("FYP_ROBOTWIN_QUALIFIED_DATA", str(data))).resolve()
    qualified = qualified_environment(source, require_native=False)
    # A compiled-wheel import verifies binary loading without creating a scene
    # or allocating a GPU. Rendering and model inference are covered by Slurm.
    import torch
    from curobo.curobolib import geom_cu

    prefix = Path(sys.prefix).resolve()
    extension = Path(geom_cu.__file__).resolve()
    if torch.version.cuda != "12.4" or not extension.is_relative_to(prefix):
        raise ValueError(f"Unexpected Torch CUDA build or CuRobo extension: {torch.version.cuda}, {extension}")
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    evidence = data / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    previous_build = (source / "evidence/build.complete").read_text().strip()
    if not previous_build.isdecimal():
        raise ValueError(f"Invalid qualification build job in {source / 'evidence/build.complete'}")
    details = {
        **qualified,
        "prefix": str(prefix),
        "python": sys.version,
        "cuda": torch.version.cuda,
        "geom_cu": {"path": str(extension), "sha256": digest(extension)},
        "source_evidence": str(source / "evidence"),
        "source_build_job": previous_build,
        "pip_freeze_sha256": hashlib.sha256(frozen.encode()).hexdigest(),
    }
    write_text_atomic(evidence / "requirements.freeze.txt", frozen)
    write_text_atomic(evidence / "install-patches.json", json.dumps(qualified["install_patches"], indent=2) + "\n")
    write_text_atomic(evidence / "environment-qualification.json", json.dumps(details, indent=2) + "\n")
    write_text_atomic(evidence / "build.complete", previous_build + "\n")
    print(f"ROBOTWIN_ENV_READY {prefix}", flush=True)


def receipt(path, expected, *, verify_all=False):
    obj = json.loads((path / "verified.json").read_text())
    if obj["revision"] != expected or not obj["files"]:
        raise ValueError(f"Unverified snapshot: {path}")
    for item in obj["files"]:
        file = path / item["file"]
        if not file.resolve().is_relative_to(path.resolve()):
            raise ValueError(f"Receipt path escapes snapshot: {file}")
        if file.stat().st_size != item["bytes"]:
            raise ValueError(f"File size changed after verification: {file}")
        if (verify_all or file.suffix == ".json") and digest(file) != item["sha256"]:
            raise ValueError(f"File checksum changed after verification: {file}")
    return {"path": str(path), **obj}


def asset_receipt(inputs, revision, *, verify_all=False):
    if __package__:
        from .download_assets import ARCHIVES, REPO
    else:
        from download_assets import ARCHIVES, REPO

    if not (inputs / "sim-assets" / revision / ".extracted").is_file():
        raise ValueError(f"Pinned simulator assets have not finished extracting under {inputs}")
    assets = receipt(inputs / "archives" / revision, revision, verify_all=verify_all)
    recorded = {item["file"]: (item["bytes"], item["sha256"]) for item in assets["files"]}
    if assets["repo_id"] != REPO or recorded != ARCHIVES:
        raise ValueError(f"Unexpected simulator archive checksums: {assets['path']}")
    return assets


def main(mode):
    data = Path(os.environ["FYP_ROBOTWIN_DATA"]).resolve()
    inputs = Path(os.environ.get("FYP_ROBOTWIN_INPUT_ROOT", str(data))).resolve()
    rev = dict(line.split("=", 1) for line in Path(__file__).with_name("runtime.env").read_text().splitlines())
    if mode == "prepare-environment":
        prepare_environment(data)
        return
    if mode == "verify-assets":
        assets = asset_receipt(inputs, rev["ASSETS_REV"], verify_all=True)
        write_text_atomic(data / "evidence/shared-assets-verified.json", json.dumps(assets, indent=2) + "\n")
        print(f"ROBOTWIN_ASSETS_READY {inputs}", flush=True)
        return
    qualified = qualified_environment(data)

    import torch
    import sapien
    import curobo
    import mplib
    import transformers

    if torch.cuda.device_count() != 1:
        raise ValueError("RoboTwin jobs require exactly one Slurm assigned CUDA GPU")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    uuid = str(properties.uuid).lower().removeprefix("gpu-")
    if torch.version.cuda != "12.4":
        raise ValueError(f"Expected the qualified Torch CUDA 12.4 build; found {torch.version.cuda}")
    out = Path(os.environ["FYP_ROBOTWIN_JOB"])
    (out / "allocated_gpu_uuids.json").write_text(json.dumps([uuid]) + "\n")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,pci.bus_id,driver_version", "--format=csv,noheader"], text=True)
    ((_, pci, driver),) = (tuple(map(str.strip, row)) for row in csv.reader(query.splitlines())
                          if row[0].strip().lower().removeprefix("gpu-") == uuid)
    match = re.fullmatch(r"[\da-fA-F]+:([\da-fA-F]{2}):([\da-fA-F]{2})\.([\da-fA-F])", pci)
    if not match:
        raise ValueError(f"Cannot identify Slurm GPU's PCI address: {pci}")
    # SAPIEN 3's parser accepts the hexadecimal PCI bus:device.function form.
    device = f"pci:{match[1]}:{match[2]}.{match[3]}"
    (out / "vulkan-device.txt").write_text(device + "\n")
    repo = Path(__file__).resolve().parents[3]
    robot = Path(os.environ["FYP_ROBOTWIN_ROOT"])
    code_paths = {name: path for name, path in (
        ("ROBOTWIN", robot),
        ("XVLA", Path(os.environ["FYP_ROBOTWIN_XVLA"])), ("CUROBO", Path(os.environ["FYP_ROBOTWIN_CUROBO"])),
    )}
    code = {name: {"path": str(path), "commit": commit(path)} for name, path in code_paths.items()}
    for name in code_paths:
        if code[name]["commit"] != rev[f"{name}_REV"]:
            raise ValueError(f"Wrong pinned {name} checkout: {code[name]}")
    identity = repository_identity(repo, rev["RLINF_REV"])
    code["repository"] = {"path": identity["path"], "commit": identity["commit"]}
    code["RLINF"] = identity
    files = sorted((robot / "task_config").glob("*.yml"))
    manifest = dict(job_id=os.environ["SLURM_JOB_ID"], hostname=os.uname().nodename, mode=mode, code=code,
                    python=sys.version, prefix=sys.prefix, cuda=torch.version.cuda,
                    **qualified, imports={m.__name__: m.__file__ for m in (torch, sapien, curobo, mplib, transformers)},
                    gpu=dict(uuid=uuid, model=properties.name, pci=pci, driver=driver, vulkan_device=device,
                             slurm_visible=os.environ["CUDA_VISIBLE_DEVICES"], total_bytes=properties.total_memory),
                    camera=dict(sensor="D435", near_m=0.1, far_m=100, shader="rt", samples=32),
                    configs={str(f.relative_to(robot)): digest(f) for f in files})
    if mode != "probe":
        assets = asset_receipt(inputs, rev["ASSETS_REV"])
        if inputs != data:
            verified = json.loads((data / "evidence/shared-assets-verified.json").read_text())
            if assets != verified:
                raise ValueError(f"Shared simulator assets differ from the verified snapshot at {inputs}")
        embodiment = robot / "assets/embodiments/aloha-agilex"
        manifest.update(assets=assets,
                        embodiment={f.name: digest(f) for f in sorted(embodiment.glob("*.yml"))},
                        model=receipt(Path(os.environ["FYP_ROBOTWIN_MODEL"]), rev["MODEL_REV"]))
    (out / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"ROBOTWIN_RUNTIME {out / 'runtime.json'} {device}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
