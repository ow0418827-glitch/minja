"""Run the optimized PowerShell profile through its real delegate with a mock runtime."""
import json
from pathlib import Path
import subprocess

import pytest

from src.paths import PROJECT
from src.tests.test_ssa_integration import powershell_command
from src.vision.ssa_attack import ssa_settings


@pytest.mark.parametrize("override", [False, True])
def test_optimized_profile_delegates_image_only_stage_and_frozen_options(tmp_path, override):
    # Only the Python executable is substituted in a temporary copy of the
    # real delegate. Both PowerShell entrypoints run in a separate process.
    profile = tmp_path / "run_optimized_ssa_126.ps1"
    profile.write_text((PROJECT / profile.name).read_text(encoding="utf-8"), encoding="utf-8")
    mock = tmp_path / "capture runtime.ps1"
    mock.write_text("Write-Output ('__CAPTURE__' + (ConvertTo-Json -InputObject @($args) -Compress))\n"
                    "$global:LASTEXITCODE = 0\n", encoding="utf-8")
    delegate = (PROJECT / "run_old_b_frequency_126.ps1").read_text(encoding="utf-8")
    binding = "$pipelinePython = 'D:\\Anaconda\\envs\\minja-vision\\python.exe'"
    assert delegate.count(binding) == 1
    delegate = delegate.replace(binding, "$pipelinePython = '" + str(mock).replace("'", "''") + "'")
    (tmp_path / "run_old_b_frequency_126.ps1").write_text(delegate, encoding="utf-8")
    config = tmp_path / "src/configs/ssa_fast.json"
    config.parent.mkdir(parents=True)
    config.write_text((PROJECT / "src/configs/ssa_fast.json").read_text(encoding="utf-8"), encoding="utf-8")

    arguments = []
    root = tmp_path / "new image output"
    source = tmp_path / "original history"
    if override:
        custom = tmp_path / "custom controls with spaces.json"
        custom.write_text(config.read_text(encoding="utf-8"), encoding="utf-8")
        config = custom
        arguments = ["-Stage", "Optimize", "-Root", str(root), "-SourceRoot", str(source),
                     "-SsaOptions", str(config), "-AttackSteps", "280", "-SsaSamples", "10",
                     "-IndexText", "semantic", "-KeyImageWeight", "0.8",
                     "-QueryImageWeight", "0.7", "-Device", "cpu"]
    completed = subprocess.run(powershell_command(profile, *arguments), capture_output=True,
                               text=True, encoding="utf-8", check=True)
    captures = [line.removeprefix("__CAPTURE__") for line in completed.stdout.splitlines()
                if line.startswith("__CAPTURE__")]
    assert len(captures) == 1
    forwarded = json.loads(captures[0])
    assert forwarded[:6] == ["-X", "utf8", "-u", "-m", "src.experiments.reused_frequency_pipeline", "optimize"]
    def value(flag):
        return forwarded[forwarded.index(flag) + 1]
    assert value("--method") == "ssa"
    assert Path(value("--ssa-options")) == config
    assert int(value("--attack-steps")) == (280 if override else 300)
    assert int(value("--ssa-samples")) == (10 if override else 5)
    assert value("--index-text") == ("semantic" if override else "full")
    assert float(value("--key-image-weight")) == (0.8 if override else 0.9)
    assert float(value("--query-image-weight")) == (0.7 if override else 0.9)
    if override:
        assert Path(value("--root")) == root
        assert Path(value("--source-root")) == source
        assert value("--device") == "cpu"
    controls = json.loads(config.read_text(encoding="utf-8"))
    settings = ssa_settings(ssa_options=controls)
    assert settings["ssa_early_stop"] and settings["ssa_min_steps"] == 200
    assert settings["ssa_window"] == settings["ssa_check_every"] == 50
    assert settings["ssa_patience"] == 2 and settings["ssa_min_delta"] == 0.002
