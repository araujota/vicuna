from __future__ import annotations

import os
import stat
import subprocess
import textwrap
from pathlib import Path


REPO_ROOT = Path("/Users/tyleraraujo/vicuna")
INSTALL_SCRIPT = REPO_ROOT / "tools/ops/install-vicuna-system-service.sh"
REBUILD_SCRIPT = REPO_ROOT / "tools/ops/rebuild-vicuna-runtime.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _fake_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    _write_executable(
        bin_dir / "getent",
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            if [[ "${1:-}" == "group" ]]; then
              exit 0
            fi
            if [[ "${1:-}" == "passwd" && "${2:-}" == "tester" ]]; then
              echo "tester:x:1000:1000::/home/tester:/bin/bash"
              exit 0
            fi
            exit 1
            """
        ),
    )

    _write_executable(
        bin_dir / "id",
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            if [[ "${1:-}" == "-u" && "${2:-}" == "tester" ]]; then
              echo 1000
              exit 0
            fi
            if [[ "${1:-}" == "-u" ]]; then
              echo 995
              exit 0
            fi
            exit 0
            """
        ),
    )

    _write_executable(
        bin_dir / "systemctl",
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            scope="system"
            if [[ "${1:-}" == "--user" ]]; then
              scope="user"
              shift
            fi
            cmd="${1:-}"
            unit="${2:-}"
            if [[ "$cmd" == "cat" ]]; then
              if [[ "$scope" == "system" && "${FAKE_SYSTEM_UNIT_PRESENT:-0}" == "1" ]]; then
                cat <<EOF
            [Service]
            WorkingDirectory=${FAKE_SYSTEM_RUNTIME_ROOT:-/tmp/system-root}
            ExecStart=${FAKE_SYSTEM_RUNTIME_ROOT:-/tmp/system-root}/tools/ops/run-vicuna-runtime.sh
            EOF
                exit 0
              fi
              if [[ "$scope" == "user" && "${FAKE_USER_UNIT_PRESENT:-0}" == "1" ]]; then
                cat <<EOF
            [Service]
            WorkingDirectory=${FAKE_USER_RUNTIME_ROOT:-/tmp/user-root}
            ExecStart=${FAKE_USER_RUNTIME_ROOT:-/tmp/user-root}/tools/ops/run-vicuna-runtime.sh
            EOF
                exit 0
              fi
              exit 1
            fi
            exit 0
            """
        ),
    )

    for name in ["node", "ffmpeg", "groupadd", "useradd", "usermod", "setfacl", "curl", "ss", "sudo"]:
        _write_executable(
            bin_dir / name,
            "#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n",
        )

    chrome = bin_dir / "google-chrome-stable"
    _write_executable(chrome, "#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n")
    return bin_dir


def _base_env(bin_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["VICUNA_INTERACTIVE_OWNER"] = "tester"
    env["VICUNA_INTERACTIVE_HOME"] = "/home/tester"
    env["TELEGRAM_BRIDGE_NODE_BIN"] = str(bin_dir / "node")
    env["TELEGRAM_BRIDGE_FFMPEG_BIN"] = str(bin_dir / "ffmpeg")
    env["VICUNA_WEBGL_RENDERER_CHROMIUM_BIN"] = str(bin_dir / "google-chrome-stable")
    return env


def test_install_dry_run_writes_system_scope_and_user_cleanup(tmp_path: Path) -> None:
    env = _base_env(_fake_bin(tmp_path))
    env["TELEGRAM_BRIDGE_STATE_PATH"] = "/tmp/stale-telegram-state.json"
    env["VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH"] = "/home/tester/.config/vicuna/openclaw-tool-secrets.json"
    env["VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH"] = "/home/tester/.local/state/vicuna/openclaw-catalog.json"

    result = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "VICUNA_SYSTEMD_SCOPE=system" in result.stdout
    assert "VICUNA_TELEGRAM_BRIDGE_SERVICE_NAME=vicuna-telegram-bridge.service" in result.stdout
    assert "TELEGRAM_BRIDGE_STATE_PATH=/var/lib/vicuna/telegram-bridge-state.json" in result.stdout
    assert "VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH=/var/lib/vicuna/openclaw-tool-secrets.json" in result.stdout
    assert "VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH=/var/lib/vicuna/openclaw-catalog.json" in result.stdout
    assert "/home/tester/.config/systemd/user/vicuna-runtime.service" in result.stdout


def test_install_dry_run_discovers_user_units_from_home_scan(tmp_path: Path) -> None:
    env = _base_env(_fake_bin(tmp_path))
    env.pop("VICUNA_INTERACTIVE_OWNER", None)
    env.pop("VICUNA_INTERACTIVE_HOME", None)

    home_root = tmp_path / "homes"
    unit_dir = home_root / "tester" / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    (unit_dir / "vicuna-runtime.service").write_text("[Unit]\nDescription=runtime\n")
    env["VICUNA_INTERACTIVE_HOME_ROOTS"] = str(home_root)

    result = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert str(unit_dir / "vicuna-runtime.service") in result.stdout


def test_install_dry_run_preserves_existing_env_values_when_shell_env_is_blank(tmp_path: Path) -> None:
    bin_dir = _fake_bin(tmp_path)
    env = _base_env(bin_dir)
    env.pop("TELEGRAM_BRIDGE_NODE_BIN", None)

    preserved_node = tmp_path / "preserved-node"
    _write_executable(
        preserved_node,
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            if [[ "${1:-}" == "-v" ]]; then
              echo "v20.19.5"
              exit 0
            fi
            exit 0
            """
        ),
    )

    env_file = tmp_path / "vicuna.env"
    env_file.write_text(
        textwrap.dedent(
            f"""\
            TELEGRAM_BRIDGE_NODE_BIN={preserved_node}
            VICUNA_DEEPSEEK_API_KEY=deepseek-preserved
            TELEGRAM_BOT_TOKEN=telegram-preserved
            TAVILY_API_KEY=tavily-preserved
            """
        )
    )
    env["VICUNA_SYSTEM_ENV_FILE"] = str(env_file)

    result = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"TELEGRAM_BRIDGE_NODE_BIN={preserved_node}" in result.stdout
    assert "VICUNA_DEEPSEEK_API_KEY=deepseek-preserved" in result.stdout
    assert "TELEGRAM_BOT_TOKEN=telegram-preserved" in result.stdout
    assert "TAVILY_API_KEY=tavily-preserved" in result.stdout
    assert "SUPERMEMORY_API_KEY" not in result.stdout


def test_rebuild_refuses_ambiguous_scope_when_both_unit_scopes_exist(tmp_path: Path) -> None:
    env = _base_env(_fake_bin(tmp_path))
    env["FAKE_SYSTEM_UNIT_PRESENT"] = "1"
    env["FAKE_USER_UNIT_PRESENT"] = "1"

    result = subprocess.run(
        ["bash", str(REBUILD_SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "both system and user Vicuña runtime units are installed" in result.stderr


def test_rebuild_uses_configured_system_scope_when_repo_root_matches(tmp_path: Path) -> None:
    env = _base_env(_fake_bin(tmp_path))
    env["FAKE_SYSTEM_UNIT_PRESENT"] = "1"
    env["FAKE_USER_UNIT_PRESENT"] = "0"
    env["FAKE_SYSTEM_RUNTIME_ROOT"] = str(REPO_ROOT)

    env_file = tmp_path / "vicuna.env"
    env_file.write_text(
        textwrap.dedent(
            f"""\
            VICUNA_SYSTEMD_SCOPE=system
            VICUNA_REPO_ROOT={REPO_ROOT}
            VICUNA_RUNTIME_SERVICE_NAME=vicuna-runtime.service
            VICUNA_TELEGRAM_BRIDGE_SERVICE_NAME=vicuna-telegram-bridge.service
            VICUNA_WEBGL_RENDERER_SERVICE_NAME=vicuna-webgl-renderer.service
            """
        )
    )
    env["VICUNA_SYSTEM_ENV_FILE"] = str(env_file)

    result = subprocess.run(
        ["bash", str(REBUILD_SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"repo={REPO_ROOT} scope=system" in result.stdout
    assert "dry-run: systemctl_cmd stop vicuna-runtime.service" in result.stdout
