"""inspect_host allowlist: read-only commands pass, anything that could change
state, chain commands or read files is refused."""
import asyncio

import pytest

from mcp_layer.mcptools.msight import host_inspect
from mcp_layer.mcptools.msight.host_inspect import redact, run_inspection, validate


@pytest.mark.parametrize("command", [
    "uname -r",
    "lsmod",
    "lspci -k",
    "dpkg -l nvidia*",
    "mokutil --sb-state",
    "modinfo -F version nvidia",
    "nvidia-smi",
    "nvidia-smi --query-gpu=name,driver_version --format=csv",
    "df -h /var/lib/docker",
    "ss -ltnp",
    "docker ps -a",
    "docker logs --tail 50 msight-rfdetr_detector",
    "docker inspect -f '{{.State.Running}}' msight-redis",
    "docker stats --no-stream",
    "systemctl status docker",
    "journalctl -u docker -n 50",
    "journalctl -k -n 100",
])
def test_read_only_commands_are_allowed(command):
    argv, reason = validate(command)
    assert reason is None and argv


@pytest.mark.parametrize("command", [
    "cat /home/dataengine/Github/mcity_MSight_agent/.env",   # file read
    "sudo lsmod",
    "uname -r | grep 34",                                     # pipe
    "uname -r; reboot",                                       # chaining
    "lsmod > /tmp/x",                                         # redirect
    "echo $HOME",
    "apt-get purge nvidia*",
    "docker exec msight-redis redis-cli flushall",
    "docker rm -f msight-redis",
    "docker run --rm alpine",
    "docker logs -f msight-redis",                            # follow never returns
    "docker stats",                                           # streams without --no-stream
    "nvidia-smi -r",                                          # GPU reset
    "nvidia-smi -pm 1",
    "systemctl restart docker",
    "journalctl",                                             # whole journal
    "dpkg -i evil.deb",
    "df -h ../..",
    "",
])
def test_unsafe_commands_are_refused(command):
    argv, reason = validate(command)
    assert argv is None and reason


def test_pager_is_disabled_for_systemd_tools():
    argv, _ = validate("systemctl status docker")
    assert "--no-pager" in argv


def test_redaction():
    text = "OPENAI_API_KEY=sk-abc123\nAWS_SECRET_ACCESS_KEY=xyz\nkey AKIAABCDEFGHIJKLMNOP\nPATH=/usr/bin"
    out = redact(text)
    assert "sk-abc123" not in out and "xyz" not in out and "AKIAABCDEFGHIJKLMNOP" not in out
    assert "OPENAI_API_KEY=<redacted>" in out and "PATH=/usr/bin" in out


def test_runs_and_reports_exit_code():
    r = asyncio.run(run_inspection("uname -s"))
    assert r["allowed"] and r["exit_code"] == 0 and r["output"].strip() == "Linux"


def test_output_is_capped(monkeypatch):
    monkeypatch.setattr(host_inspect, "_MAX_OUTPUT_CHARS", 10)
    r = asyncio.run(run_inspection("uname -a"))
    assert r["truncated"] and len(r["output"]) == 10


def test_timeout(monkeypatch):
    monkeypatch.setattr(host_inspect, "_TIMEOUT_SECONDS", 0.001)
    r = asyncio.run(run_inspection("lspci -v"))
    assert r["exit_code"] is None and "timed out" in r["output"]
