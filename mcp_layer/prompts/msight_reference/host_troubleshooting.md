# Investigating host-level problems

Use when the pipeline graph looks fine (or every node fails the same way) and
the cause is likely the machine itself. All commands go through `inspect_host`,
one per call, read-only.

## Method

1. State a hypothesis from what you already know (diagnosis, status, logs).
2. Run the commands that would confirm **or rule it out**.
3. Repeat until the evidence points at one cause. Quote the output.
4. If you can't pin it down, say what you checked and what's still unknown.
   Don't fall back to a generic checklist.

## Which commands answer what

| Question | Commands |
|---|---|
| What kernel / OS is running, since when? | `uname -r`, `uptime` |
| Is a kernel module loaded / does it exist for this kernel? | `lsmod`, `modinfo <module>`, `dpkg -l <package-pattern>` |
| Why did a driver/module fail to load? | `journalctl -k -n 200` |
| Is Secure Boot on? | `mokutil --sb-state` |
| What hardware is present, which driver claims it? | `lspci -k` |
| Is the GPU usable? | `nvidia-smi`, `nvidia-smi -L` |
| Disk full? | `df -h`, `df -h /var/lib/docker` |
| Memory pressure? | `free -h`, `docker stats --no-stream` |
| Is a port already taken, and by what? | `ss -ltnp` |
| Is Docker healthy / which runtimes does it have? | `systemctl status docker`, `docker info` |
| Did a container crash, and why? | `docker ps -a`, `docker logs --tail 100 <container>`, `docker inspect <container>` |
| Is Redis (the message bus) up? | `docker ps -a` (look for `msight-redis`), `docker logs --tail 50 msight-redis`, `ss -ltnp` (port 6379) |

## Advising fixes

You can't run sudo, reboot or install anything. Give the user the exact command
and the output that justifies it, and prefer the narrowest fix the evidence
supports. Never suggest purging/removing packages, deleting data, or disabling
security features.
