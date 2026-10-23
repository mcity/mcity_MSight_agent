"""Dataset ingestion: patch config.py for the engine's data_ingest workflow, run
main.py, stream its output, then restore config.py and select the new dataset.
Driven by ingest_server.py (the upload window), not by an MCP tool."""
import asyncio
import json
import logging
import re
import shutil
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Awaitable, Callable, List, Optional, Tuple

from msight_agent.agent.state import WorkflowState
from msight_agent.agent.workflow_tools import SELECTED_WORKFLOW_RE, with_selected_workflow, write_selected_dataset
from msight_agent.shared.paths import CONFIG_PATH, DATASET_UPLOADS_DIR, LOG_DIR, MAIN_PATH

UPLOAD_ROOT = DATASET_UPLOADS_DIR.resolve()
EmitFn = Callable[[dict], Awaitable[None]]


def _extract_zip(zip_path: str) -> tuple[Path, Path]:
    """(dataset folder, temp root to delete): the zip's single top-level folder if it has one."""
    tmp_root = Path(tempfile.mkdtemp(prefix="ingest_zip_"))
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_root)
    entries = list(tmp_root.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0], tmp_root
    return tmp_root, tmp_root


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def _write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def _patch_data_ingest_block(
    src: str,
    dataset_dir: str,
    fps: int,
    split_percentages: list[float],
    dataset_name: Optional[str],
    annotation_format: Optional[str],
) -> str:
    def upsert_key(block: str, key: str, value_src: str) -> str:
        patt = rf'(["\']{re.escape(key)}["\']\s*:\s*)(.*?)(,?\s*\n)'
        if re.search(patt, block, flags=re.DOTALL):
            def _repl(m: re.Match) -> str:
                return f"{m.group(1)}{value_src},\n"
            return re.sub(patt, _repl, block, flags=re.DOTALL)
        return re.sub(r'\}\s*$', f'    "{key}": {value_src},\n' + "}", block)

    fps_val = int(fps)
    splits_val = json.dumps([float(x) for x in split_percentages])

    if _data_ingest_block_span(src) is None:
        m = re.search(r'WORKFLOWS\s*=\s*\{', src)
        block_lines = [
            '    "data_ingest": {',
            f'        "dataset_dir": {repr(dataset_dir)},',
            f'        "fps": {fps_val},',
            f'        "split_percentages": {splits_val},',
        ]
        if dataset_name:
            block_lines.append(f'        "dataset_name": {repr(dataset_name)},')
        if annotation_format:
            block_lines.append(f'        "annotation_format": {repr(annotation_format)},')
        block_lines.append('    },\n')
        block = "\n".join(block_lines)
        if not m:
            inject = "\n".join(["", "WORKFLOWS = {", block, "}", ""])
            return src + inject
        insert_at = src.find("{", m.end())
        if insert_at == -1:
            insert_at = m.end()
        return src[:insert_at + 1] + "\n" + block + src[insert_at + 1:]

    span = _data_ingest_block_span(src)
    if span is None:
        return src
    start, end = span
    block_src = src[start:end]
    block_new = block_src
    block_new = upsert_key(block_new, "dataset_dir", repr(dataset_dir))
    block_new = upsert_key(block_new, "fps", str(fps_val))
    block_new = upsert_key(block_new, "split_percentages", splits_val)
    if dataset_name:
        block_new = upsert_key(block_new, "dataset_name", repr(dataset_name))
    if annotation_format:
        block_new = upsert_key(block_new, "annotation_format", repr(annotation_format))
    return src[:start] + block_new + src[end:]

def _parse_name_samples(text: str) -> Tuple[Optional[str], Optional[int]]:
    name, samples = None, None
    m = re.search(r'(Created dataset|Dataset(?:\s+name)?):?\s*([A-Za-z0-9_\-\.]+)', text, flags=re.IGNORECASE)
    if m: name = m.group(2)
    m2 = re.search(r'(?:samples\s*=\s*|)(\d+)\s+samples', text, flags=re.IGNORECASE)
    if m2:
        try: samples = int(m2.group(1))
        except ValueError: pass
    if samples is None:
        m3 = re.search(r'samples\s*=\s*(\d+)', text, flags=re.IGNORECASE)
        if m3:
            try: samples = int(m3.group(1))
            except ValueError: pass
    return name, samples

async def _stream_subprocess(proc: asyncio.subprocess.Process, on_line):
    async def _reader(stream, source):
        while True:
            line = await stream.readline()
            if not line:
                break
            await on_line(source, line.decode("utf-8", errors="ignore"))
    await asyncio.gather(_reader(proc.stdout, "stdout"), _reader(proc.stderr, "stderr"))

def _data_ingest_block_span(src: str) -> Optional[tuple[int, int]]:
    """(start, end) of the WORKFLOWS["data_ingest"] dict literal, braces matched."""
    start = src.find('"data_ingest": {')
    if start == -1:
        start = src.find("'data_ingest': {")
    if start == -1:
        return None
    depth, i = 0, src.index("{", start)
    for j in range(i, len(src)):
        depth += {"{": 1, "}": -1}.get(src[j], 0)
        if depth == 0:
            return start, j + 1
    return None


def _restore_ingest_patch(original: str) -> None:
    """Undo only what ingest patched (SELECTED_WORKFLOW and the data_ingest block).
    Restoring the whole file would also undo anything the chat changed meanwhile,
    e.g. a dataset selected while the upload was running."""
    current = _read(CONFIG_PATH)
    if m := SELECTED_WORKFLOW_RE.search(original):
        current = SELECTED_WORKFLOW_RE.sub(lambda _: m.group(0).strip(), current, count=1)
    orig_span, cur_span = _data_ingest_block_span(original), _data_ingest_block_span(current)
    if orig_span and cur_span:
        current = current[:cur_span[0]] + original[orig_span[0]:orig_span[1]] + current[cur_span[1]:]
    _write(CONFIG_PATH, current)


def _mark_ingested(dataset_name: str) -> None:
    """Make the new dataset the selected one, in both the chat's session state and
    SELECTED_DATASET (what main.py runs on) -- they must agree."""
    try:
        write_selected_dataset(dataset_name)
        state = WorkflowState.load()
        state.dataset_name = dataset_name
        state.dataset_confirmed = True
        state.save()
    except Exception as e:
        logging.warning(f"[INGEST] Failed to select ingested dataset '{dataset_name}': {e}")


async def run_data_ingest(
    emit: EmitFn,
    zip_path: Optional[str] = None,
    source_path: Optional[str] = None,
    fps: int = 2,
    split_percentages: Optional[List[float]] = None,
    dataset_prefix: Optional[str] = None,
    annotation_format: Optional[str] = None,
) -> None:
    """Ingest a zip or folder, sending {"type", "data"} events to `emit`."""
    if not zip_path and not source_path:
        await emit({"type": "error", "data": {"msg": "Provide zip_path or source_path"}})
        return

    tmp_root = None
    backup_path = CONFIG_PATH.with_suffix(".py.bak")
    combined_lines: List[str] = []
    ds_name, samples, final_rc = None, None, None

    try:
        dataset_dir = source_path
        if zip_path:
            # Extracted under /tmp; move into the persistent upload folder.
            extracted, tmp_root = _extract_zip(zip_path)
            persistent_dir = UPLOAD_ROOT / f"job_{uuid.uuid4().hex}"
            persistent_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extracted), str(persistent_dir))
            dataset_dir = str(persistent_dir)
        await emit({"type": "log", "data": {"msg": f"Prepared dataset_dir: {dataset_dir}"}})

        original = _read(CONFIG_PATH)
        _write(backup_path, original)
        patched = with_selected_workflow(original, "data_ingest")
        patched = _patch_data_ingest_block(
            patched, dataset_dir, int(fps), split_percentages or [0.7, 0.15, 0.15],
            dataset_prefix, annotation_format
        )
        _write(CONFIG_PATH, patched)
        await emit({"type": "log", "data": {"msg": "Config patched for data_ingest"}})

        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", str(MAIN_PATH),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(MAIN_PATH.parent)
        )

        async def on_line(source: str, line: str):
            nonlocal ds_name, samples
            combined_lines.append(f"[{source}] {line}")
            text = line.strip()
            # Target the specific ingestion line directly
            ingest_match = re.search(r"Ingesting dataset:\s*([A-Za-z0-9_\-\.]+)", text)
            if ingest_match:
                ds_name = ingest_match.group(1)

            _, s = _parse_name_samples(text)
            if s: samples = s
            await emit({"type": "log", "data": {"msg": text}})

        await _stream_subprocess(proc, on_line)
        final_rc = await proc.wait()

        log_path = LOG_DIR / "last_data_ingest_log.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _write(log_path, "".join(combined_lines))

        try:
            _restore_ingest_patch(original)
            backup_path.unlink(missing_ok=True)
        except Exception as restore_err:
            logging.warning(f"[INGEST] Failed to restore config.py: {restore_err}")

        if not ds_name:
            ds_name = dataset_prefix or "custom_dataset"

        if final_rc == 0:
            # Ingestion runs outside chat, so select the dataset here.
            _mark_ingested(ds_name)

            await emit({
                "type": "agent",
                "data": {
                    "msg": (
                        f"✅ Ingestion complete. Dataset **{ds_name}**"
                    ),
                    "dataset": ds_name,
                    "samples": samples,
                },
            })
            await emit({"type": "done", "data": {"ok": True}})
        else:
            await emit({"type": "error", "data": {"msg": f"data_ingest failed with exit code {final_rc}"}})

    except Exception as e:
        logging.exception("run_data_ingest failed")
        await emit({"type": "error", "data": {"msg": str(e)}})
        try:
            if backup_path.exists():
                _restore_ingest_patch(_read(backup_path))
                backup_path.unlink(missing_ok=True)
        except Exception as restore_err:
            logging.warning(f"[INGEST] Failed to restore config.py: {restore_err}")
    finally:
        if tmp_root is not None:
            shutil.rmtree(tmp_root, ignore_errors=True)
