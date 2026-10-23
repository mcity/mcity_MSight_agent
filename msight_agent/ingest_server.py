"""Dataset upload backend (port 8002) for the chat UI's ingestion window: accepts a
zip, runs ingestion in the background, and streams its events over SSE."""
import asyncio
import json
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from msight_agent.auto_labeling.tools.data_ingest import run_data_ingest

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Holds queues for jobs: job_id -> asyncio.Queue
JOB_LOGS: dict[str, asyncio.Queue[str | None]] = {}


async def run_ingest_and_stream(job_id, zip_path, fps, train, val, test):
    """Run ingestion, pushing events to this job's JOB_LOGS queue."""
    q = JOB_LOGS[job_id]

    async def emit(ev: dict):
        # Each event is JSON-encoded for SSE
        await q.put(json.dumps(ev))

    try:
        await run_data_ingest(
            emit,
            zip_path=zip_path,
            fps=fps,
            split_percentages=[train, val, test],
            dataset_prefix="custom_dataset",
            annotation_format="auto",
        )
    except Exception as e:
        await q.put(json.dumps({"type": "error", "data": {"msg": str(e)}}))
    finally:
        Path(zip_path).unlink(missing_ok=True)  # contents were extracted to the upload folder
        await q.put(None)  # end of stream


@app.post("/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    fps: int = Form(...),
    train: float = Form(...),
    val: float = Form(...),
    test: float = Form(...)
):
    """Start background ingestion of an uploaded zip; returns job_id."""
    # Basename only: the client controls file.filename.
    tmp_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}_{Path(file.filename or 'upload.zip').name}"
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    job_id = str(uuid.uuid4())
    JOB_LOGS[job_id] = asyncio.Queue()

    asyncio.create_task(run_ingest_and_stream(job_id, str(tmp_path), fps, train, val, test))

    return {"status": "started", "job_id": job_id}


@app.get("/ingest/stream/{job_id}")
async def ingest_stream(job_id: str):
    """Stream ingestion events as SSE."""
    q = JOB_LOGS.get(job_id)
    if q is None:
        return StreamingResponse(iter(["event: done\ndata: {}\n\n"]), media_type="text/event-stream")

    async def event_generator():
        # Optional SSE retry instruction for client reconnects
        yield "retry: 1000\n\n"
        had_error = False
        while True:
            msg = await q.get()
            if msg is None:
                # The stream always ends here, even after a failure; ok carries the outcome.
                payload = json.dumps({"type": "done", "data": {"ok": not had_error}})
                yield f"event: done\ndata: {payload}\n\n"
                JOB_LOGS.pop(job_id, None)  # job finished; a reconnect gets the default "done"
                break

            # Forward each event under its own type (hardcoding "log" hid ingestion errors).
            try:
                parsed = json.loads(msg)
            except ValueError:
                parsed = None
            ev_type = parsed.get("type", "log") if isinstance(parsed, dict) else "log"

            if ev_type == "error":
                had_error = True
                # EventSource reports its own transport failures as "error"; keep them apart.
                ev_type = "ingest_error"

            yield f"event: {ev_type}\ndata: {msg}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
