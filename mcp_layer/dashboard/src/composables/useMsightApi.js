// Thin wrapper over msight_api.py's REST surface (port 8003) -- same
// same-hostname-different-port convention the existing chat UI already uses
// for its own backends (mcp_layer/ui/index.html's CHAT_BASE/INGEST_BASE).
const BASE = `http://${window.location.hostname}:8003`

async function request(path, options = {}) {
  let res
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options
    })
  } catch {
    return { status: 'error', message: `Could not reach msight_api at ${BASE} -- is it running?` }
  }
  try {
    return await res.json()
  } catch {
    return { status: 'error', message: `Non-JSON response from ${path} (HTTP ${res.status})` }
  }
}

const get = (path) => request(path)
const post = (path, body) => request(path, { method: 'POST', body: JSON.stringify(body ?? {}) })

export function useMsightApi() {
  return {
    // status / diagnostics
    getStatus: () => get('/status'),
    getLogs: (service, tail = 200) => {
      const params = new URLSearchParams({ tail: String(tail) })
      if (service) params.set('service', service)
      return get(`/logs?${params}`)
    },
    getCalibration: () => get('/calibration'),

    // pipeline
    startPipeline: (videoInput, rtspUrl, sensorName, build = false) =>
      post('/pipeline/start', { video_input: videoInput || null, rtsp_url: rtspUrl || null, sensor_name: sensorName || null, build }),
    stopPipeline: (removeVolumes = false) => post('/pipeline/stop', { remove_volumes: removeVolumes }),

    // recording / archiving
    startRecording: (sensorName) => post('/recording/start', { sensor_name: sensorName || null }),
    stopRecording: () => post('/recording/stop'),
    startArchiving: (s3Bucket, s3Prefix) => post('/archiving/start', { s3_bucket: s3Bucket, s3_prefix: s3Prefix || null }),
    stopArchiving: () => post('/archiving/stop'),
    getRecordArchiveStatus: () => get('/record_archive/status'),

    // generic nodes
    listNodeTypes: () => get('/nodes/types'),
    addNode: (nodeType, name, config) => post('/nodes/add', { node_type: nodeType, name: name || null, config: config || null }),
    removeNode: (name) => post('/nodes/remove', { name }),

    // reference
    getReference: (topic) => get(`/reference/${topic}`)
  }
}
