<template>
  <q-page class="q-pa-md column q-gutter-md">
    <!-- Pipeline control -->
    <q-card>
      <q-card-section>
        <div class="text-h6">Pipeline</div>
        <div class="text-caption text-grey-7">
          Calibration:
          <span :class="calibration.state === 'user_calibrated' ? 'text-positive' : 'text-warning'">
            {{ calibration.state || 'unknown' }}
          </span>
          <span v-if="calibration.message"> — {{ calibration.message }}</span>
        </div>
      </q-card-section>
      <q-card-section class="row q-gutter-sm">
        <q-input dense outlined v-model="pipeline.videoInput" label="video_input (file/folder)" class="col" />
        <q-input dense outlined v-model="pipeline.rtspUrl" label="rtsp_url" class="col" />
        <q-input dense outlined v-model="pipeline.sensorName" label="sensor_name" class="col" />
      </q-card-section>
      <q-card-actions>
        <q-btn color="primary" label="Start pipeline" @click="onStartPipeline" :loading="busy.pipeline" />
        <q-btn color="negative" outline label="Stop pipeline" @click="onStopPipeline" :loading="busy.pipeline" />
      </q-card-actions>
    </q-card>

    <!-- Nodes -->
    <q-card>
      <q-card-section class="row items-center">
        <div class="text-h6 col">Nodes</div>
        <q-btn dense flat icon="refresh" @click="refreshStatus" />
      </q-card-section>
      <q-table
        :rows="nodes"
        :columns="nodeColumns"
        row-key="name"
        dense
        flat
        :pagination="{ rowsPerPage: 0 }"
      >
        <template v-slot:body-cell-status="props">
          <q-td :props="props">
            <q-badge :color="statusColor(props.value)">{{ props.value }}</q-badge>
          </q-td>
        </template>
        <template v-slot:body-cell-actions="props">
          <q-td :props="props">
            <q-btn dense flat icon="description" @click="openLogs(props.row.name)" title="View logs" />
            <q-btn dense flat icon="delete" color="negative" @click="onRemoveNode(props.row.name)" title="Remove node" />
          </q-td>
        </template>
      </q-table>
    </q-card>

    <!-- Add node -->
    <q-card>
      <q-card-section>
        <div class="text-h6">Add node</div>
      </q-card-section>
      <q-card-section class="row q-gutter-sm">
        <q-select
          dense outlined class="col-3"
          v-model="addForm.nodeType"
          :options="nodeTypeOptions"
          emit-value map-options
          label="node_type"
          @update:model-value="onNodeTypeChange"
        />
        <q-input dense outlined class="col-2" v-model="addForm.name" label="name (optional)" />
        <q-input
          v-for="key in addForm.configKeys"
          :key="key"
          dense outlined class="col-2"
          v-model="addForm.config[key]"
          :label="key"
        />
      </q-card-section>
      <q-card-section v-if="selectedNodeTypeInfo" class="text-caption text-grey-7">
        {{ selectedNodeTypeInfo.description }}
        <span v-if="selectedNodeTypeInfo.needs_gpu"> (uses GPU if available, falls back to CPU otherwise)</span>
      </q-card-section>
      <q-card-actions>
        <q-btn color="primary" label="Add node" @click="onAddNode" :disable="!addForm.nodeType" :loading="busy.addNode" />
      </q-card-actions>
    </q-card>

    <!-- Record & Archive -->
    <q-card>
      <q-card-section>
        <div class="text-h6">Record &amp; Archive</div>
        <div class="text-caption text-grey-7">
          recording: {{ recordArchive.local_dumper ? 'active' : 'inactive' }},
          archiving: {{ recordArchive.s3_pusher ? 'active' : 'inactive' }}
        </div>
      </q-card-section>
      <q-card-section class="row q-gutter-sm items-center">
        <q-input dense outlined v-model="recording.sensorName" label="sensor_name (optional)" class="col-3" />
        <q-btn color="primary" label="Start recording" @click="onStartRecording" :loading="busy.recording" />
        <q-btn color="negative" outline label="Stop recording" @click="onStopRecording" :loading="busy.recording" />
      </q-card-section>
      <q-card-section class="row q-gutter-sm items-center">
        <q-input dense outlined v-model="archiving.bucket" label="s3_bucket" class="col-3" />
        <q-input dense outlined v-model="archiving.prefix" label="s3_prefix (optional)" class="col-3" />
        <q-btn color="primary" label="Start archiving" @click="onStartArchiving" :loading="busy.archiving" />
        <q-btn color="negative" outline label="Stop archiving" @click="onStopArchiving" :loading="busy.archiving" />
      </q-card-section>
    </q-card>

    <!-- Reference / Troubleshooting -->
    <q-card>
      <q-expansion-item icon="help_outline" label="Reference / Troubleshooting">
        <q-card-section class="row items-center q-gutter-sm">
          <q-select
            dense outlined class="col-3"
            v-model="referenceTopic"
            :options="['node_catalog', 'common_failure_modes', 'diagnosing_stalled_nodes']"
            @update:model-value="onReferenceChange"
          />
        </q-card-section>
        <q-card-section>
          <pre class="reference-text">{{ referenceText }}</pre>
        </q-card-section>
      </q-expansion-item>
    </q-card>

    <!-- Logs dialog -->
    <q-dialog v-model="logsDialog.open">
      <q-card style="min-width: 700px; max-width: 90vw">
        <q-card-section class="row items-center">
          <div class="text-h6 col">Logs — {{ logsDialog.service }}</div>
          <q-btn dense flat icon="refresh" @click="refreshLogs" />
        </q-card-section>
        <q-card-section>
          <pre ref="logsPre" class="reference-text" style="max-height: 60vh; overflow-y: auto">{{ logsDialog.text }}</pre>
          <div class="text-caption text-grey-7 q-mt-sm">
            <span v-if="logsDialog.freshness !== null">last log line: {{ logsDialog.freshness }}s ago — </span>
            live, updates every 3s
          </div>
        </q-card-section>
      </q-card>
    </q-dialog>
  </q-page>
</template>

<script setup>
import { ref, reactive, computed, watch, nextTick, onMounted, onUnmounted } from 'vue'
import { useQuasar } from 'quasar'
import { useMsightApi } from '@/composables/useMsightApi'

const $q = useQuasar()
const api = useMsightApi()

const nodes = ref([])
const nodeTypes = ref([])
const calibration = ref({})
const recordArchive = ref({})

const pipeline = reactive({ videoInput: '', rtspUrl: '', sensorName: '' })
const recording = reactive({ sensorName: '' })
const archiving = reactive({ bucket: '', prefix: '' })
const addForm = reactive({ nodeType: null, name: '', config: {}, configKeys: [] })
const referenceTopic = ref('node_catalog')
const referenceText = ref('')
const busy = reactive({ pipeline: false, recording: false, archiving: false, addNode: false })

const logsDialog = reactive({ open: false, service: '', text: '', freshness: null })
const logsPre = ref(null)
let logsPollTimer = null

const nodeColumns = [
  { name: 'name', label: 'name', field: 'name', align: 'left' },
  { name: 'type', label: 'type', field: 'type', align: 'left' },
  { name: 'status', label: 'status', field: 'status', align: 'left' },
  { name: 'heartbeat', label: 'heartbeat age', field: (r) => r.seconds_since_heartbeat ?? '-', align: 'left' },
  { name: 'publish_topic', label: 'publishes', field: 'publish_topic', align: 'left' },
  { name: 'subscribe_topic', label: 'subscribes', field: 'subscribe_topic', align: 'left' },
  { name: 'actions', label: '', field: 'name', align: 'right' }
]

const nodeTypeOptions = computed(() =>
  nodeTypes.value.map((t) => ({ label: t.node_type, value: t.node_type }))
)
const selectedNodeTypeInfo = computed(() =>
  nodeTypes.value.find((t) => t.node_type === addForm.nodeType)
)

function statusColor(status) {
  if (status === 'RUNNING') return 'positive'
  if (status === 'REGISTERED') return 'warning'
  return 'negative'
}

function notify(result) {
  $q.notify({
    type: result?.status === 'ok' ? 'positive' : 'negative',
    message: result?.message || JSON.stringify(result)
  })
  return result
}

// Guards against a stale poll response (the 4s interval) resolving after a
// fresher one (e.g. the refresh right after add/remove) and clobbering it
// back to the old list -- confirmed happening live: removing a node updated
// the backend correctly but a late in-flight poll re-populated the row.
let statusRequestId = 0
async function refreshStatus() {
  const requestId = ++statusRequestId
  const r = await api.getStatus()
  if (requestId !== statusRequestId) return
  if (r.status === 'ok') nodes.value = r.services
}

async function refreshCalibration() {
  calibration.value = await api.getCalibration()
}

async function refreshRecordArchive() {
  const r = await api.getRecordArchiveStatus()
  recordArchive.value = r
}

async function onStartPipeline() {
  busy.pipeline = true
  notify(await api.startPipeline(pipeline.videoInput, pipeline.rtspUrl, pipeline.sensorName))
  await refreshStatus()
  await refreshCalibration()
  busy.pipeline = false
}

async function onStopPipeline() {
  busy.pipeline = true
  notify(await api.stopPipeline())
  await refreshStatus()
  busy.pipeline = false
}

async function onStartRecording() {
  busy.recording = true
  notify(await api.startRecording(recording.sensorName))
  await refreshRecordArchive()
  busy.recording = false
}

async function onStopRecording() {
  busy.recording = true
  notify(await api.stopRecording())
  await refreshRecordArchive()
  busy.recording = false
}

async function onStartArchiving() {
  busy.archiving = true
  notify(await api.startArchiving(archiving.bucket, archiving.prefix))
  await refreshRecordArchive()
  busy.archiving = false
}

async function onStopArchiving() {
  busy.archiving = true
  notify(await api.stopArchiving())
  await refreshRecordArchive()
  busy.archiving = false
}

// publish_topic/subscribe_topic/sensor_name come from category, not
// required_config -- mirrors msight_node_catalog.py's resolve_cmd exactly,
// since /nodes/types doesn't expose this derivation itself.
function categoryFields(category) {
  if (category === 'source') return ['publish_topic', 'sensor_name']
  if (category === 'processing') return ['publish_topic', 'subscribe_topic']
  if (category === 'sink') return ['subscribe_topic']
  return []
}

function onNodeTypeChange() {
  const info = selectedNodeTypeInfo.value
  const keys = info ? [...categoryFields(info.category), ...info.required_config] : []
  addForm.configKeys = [...new Set(keys)]
  addForm.config = Object.fromEntries(addForm.configKeys.map((k) => [k, '']))
}

async function onAddNode() {
  busy.addNode = true
  notify(await api.addNode(addForm.nodeType, addForm.name, addForm.config))
  await refreshStatus()
  busy.addNode = false
}

async function onRemoveNode(name) {
  notify(await api.removeNode(name))
  await refreshStatus()
}

async function openLogs(service) {
  logsDialog.service = service
  logsDialog.open = true
  await refreshLogs()
}

async function refreshLogs() {
  const r = await api.getLogs(logsDialog.service, 200)
  logsDialog.text = r.status === 'ok' ? r.logs : (r.message || 'error fetching logs')
  logsDialog.freshness = r.seconds_since_last_line ?? null
  await nextTick()
  if (logsPre.value) logsPre.value.scrollTop = logsPre.value.scrollHeight
}

// Polling, not a real tail -f -- reuses the same snapshot endpoint the
// dialog already calls on open/refresh, just on an interval while it's
// open. Watches logsDialog.open (not just openLogs()) so this also stops
// correctly when the dialog is dismissed via backdrop click or ESC.
watch(() => logsDialog.open, (open) => {
  if (open) {
    logsPollTimer = setInterval(refreshLogs, 3000)
  } else if (logsPollTimer) {
    clearInterval(logsPollTimer)
    logsPollTimer = null
  }
})

async function onReferenceChange() {
  const r = await api.getReference(referenceTopic.value)
  referenceText.value = r.status === 'ok' ? r.text : (r.message || 'error fetching reference')
}

let pollTimer = null

onMounted(async () => {
  const r = await api.listNodeTypes()
  if (r.status === 'ok') nodeTypes.value = r.node_types
  await refreshStatus()
  await refreshCalibration()
  await refreshRecordArchive()
  await onReferenceChange()
  pollTimer = setInterval(refreshStatus, 4000)
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
  if (logsPollTimer) clearInterval(logsPollTimer)
})
</script>

<style scoped>
.reference-text {
  white-space: pre-wrap;
  word-break: break-word;
  font-family: monospace;
  font-size: 12px;
}
</style>
