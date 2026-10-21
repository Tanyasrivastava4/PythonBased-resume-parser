import { useEffect, useState } from 'react'
import { getHealth, uploadResume, pollJob } from './api'

export default function App() {
  const [health, setHealth] = useState(null)
  const [healthError, setHealthError] = useState(null)

  const [items, setItems] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((e) => setHealthError(e.message))
  }, [])

  function updateItem(id, patch) {
    setItems((prev) => prev.map((it) => (it.id === id ? { ...it, ...patch } : it)))
  }

  async function processOne(item) {
    updateItem(item.id, { status: 'processing' })
    try {
      let res = await uploadResume(item.file)
      if (res.status === 'processing') {
        res = await pollJob(res.job_id)
      }
      if (res.status === 'failed') {
        throw new Error(res.error || 'Parsing failed')
      }
      updateItem(item.id, { status: 'done', data: res.parsed_data ?? res.result })
    } catch (e) {
      updateItem(item.id, { status: 'failed', error: e.message })
    }
  }

  async function handleFileChange(event) {
    const files = Array.from(event.target.files)
    event.target.value = ''
    if (files.length === 0) return

    const newItems = files.map((file) => ({
      id: crypto.randomUUID(),
      file,
      name: file.name,
      status: 'queued',
      data: null,
      error: null,
    }))
    setItems((prev) => [...prev, ...newItems])

    const queue = [...newItems]
    async function worker() {
      while (queue.length > 0) {
        const next = queue.shift()
        await processOne(next)
      }
    }
    await Promise.all([worker(), worker()])
  }

  async function copyJson(item) {
    await navigator.clipboard.writeText(JSON.stringify(item.data, null, 2))
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  function downloadJson(item) {
    const text = JSON.stringify(item.data, null, 2)
    const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }))
    const a = document.createElement('a')
    a.href = url
    a.download = `${item.data.id}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const selected = items.find((it) => it.id === selectedId)

  return (
    <div style={{ padding: 24, fontFamily: 'sans-serif', textAlign: 'left' }}>
      <h1>ATS Resume Parser</h1>

      {!health && !healthError && <p>Checking server...</p>}
      {healthError && <p style={{ color: 'crimson' }}>Server not reachable: {healthError}</p>}
      {health && <p style={{ color: 'green' }}>Server {health.status}</p>}

      <hr />

      <h2>Upload resumes</h2>
      <input
        type="file"
        multiple
        accept=".pdf,.docx,.doc,.png,.jpg,.jpeg"
        onChange={handleFileChange}
      />

      <div style={{ display: 'flex', gap: 24, marginTop: 16 }}>
        <div style={{ width: 340 }}>
          {items.length === 0 && <p>No files yet.</p>}
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {items.map((it) => (
              <li key={it.id} style={{ marginBottom: 8 }}>
                <button
                  disabled={it.status !== 'done'}
                  onClick={() => setSelectedId(it.id)}
                  style={{ marginRight: 8 }}
                >
                  {it.name}
                </button>
                <span>{it.status}</span>
                {it.error && <span style={{ color: 'crimson' }}>: {it.error}</span>}
              </li>
            ))}
          </ul>
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          {selected && (
            <div>
              <div style={{ marginBottom: 8 }}>
                <button onClick={() => copyJson(selected)} style={{ marginRight: 8 }}>
                  {copied ? 'Copied!' : 'Copy JSON'}
                </button>
                <button onClick={() => downloadJson(selected)}>Download JSON</button>
              </div>
              <pre style={{ background: '#111', color: '#0f0', padding: 16, overflow: 'auto', maxHeight: '70vh' }}>
                {JSON.stringify(selected.data, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}