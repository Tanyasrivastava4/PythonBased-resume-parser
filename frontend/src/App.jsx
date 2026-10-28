import { useEffect, useState } from 'react'
import { getHealth, uploadBatch, pollBatch } from './api'

export default function App() {
  const [health, setHealth] = useState(null)
  const [healthError, setHealthError] = useState(null)

  const [items, setItems] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [viewMode, setViewMode] = useState('selected') // 'selected' | 'all'
  const [copied, setCopied] = useState(false)
  const [searchFilter, setSearchFilter] = useState('')

  const [batchInfo, setBatchInfo] = useState({
    active: false,
    batchId: null,
    total: 0,
    completed: 0,
    failed: 0,
    percentage: 0,
    statusText: '',
  })

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((e) => setHealthError(e.message))
  }, [])

  async function handleFileChange(event) {
    const files = Array.from(event.target.files)
    event.target.value = ''
    if (files.length === 0) return

    setItems([])
    setSelectedId(null)
    setViewMode('selected')
    setBatchInfo({
      active: true,
      batchId: null,
      total: files.length,
      completed: 0,
      failed: 0,
      percentage: 0,
      statusText: `Uploading ${files.length} resume(s) in a single batch...`,
    })

    try {
      // 1. Single API call sending all 100-200 files!
      const initRes = await uploadBatch(files)
      const batchId = initRes.batch_id

      setBatchInfo((prev) => ({
        ...prev,
        batchId,
        statusText: `Processing ${files.length} resumes in background with 4 workers...`,
      }))

      // 2. Poll live batch progress and receive candidates as they finish
      await pollBatch(batchId, (update) => {
        setBatchInfo({
          active: update.status === 'processing',
          batchId: update.batch_id,
          total: update.total,
          completed: update.completed,
          failed: update.failed,
          percentage: update.percentage,
          statusText:
            update.status === 'completed'
              ? `All ${update.total} resumes processed successfully!`
              : `Processing ${update.completed}/${update.total} resumes...`,
        })

        if (update.candidates && update.candidates.length > 0) {
          const mapped = update.candidates.map((c) => ({
            id: c.id || crypto.randomUUID(),
            name: c.name && c.name !== 'Failed to parse' ? c.name : (c.source_file || 'Unknown candidate'),
            filename: c.source_file,
            status: c.batch_status || 'done',
            data: c,
            error: c.error || null,
          }))

          setItems(mapped)
          setSelectedId((curr) => curr || mapped[0]?.id)
        }
      })
    } catch (err) {
      setBatchInfo((prev) => ({
        ...prev,
        active: false,
        statusText: `Batch error: ${err.message}`,
      }))
    }
  }

  async function copyJson(dataToCopy) {
    await navigator.clipboard.writeText(JSON.stringify(dataToCopy, null, 2))
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  function downloadJson(dataToDownload, filename) {
    const text = JSON.stringify(dataToDownload, null, 2)
    const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }))
    const a = document.createElement('a')
    a.href = url
    a.download = filename || `resume_${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  function downloadAllBatchJson() {
    if (items.length === 0) return
    const allData = items.map((it) => it.data)
    downloadJson(allData, `all_${items.length}_parsed_resumes_${Date.now()}.json`)
  }

  const filteredItems = items.filter((it) => {
    if (!searchFilter.trim()) return true
    const q = searchFilter.toLowerCase()
    return (
      it.name.toLowerCase().includes(q) ||
      (it.filename && it.filename.toLowerCase().includes(q))
    )
  })

  const selected = items.find((it) => it.id === selectedId)
  const allCandidatesData = items.map((it) => it.data)

  return (
    <div style={{ padding: 24, fontFamily: 'sans-serif', textAlign: 'left', maxWidth: 1440, margin: '0 auto' }}>
      <h1>ATS Resume Parser (Production Bulk Engine)</h1>

      {!health && !healthError && <p>Checking server...</p>}
      {healthError && <p style={{ color: 'crimson' }}>Server not reachable: {healthError}</p>}
      {health && (
        <p style={{ color: 'green', fontWeight: 'bold' }}>
          ● Server operational ({health.ocr_engine} engine) | Active batches: {health.active_batches ?? 0}
        </p>
      )}

      <hr style={{ margin: '16px 0', borderColor: '#e2e8f0' }} />

      <div style={{ marginBottom: 20 }}>
        <h2 style={{ marginBottom: 8 }}>Upload Resumes (Single or 100+ Bulk Batch)</h2>
        <input
          type="file"
          multiple
          accept=".pdf,.docx,.doc,.png,.jpg,.jpeg"
          onChange={handleFileChange}
          style={{ padding: '8px 12px', border: '1px solid #cbd5e1', borderRadius: 6, cursor: 'pointer' }}
        />
      </div>

      {/* Bulk Progress Bar */}
      {batchInfo.total > 0 && (
        <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 16, marginBottom: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, fontWeight: 500 }}>
            <span>{batchInfo.statusText}</span>
            <span>
              {batchInfo.completed} / {batchInfo.total} Completed ({batchInfo.percentage}%)
              {batchInfo.failed > 0 && <span style={{ color: 'crimson', marginLeft: 8 }}>({batchInfo.failed} failed)</span>}
            </span>
          </div>
          <div style={{ height: 10, background: '#e2e8f0', borderRadius: 5, overflow: 'hidden' }}>
            <div
              style={{
                height: '100%',
                width: `${batchInfo.percentage}%`,
                background: batchInfo.percentage === 100 ? '#10b981' : '#3b82f6',
                transition: 'width 0.4s ease',
              }}
            />
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <div style={{ display: 'flex', gap: 24, marginTop: 16 }}>
        {/* Left Side: Candidates List */}
        <div style={{ width: 380, display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <h3 style={{ margin: 0 }}>Candidates ({items.length})</h3>
            {items.length > 1 && (
              <button
                onClick={downloadAllBatchJson}
                style={{
                  padding: '4px 8px',
                  fontSize: 12,
                  background: '#0284c7',
                  color: '#fff',
                  border: 'none',
                  borderRadius: 4,
                  cursor: 'pointer',
                  fontWeight: 600,
                }}
              >
                Download All JSON
              </button>
            )}
          </div>

          {items.length > 5 && (
            <input
              type="text"
              placeholder="Search by candidate name or file..."
              value={searchFilter}
              onChange={(e) => setSearchFilter(e.target.value)}
              style={{
                padding: '6px 10px',
                marginBottom: 10,
                border: '1px solid #cbd5e1',
                borderRadius: 4,
                fontSize: 13,
              }}
            />
          )}

          {items.length === 0 && !batchInfo.active && <p style={{ color: '#64748b' }}>No files uploaded yet.</p>}

          <ul style={{ listStyle: 'none', padding: 0, margin: 0, maxHeight: '72vh', overflowY: 'auto' }}>
            {filteredItems.map((it) => {
              const isSelected = it.id === selectedId && viewMode === 'selected'
              return (
                <li
                  key={it.id}
                  style={{
                    padding: '8px 10px',
                    marginBottom: 6,
                    borderRadius: 6,
                    border: isSelected ? '1px solid #3b82f6' : '1px solid #e2e8f0',
                    background: isSelected ? '#eff6ff' : '#fff',
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 4,
                  }}
                  onClick={() => {
                    setSelectedId(it.id)
                    setViewMode('selected')
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 600, fontSize: 14 }}>{it.name}</span>
                    <span
                      style={{
                        fontSize: 11,
                        padding: '2px 6px',
                        borderRadius: 4,
                        background: it.status === 'failed' ? '#fee2e2' : '#dcfce7',
                        color: it.status === 'failed' ? '#b91c1c' : '#15803d',
                        fontWeight: 500,
                      }}
                    >
                      {it.status}
                    </span>
                  </div>
                  {it.filename && (
                    <span style={{ fontSize: 11, color: '#64748b', wordBreak: 'break-all' }}>
                      {it.filename}
                    </span>
                  )}
                  {it.error && (
                    <span style={{ fontSize: 11, color: '#dc2626' }}>{it.error}</span>
                  )}
                </li>
              )
            })}
          </ul>
        </div>

        {/* Right Side: Candidate JSON Inspector / Unified View */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {/* View Mode Toggle Bar */}
          <div style={{ display: 'flex', gap: 10, marginBottom: 12 }}>
            <button
              onClick={() => setViewMode('selected')}
              style={{
                padding: '6px 14px',
                borderRadius: 6,
                cursor: 'pointer',
                fontWeight: 600,
                fontSize: 13,
                border: viewMode === 'selected' ? '2px solid #3b82f6' : '1px solid #cbd5e1',
                background: viewMode === 'selected' ? '#eff6ff' : '#fff',
                color: viewMode === 'selected' ? '#1d4ed8' : '#64748b',
                transition: 'all 0.2s',
              }}
            >
              ● Selected Candidate {selected ? `(${selected.name})` : ''}
            </button>
            <button
              onClick={() => setViewMode('all')}
              disabled={items.length === 0}
              style={{
                padding: '6px 14px',
                borderRadius: 6,
                cursor: items.length === 0 ? 'not-allowed' : 'pointer',
                fontWeight: 600,
                fontSize: 13,
                border: viewMode === 'all' ? '2px solid #3b82f6' : '1px solid #cbd5e1',
                background: viewMode === 'all' ? '#eff6ff' : '#fff',
                color: viewMode === 'all' ? '#1d4ed8' : '#64748b',
                transition: 'all 0.2s',
              }}
            >
              ★ View All JSONs at Once ({items.length})
            </button>
          </div>

          {/* VIEW MODE 1: ALL JSONs AT ONCE */}
          {viewMode === 'all' && items.length > 0 && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <div>
                  <h3 style={{ margin: 0 }}>All Parsed Resumes ({items.length} Candidates)</h3>
                  <span style={{ fontSize: 12, color: '#64748b' }}>
                    Unified JSON array containing all candidate records
                  </span>
                </div>
                <div>
                  <button
                    onClick={() => copyJson(allCandidatesData)}
                    style={{
                      padding: '6px 12px',
                      marginRight: 8,
                      background: '#f1f5f9',
                      border: '1px solid #cbd5e1',
                      borderRadius: 4,
                      cursor: 'pointer',
                      fontWeight: 500,
                    }}
                  >
                    {copied ? '✓ All Copied!' : `Copy All JSON (${items.length})`}
                  </button>
                  <button
                    onClick={downloadAllBatchJson}
                    style={{
                      padding: '6px 12px',
                      background: '#0284c7',
                      color: '#fff',
                      border: 'none',
                      borderRadius: 4,
                      cursor: 'pointer',
                      fontWeight: 500,
                    }}
                  >
                    Download All JSON
                  </button>
                </div>
              </div>

              <pre
                style={{
                  background: '#0f172a',
                  color: '#38bdf8',
                  padding: 16,
                  borderRadius: 8,
                  overflow: 'auto',
                  maxHeight: '68vh',
                  fontSize: 13,
                  lineHeight: 1.5,
                }}
              >
                {JSON.stringify(allCandidatesData, null, 2)}
              </pre>
            </div>
          )}

          {/* VIEW MODE 2: SINGLE SELECTED CANDIDATE */}
          {viewMode === 'selected' && (
            selected ? (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                  <div>
                    <h3 style={{ margin: 0 }}>{selected.name}</h3>
                    <span style={{ fontSize: 12, color: '#64748b' }}>{selected.filename}</span>
                  </div>
                  <div>
                    <button
                      onClick={() => copyJson(selected.data)}
                      style={{
                        padding: '6px 12px',
                        marginRight: 8,
                        background: '#f1f5f9',
                        border: '1px solid #cbd5e1',
                        borderRadius: 4,
                        cursor: 'pointer',
                      }}
                    >
                      {copied ? '✓ Copied!' : 'Copy JSON'}
                    </button>
                    <button
                      onClick={() => downloadJson(selected.data, `${selected.data.id || selected.name}.json`)}
                      style={{
                        padding: '6px 12px',
                        background: '#3b82f6',
                        color: '#fff',
                        border: 'none',
                        borderRadius: 4,
                        cursor: 'pointer',
                      }}
                    >
                      Download JSON
                    </button>
                  </div>
                </div>

                <pre
                  style={{
                    background: '#0f172a',
                    color: '#38bdf8',
                    padding: 16,
                    borderRadius: 8,
                    overflow: 'auto',
                    maxHeight: '68vh',
                    fontSize: 13,
                    lineHeight: 1.5,
                  }}
                >
                  {JSON.stringify(selected.data, null, 2)}
                </pre>
              </div>
            ) : (
              <div style={{ height: '50vh', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8' }}>
                Select a candidate from the left list to inspect their JSON
              </div>
            )
          )}
        </div>
      </div>
    </div>
  )
}
