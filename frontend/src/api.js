export async function getHealth() {
  const res = await fetch('/health')
  if (!res.ok) {
    throw new Error(`Server returned ${res.status}`)
  }
  return res.json()
}

export async function uploadResume(file) {
  const form = new FormData()
  form.append('file', file)

  const res = await fetch('/api/v1/parse-resume', {
    method: 'POST',
    body: form,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Server returned ${res.status}`)
  }

  return res.json()
}

export async function pollJob(jobId) {
  while (true) {
    const res = await fetch(`/api/v1/job/${jobId}`)
    if (!res.ok) {
      throw new Error(`Job check failed: ${res.status}`)
    }
    const job = await res.json()
    if (job.status !== 'processing') {
      return job
    }
    await new Promise((resolve) => setTimeout(resolve, 2500))
  }
}