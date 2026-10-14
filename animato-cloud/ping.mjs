#!/usr/bin/env node
// Tells the app what the GitHub runner is doing during setup — before the renderer
// starts — so the person sees "Installing the video tools…" instead of a frozen bar.
//   node animato-cloud/ping.mjs "<step>" <progress> ["<detail>"]
// Never fails the workflow: any error is ignored.
import fs from 'node:fs';

const [step = '', progressArg = '', detail = ''] = process.argv.slice(2);
try {
  const ev = JSON.parse(fs.readFileSync(process.env.GITHUB_EVENT_PATH || '', 'utf8'));
  const p = ev.client_payload || ev.inputs || {};
  const job = p.job || {};
  const app = String(p.app_url || '').replace(/\/+$/, '');
  const key = String(p.auth?.runner_key || '');
  let route = '';
  if (job.studio_job) {
    try { const s = typeof job.studio_job === 'string' ? JSON.parse(job.studio_job) : job.studio_job; if (s?.id) route = `/api/animate/runner/${encodeURIComponent(s.id)}/status`; } catch {}
  } else if (p.campaign_id) route = `/api/automation/campaigns/${encodeURIComponent(p.campaign_id)}/status`;
  if (app && key && route) {
    const body = { status: 'running', step, detail: detail || step, partNumber: Number(p.part_number) || undefined, runUrl: process.env.GITHUB_SERVER_URL && process.env.GITHUB_RUN_ID ? `${process.env.GITHUB_SERVER_URL}/${process.env.GITHUB_REPOSITORY}/actions/runs/${process.env.GITHUB_RUN_ID}` : undefined, runId: process.env.GITHUB_RUN_ID };
    if (progressArg !== '') body.progress = Number(progressArg) || 0;
    await fetch(`${app}${route}`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Animato-Runner-Key': key }, body: JSON.stringify(body), signal: AbortSignal.timeout(8000) }).catch(() => {});
  }
} catch { /* setup must never fail because of a progress ping */ }
