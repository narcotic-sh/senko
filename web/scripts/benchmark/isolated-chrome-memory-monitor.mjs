#!/usr/bin/env node

/**
 * RSS monitor scoped to one isolated Chrome browser process tree.
 *
 * Launch Chrome with a unique --user-data-dir and pass the resulting browser
 * PID here. Only that PID and its descendants are sampled, so unrelated Chrome
 * profiles, windows, tabs, extensions, and their shared GPU process are never
 * counted. Write any line to stdin (or send SIGINT/SIGTERM) to print JSON.
 */

import { execFileSync } from "node:child_process";

const POLL_MS = 100;
const rootPid = Number(process.argv[2]);
if (!Number.isSafeInteger(rootPid) || rootPid <= 0) {
  throw new Error(
    "Usage: node isolated-chrome-memory-monitor.mjs <isolated-browser-pid>",
  );
}

function allProcesses() {
  const output = execFileSync(
    "ps",
    ["-axo", "pid=,ppid=,rss=,command="],
    { encoding: "utf8" },
  );
  const processes = new Map();
  for (const line of output.split("\n")) {
    const match = /^\s*(\d+)\s+(\d+)\s+(\d+)\s+(.+)$/.exec(line);
    if (match === null) continue;
    const command = match[4];
    processes.set(Number(match[1]), {
      ppid: Number(match[2]),
      rssBytes: Number(match[3]) * 1024,
      command,
      kind: command.includes("--type=gpu-process")
        ? "gpu"
        : command.includes("--type=renderer")
          ? command.includes("--extension-process")
            ? "extension-renderer"
            : "renderer"
          : command.includes("--type=utility")
            ? "utility"
            : command.includes("--type=")
              ? "other-child"
              : "browser",
    });
  }
  return processes;
}

function scopedSnapshot() {
  const all = allProcesses();
  if (!all.has(rootPid)) {
    throw new Error(`Isolated Chrome root PID ${rootPid} is not running`);
  }
  const scoped = new Map();
  const included = new Set([rootPid]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const [pid, process] of all) {
      if (!included.has(pid) && included.has(process.ppid)) {
        included.add(pid);
        changed = true;
      }
    }
  }
  for (const pid of included) {
    const process = all.get(pid);
    if (process !== undefined) scoped.set(pid, process);
  }
  return scoped;
}

const baseline = scopedSnapshot();
const baselineAggregateBytes = [...baseline.values()].reduce(
  (sum, process) => sum + process.rssBytes,
  0,
);
const peaks = new Map();
let sampleCount = 0;
let peakAggregateBytes = baselineAggregateBytes;

function sample() {
  const current = scopedSnapshot();
  sampleCount += 1;
  const aggregateBytes = [...current.values()].reduce(
    (sum, process) => sum + process.rssBytes,
    0,
  );
  peakAggregateBytes = Math.max(peakAggregateBytes, aggregateBytes);
  for (const [pid, process] of current) {
    const existing = peaks.get(pid);
    if (existing === undefined) {
      peaks.set(pid, {
        pid,
        kind: process.kind,
        baselineBytes: baseline.get(pid)?.rssBytes ?? 0,
        peakBytes: process.rssBytes,
      });
    } else {
      existing.peakBytes = Math.max(existing.peakBytes, process.rssBytes);
    }
  }
}

let summarized = false;
function summarize() {
  if (summarized) return;
  summarized = true;
  clearInterval(timer);
  sample();
  const processes = [...peaks.values()]
    .map((process) => ({
      ...process,
      appearedDuringRun: !baseline.has(process.pid),
      peakDeltaBytes: Math.max(0, process.peakBytes - process.baselineBytes),
    }))
    .sort((left, right) => right.peakDeltaBytes - left.peakDeltaBytes);
  process.stdout.write(
    `${JSON.stringify(
      {
        scope: "isolated Chrome browser PID and descendants only",
        rootPid,
        pollIntervalMs: POLL_MS,
        sampleCount,
        baselineAggregateBytes,
        peakAggregateBytes,
        peakAggregateDeltaBytes: Math.max(
          0,
          peakAggregateBytes - baselineAggregateBytes,
        ),
        processes,
      },
      null,
      2,
    )}\n`,
  );
  process.exit(0);
}

const timer = setInterval(sample, POLL_MS);
sample();
process.stdin.setEncoding("utf8");
process.stdin.once("data", summarize);
process.once("SIGINT", summarize);
process.once("SIGTERM", summarize);
