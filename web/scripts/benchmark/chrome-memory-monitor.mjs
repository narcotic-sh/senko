#!/usr/bin/env node

/**
 * Low-overhead macOS RSS monitor for all Chrome processes.
 *
 * This cannot attribute shared-profile processes to one tab. Prefer
 * isolated-chrome-memory-monitor.mjs for pipeline memory acceptance. This
 * broader tool remains useful only when the entire Chrome instance is already
 * known to be dedicated to the benchmark.
 */

import { execFileSync } from "node:child_process";

const POLL_MS = 100;
const CHROME_MARKER = "/Applications/Google Chrome.app/";

function snapshot() {
  const output = execFileSync("ps", ["-axo", "pid=,rss=,command="], {
    encoding: "utf8",
  });
  const processes = new Map();
  for (const line of output.split("\n")) {
    const match = /^\s*(\d+)\s+(\d+)\s+(.+)$/.exec(line);
    if (match === null || !match[3].includes(CHROME_MARKER)) continue;
    const command = match[3];
    processes.set(Number(match[1]), {
      rssBytes: Number(match[2]) * 1024,
      kind: command.includes("--type=gpu-process")
        ? "gpu"
        : command.includes("--type=renderer")
          ? "renderer"
          : command.endsWith("/Google Chrome")
            ? "browser"
            : "other",
      command,
    });
  }
  return processes;
}

const baseline = snapshot();
const peaks = new Map();
let sampleCount = 0;
let peakAggregateBytes = 0;
let peakPositiveDeltaBytes = 0;
let peakNewProcessBytes = 0;

function sample() {
  const current = snapshot();
  sampleCount += 1;
  let aggregateBytes = 0;
  let positiveDeltaBytes = 0;
  let newProcessBytes = 0;
  for (const [pid, process] of current) {
    aggregateBytes += process.rssBytes;
    const baselineBytes = baseline.get(pid)?.rssBytes ?? 0;
    positiveDeltaBytes += Math.max(0, process.rssBytes - baselineBytes);
    if (!baseline.has(pid)) newProcessBytes += process.rssBytes;
    const existing = peaks.get(pid);
    if (existing === undefined) {
      peaks.set(pid, {
        pid,
        kind: process.kind,
        baselineBytes: baseline.get(pid)?.rssBytes ?? 0,
        peakBytes: process.rssBytes,
        command: process.command,
      });
    } else {
      existing.peakBytes = Math.max(existing.peakBytes, process.rssBytes);
    }
  }
  peakAggregateBytes = Math.max(peakAggregateBytes, aggregateBytes);
  peakPositiveDeltaBytes = Math.max(peakPositiveDeltaBytes, positiveDeltaBytes);
  peakNewProcessBytes = Math.max(peakNewProcessBytes, newProcessBytes);
}

function summarize() {
  clearInterval(timer);
  sample();
  const processes = [...peaks.values()]
    .map((process) => ({
      pid: process.pid,
      kind: process.kind,
      appearedDuringRun: !baseline.has(process.pid),
      baselineBytes: process.baselineBytes,
      peakBytes: process.peakBytes,
      peakDeltaBytes: Math.max(0, process.peakBytes - process.baselineBytes),
    }))
    .filter(
      (process) =>
        process.appearedDuringRun ||
        process.kind === "gpu" ||
        process.kind === "browser" ||
        (process.kind === "renderer" && process.peakDeltaBytes >= 1024 * 1024),
    )
    .sort((left, right) => right.peakDeltaBytes - left.peakDeltaBytes);
  const baselineAggregateBytes = [...baseline.values()].reduce(
    (sum, process) => sum + process.rssBytes,
    0,
  );
  process.stdout.write(
    `${JSON.stringify(
      {
        scope: "all macOS Google Chrome processes",
        warning:
          "Not tab-scoped; unrelated Chrome windows/tabs can contribute. Prefer the isolated-profile monitor.",
        pollIntervalMs: POLL_MS,
        sampleCount,
        baselineAggregateBytes,
        peakAggregateBytes,
        peakAggregateDeltaBytes: Math.max(
          0,
          peakAggregateBytes - baselineAggregateBytes,
        ),
        // Aggregate RSS can hide growth when an unrelated baseline process
        // exits during a run. These counters retain positive per-process
        // growth and all newly created Chrome-process RSS respectively.
        peakPositiveDeltaBytes,
        peakNewProcessBytes,
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
