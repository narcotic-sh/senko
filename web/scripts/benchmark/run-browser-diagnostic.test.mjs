import assert from "node:assert/strict";
import test from "node:test";

import {
  decodeDiagnosticState,
  normalizeDiagnosticUrl,
  parseDiagnosticArguments,
} from "./run-browser-diagnostic.mjs";

test("argument parser defaults to the raw CAM++ graph diagnostic", () => {
  const parsed = parseDiagnosticArguments([]);
  assert.equal(parsed.help, false);
  assert.equal(
    parsed.options.url,
    "http://127.0.0.1:4173/?raw-campplus-graph-diagnostic=1",
  );
  assert.equal(
    parsed.options.eventName,
    "senko-raw-campplus-graph-diagnostic",
  );
  assert.equal(parsed.options.selector, "#raw-campplus-graph-result");
  assert.deepEqual(parsed.options.terminalStatuses, ["passed", "failed", "error"]);
  assert.equal(parsed.options.keepProfile, false);
});

test("argument parser supports event-only and caller-selected DOM capture", () => {
  const eventOnly = parseDiagnosticArguments([
    "--url",
    "http://127.0.0.1:5173/?probe=1",
    "--event",
    "probe-finished",
    "--no-selector",
    "--timeout-ms",
    "987",
    "--keep-profile",
  ]).options;
  assert.equal(eventOnly.eventName, "probe-finished");
  assert.equal(eventOnly.selector, undefined);
  assert.equal(eventOnly.timeoutMs, 987);
  assert.equal(eventOnly.keepProfile, true);

  const domOnly = parseDiagnosticArguments([
    "--no-event",
    "--selector",
    "#answer",
    "--status-attribute",
    "data-state",
    "--status",
    "complete",
    "--status",
    "broken",
  ]).options;
  assert.equal(domOnly.eventName, undefined);
  assert.equal(domOnly.selector, "#answer");
  assert.equal(domOnly.statusAttribute, "data-state");
  assert.deepEqual(domOnly.terminalStatuses, ["complete", "broken"]);
});

test("argument parser rejects contradictory or incomplete capture options", () => {
  assert.throws(
    () => parseDiagnosticArguments(["--event", "done", "--no-event"]),
    /mutually exclusive/,
  );
  assert.throws(
    () => parseDiagnosticArguments(["--selector", "#x", "--no-selector"]),
    /mutually exclusive/,
  );
  assert.throws(
    () => parseDiagnosticArguments(["--no-event", "--no-selector"]),
    /At least one/,
  );
  assert.throws(
    () => parseDiagnosticArguments(["--keep-profile", "--remove-profile"]),
    /mutually exclusive/,
  );
  assert.throws(
    () => parseDiagnosticArguments(["--timeout-ms", "0"]),
    /positive integer/,
  );
  assert.throws(
    () => parseDiagnosticArguments(["--unknown"]),
    /Unknown argument/,
  );
});

test("diagnostic URL normalization only accepts absolute HTTP URLs", () => {
  assert.equal(
    normalizeDiagnosticUrl("http://127.0.0.1:4173/?probe=1"),
    "http://127.0.0.1:4173/?probe=1",
  );
  assert.throws(() => normalizeDiagnosticUrl("/?probe=1"), /must be absolute/);
  assert.throws(() => normalizeDiagnosticUrl("file:///probe.html"), /HTTP\(S\)/);
});

test("event capture takes precedence and returns parsed JSON", () => {
  assert.deepEqual(
    decodeDiagnosticState(
      {
        event: { fired: true, json: '{"ok":true,"runs":[1,2]}' },
        dom: { status: "failed", text: '{"ok":false}' },
      },
      ["passed", "failed", "error"],
    ),
    {
      source: "event",
      result: { ok: true, runs: [1, 2] },
    },
  );
});

test("DOM capture waits for a selected terminal status and parses its text", () => {
  assert.equal(
    decodeDiagnosticState(
      { dom: { status: "running", text: "warming" } },
      ["passed", "failed"],
    ),
    undefined,
  );
  assert.deepEqual(
    decodeDiagnosticState(
      { dom: { status: "passed", text: '{"ok":true}' } },
      ["passed", "failed"],
    ),
    {
      source: "dom",
      status: "passed",
      result: { ok: true },
    },
  );
});

test("capture decoding reports serialization, selector, and JSON failures", () => {
  assert.throws(
    () =>
      decodeDiagnosticState(
        { event: { fired: true, serializationError: "cyclic value" } },
        ["passed"],
      ),
    /cyclic value/,
  );
  assert.throws(
    () =>
      decodeDiagnosticState(
        { dom: { selectorError: "not a selector" } },
        ["passed"],
      ),
    /Invalid result selector/,
  );
  assert.throws(
    () =>
      decodeDiagnosticState(
        { dom: { status: "passed", text: "not JSON" } },
        ["passed"],
      ),
    /invalid JSON/,
  );
});
