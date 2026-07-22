function normalizeSegments(segments, shape) {
  if (!Array.isArray(segments)) {
    throw new Error(`${shape} segments must be an array`);
  }
  return segments.map((segment, index) => {
    const start =
      shape === "offline" ? segment?.start : segment?.startSeconds;
    const end = shape === "offline" ? segment?.end : segment?.endSeconds;
    if (
      !Number.isFinite(start) ||
      !Number.isFinite(end) ||
      start < 0 ||
      end < start ||
      typeof segment?.speaker !== "string" ||
      segment.speaker.length === 0
    ) {
      throw new Error(`Malformed ${shape} segment at index ${index}`);
    }
    return { start, end, speaker: segment.speaker };
  });
}

function uniqueSpeakers(segments) {
  return [...new Set(segments.map((segment) => segment.speaker))].sort();
}

function rasterizeTimeline(segments, speakerIds, frameSeconds, frameCount) {
  const timeline = new Int16Array(frameCount);
  for (const segment of segments) {
    const speaker = speakerIds.get(segment.speaker);
    const startFrame = Math.max(0, Math.floor(segment.start / frameSeconds));
    const endFrame = Math.min(
      frameCount,
      Math.ceil(segment.end / frameSeconds),
    );
    timeline.fill(speaker, startFrame, endFrame);
  }
  return timeline;
}

function maximumWeightSpeakerMapping(weights, referenceSpeakers, hypothesisSpeakers) {
  if (referenceSpeakers.length === 0 || hypothesisSpeakers.length === 0) {
    return { matchedFrames: 0, mapping: [] };
  }

  let rows = referenceSpeakers;
  let columns = hypothesisSpeakers;
  let matrix = weights;
  let transposed = false;
  if (rows.length > columns.length) {
    transposed = true;
    rows = hypothesisSpeakers;
    columns = referenceSpeakers;
    matrix = Array.from({ length: rows.length }, (_, row) =>
      Array.from({ length: columns.length }, (_, column) =>
        weights[column][row],
      ),
    );
  }
  if (columns.length > 15) {
    throw new Error(
      `Offline scorer supports at most 15 speakers, received ${columns.length}`,
    );
  }

  const stateCount = 1 << columns.length;
  let scores = new Float64Array(stateCount);
  scores.fill(Number.NEGATIVE_INFINITY);
  scores[0] = 0;
  const parentMasks = [];
  const parentColumns = [];

  for (let row = 0; row < rows.length; row += 1) {
    const next = new Float64Array(stateCount);
    next.fill(Number.NEGATIVE_INFINITY);
    const masks = new Int32Array(stateCount);
    masks.fill(-1);
    const chosen = new Int16Array(stateCount);
    chosen.fill(-1);
    for (let mask = 0; mask < stateCount; mask += 1) {
      const score = scores[mask];
      if (!Number.isFinite(score)) continue;
      for (let column = 0; column < columns.length; column += 1) {
        const bit = 1 << column;
        if ((mask & bit) !== 0) continue;
        const nextMask = mask | bit;
        const candidate = score + matrix[row][column];
        if (candidate > next[nextMask]) {
          next[nextMask] = candidate;
          masks[nextMask] = mask;
          chosen[nextMask] = column;
        }
      }
    }
    scores = next;
    parentMasks.push(masks);
    parentColumns.push(chosen);
  }

  let bestMask = -1;
  let matchedFrames = Number.NEGATIVE_INFINITY;
  for (let mask = 0; mask < stateCount; mask += 1) {
    if (scores[mask] > matchedFrames) {
      matchedFrames = scores[mask];
      bestMask = mask;
    }
  }

  const mapping = [];
  for (let row = rows.length - 1; row >= 0; row -= 1) {
    const column = parentColumns[row][bestMask];
    const referenceSpeaker = transposed ? columns[column] : rows[row];
    const hypothesisSpeaker = transposed ? rows[row] : columns[column];
    mapping.push({
      referenceSpeaker,
      hypothesisSpeaker,
      jointSpeechFrames: matrix[row][column],
    });
    bestMask = parentMasks[row][bestMask];
  }
  mapping.reverse();
  return { matchedFrames, mapping };
}

export function scoreTimelineAtFrameMilliseconds(
  referenceSegments,
  hypothesisSegments,
  frameMilliseconds,
) {
  if (!Number.isSafeInteger(frameMilliseconds) || frameMilliseconds <= 0) {
    throw new Error("frameMilliseconds must be a positive integer");
  }
  const frameSeconds = frameMilliseconds / 1_000;
  const referenceSpeakers = uniqueSpeakers(referenceSegments);
  const hypothesisSpeakers = uniqueSpeakers(hypothesisSegments);
  const referenceIds = new Map(
    referenceSpeakers.map((speaker, index) => [speaker, index + 1]),
  );
  const hypothesisIds = new Map(
    hypothesisSpeakers.map((speaker, index) => [speaker, index + 1]),
  );
  const durationSeconds = Math.max(
    0,
    ...referenceSegments.map((segment) => segment.end),
    ...hypothesisSegments.map((segment) => segment.end),
  );
  const frameCount = Math.ceil(durationSeconds / frameSeconds);
  const reference = rasterizeTimeline(
    referenceSegments,
    referenceIds,
    frameSeconds,
    frameCount,
  );
  const hypothesis = rasterizeTimeline(
    hypothesisSegments,
    hypothesisIds,
    frameSeconds,
    frameCount,
  );
  const confusion = Array.from({ length: referenceSpeakers.length }, () =>
    new Uint32Array(hypothesisSpeakers.length),
  );
  let referenceSpeechFrames = 0;
  let hypothesisSpeechFrames = 0;
  let jointSpeechFrames = 0;
  let unionSpeechFrames = 0;
  for (let frame = 0; frame < frameCount; frame += 1) {
    const referenceSpeaker = reference[frame];
    const hypothesisSpeaker = hypothesis[frame];
    if (referenceSpeaker !== 0) referenceSpeechFrames += 1;
    if (hypothesisSpeaker !== 0) hypothesisSpeechFrames += 1;
    if (referenceSpeaker !== 0 || hypothesisSpeaker !== 0) {
      unionSpeechFrames += 1;
    }
    if (referenceSpeaker !== 0 && hypothesisSpeaker !== 0) {
      jointSpeechFrames += 1;
      confusion[referenceSpeaker - 1][hypothesisSpeaker - 1] += 1;
    }
  }

  const { matchedFrames, mapping } = maximumWeightSpeakerMapping(
    confusion,
    referenceSpeakers,
    hypothesisSpeakers,
  );
  return {
    frameMilliseconds,
    referenceSpeechFrames,
    hypothesisSpeechFrames,
    jointSpeechFrames,
    unionSpeechFrames,
    speechIntersectionOverUnion:
      unionSpeechFrames === 0 ? 1 : jointSpeechFrames / unionSpeechFrames,
    mappedSpeakerAgreementOnJointSpeech:
      jointSpeechFrames === 0 ? 1 : matchedFrames / jointSpeechFrames,
    matchedSpeakerFrames: matchedFrames,
    mapping,
  };
}

export function scoreAgainstOfflineSenkoReference(browserResult, reference) {
  if (browserResult === null || typeof browserResult !== "object") {
    throw new Error("Browser result must be an object");
  }
  if (reference === null || typeof reference !== "object") {
    throw new Error("Offline reference must be an object");
  }
  const referenceSegments = normalizeSegments(reference.merged_segments, "offline");
  const hypothesisSegments = normalizeSegments(browserResult.segments, "browser");
  const referenceSpeakerCount = Number.isSafeInteger(
    reference.merged_speakers_detected,
  )
    ? reference.merged_speakers_detected
    : uniqueSpeakers(referenceSegments).length;
  const hypothesisSpeakerCount = Number.isSafeInteger(browserResult.speakerCount)
    ? browserResult.speakerCount
    : uniqueSpeakers(hypothesisSegments).length;

  return {
    oracle: "offline-senko-merged-segments",
    referenceSpeakerCount,
    hypothesisSpeakerCount,
    speakerCountDelta: hypothesisSpeakerCount - referenceSpeakerCount,
    referenceSegmentCount: referenceSegments.length,
    hypothesisSegmentCount: hypothesisSegments.length,
    segmentCountDelta: hypothesisSegments.length - referenceSegments.length,
    timelines: {
      "10ms": scoreTimelineAtFrameMilliseconds(
        referenceSegments,
        hypothesisSegments,
        10,
      ),
      "50ms": scoreTimelineAtFrameMilliseconds(
        referenceSegments,
        hypothesisSegments,
        50,
      ),
    },
  };
}
