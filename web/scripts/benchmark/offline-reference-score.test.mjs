import assert from "node:assert/strict";
import test from "node:test";

import { scoreAgainstOfflineSenkoReference } from "./offline-reference-score.mjs";

test("offline scoring is invariant to speaker labels", () => {
  const score = scoreAgainstOfflineSenkoReference(
    {
      speakerCount: 2,
      segments: [
        { speaker: "browser-b", startSeconds: 0, endSeconds: 1 },
        { speaker: "browser-a", startSeconds: 1, endSeconds: 2 },
      ],
    },
    {
      merged_speakers_detected: 2,
      merged_segments: [
        { speaker: "offline-a", start: 0, end: 1 },
        { speaker: "offline-b", start: 1, end: 2 },
      ],
    },
  );

  assert.equal(score.timelines["10ms"].speechIntersectionOverUnion, 1);
  assert.equal(
    score.timelines["10ms"].mappedSpeakerAgreementOnJointSpeech,
    1,
  );
  assert.deepEqual(
    score.timelines["10ms"].mapping.map(
      ({ referenceSpeaker, hypothesisSpeaker }) => [
        referenceSpeaker,
        hypothesisSpeaker,
      ],
    ),
    [
      ["offline-a", "browser-b"],
      ["offline-b", "browser-a"],
    ],
  );
});

test("speech IoU counts every frame touched by a segment", () => {
  const score = scoreAgainstOfflineSenkoReference(
    {
      speakerCount: 1,
      segments: [
        { speaker: "hyp", startSeconds: 0.009, endSeconds: 0.021 },
      ],
    },
    {
      merged_speakers_detected: 1,
      merged_segments: [{ speaker: "ref", start: 0, end: 0.02 }],
    },
  );

  assert.equal(score.timelines["10ms"].jointSpeechFrames, 2);
  assert.equal(score.timelines["10ms"].unionSpeechFrames, 3);
  assert.equal(score.timelines["10ms"].speechIntersectionOverUnion, 2 / 3);
  assert.equal(score.speakerCountDelta, 0);
  assert.equal(score.segmentCountDelta, 0);
});
