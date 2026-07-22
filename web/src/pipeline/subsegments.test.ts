import { describe, expect, it } from "vitest";

import { createSubsegments } from "./subsegments";

describe("createSubsegments", () => {
  it("uses Senko's default sliding-window policy", () => {
    expect(createSubsegments([{ start: 2, end: 5 }])).toEqual([
      { index: 0, start: 2, end: 3.5 },
      { index: 1, start: 2.6, end: 4.1 },
      { index: 2, start: 3.2, end: 4.7 },
      { index: 3, start: 3.5, end: 5 },
    ]);
  });

  it("preserves Senko's negative start for a short leading speech island", () => {
    expect(createSubsegments([{ start: 0.1, end: 0.5 }])).toEqual([
      { index: 0, start: -1, end: 0.5 },
    ]);
  });
});
