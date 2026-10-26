/**
 * Contract test: the TypeScript mirror against the real OpenAPI document.
 *
 * `src/api/types.ts` is hand-written, which is a deliberate choice (the surface
 * is small), but a hand-written mirror drifts. This test reads the OpenAPI
 * snapshot the backend actually emits and the type file itself, and fails if
 * they disagree — a field renamed, added or removed in `schemas.py` shows up
 * here rather than as an `undefined` in the UI.
 *
 * The snapshot is regenerated with `frontend/scripts/export_openapi.py`; if this
 * test fails after a backend change, re-run that script and reconcile the types.
 *
 * It parses the `.ts` source text rather than inspecting values, because types
 * do not exist at runtime — parsing the declaration is the only way to compare
 * the mirror against the spec.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";

const here = dirname(fileURLToPath(import.meta.url));
const typesSource = readFileSync(resolve(here, "../api/types.ts"), "utf8");
const spec = JSON.parse(
  readFileSync(resolve(here, "contract/openapi.json"), "utf8"),
) as OpenApiDocument;

interface OpenApiDocument {
  components: {
    schemas: Record<string, OpenApiSchema>;
  };
}

interface OpenApiSchema {
  properties?: Record<string, unknown>;
  required?: string[];
  enum?: string[];
  type?: string;
}

/**
 * Extract the top-level property names of a TypeScript interface.
 *
 * Tracks brace depth so a nested object type's keys are not mistaken for the
 * interface's own, and skips JSDoc and comment lines.
 */
function interfaceFields(name: string): string[] {
  const declaration = new RegExp(`export interface ${name}\\b[^{]*\\{`, "m");
  const match = declaration.exec(typesSource);
  if (!match) throw new Error(`Interface ${name} not found in types.ts`);

  const start = match.index + match[0].length;
  let depth = 1;
  let index = start;
  while (index < typesSource.length && depth > 0) {
    const char = typesSource[index];
    if (char === "{") depth += 1;
    else if (char === "}") depth -= 1;
    index += 1;
  }
  const body = typesSource.slice(start, index - 1);

  const fields: string[] = [];
  let nested = 0;
  for (const rawLine of body.split("\n")) {
    const line = rawLine.trim();
    if (line.startsWith("//") || line.startsWith("*") || line.startsWith("/*")) {
      continue;
    }
    if (nested === 0) {
      const field = /^([A-Za-z_$][\w$]*)\??\s*:/.exec(line);
      if (field) fields.push(field[1]);
    }
    nested += (line.match(/[{[(]/g) ?? []).length;
    nested -= (line.match(/[}\])]/g) ?? []).length;
  }
  return fields;
}

/** Extract the string literals of a TypeScript union type alias. */
function unionValues(name: string): string[] {
  const declaration = new RegExp(
    `export type ${name}\\s*=([\\s\\S]*?);`,
    "m",
  );
  const match = declaration.exec(typesSource);
  if (!match) throw new Error(`Type alias ${name} not found in types.ts`);
  return [...match[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
}

const SCHEMA_TO_INTERFACE: Record<string, string> = {
  HealthResponse: "HealthResponse",
  SystemResponse: "SystemResponse",
  ComponentStatus: "ComponentStatus",
  LanguageInfo: "LanguageInfo",
  LanguagesResponse: "LanguagesResponse",
  BackendInfo: "BackendInfo",
  PresetInfo: "PresetInfo",
  RouteInfo: "RouteInfo",
  OptionItem: "OptionItem",
  OptionsResponse: "OptionsResponse",
  VoicesResponse: "VoicesResponse",
  StageStatus: "StageStatus",
  SegmentCounts: "SegmentCounts",
  OutputInfo: "OutputInfo",
  ResumeInfo: "ResumeInfo",
  JobStatusResponse: "JobStatusResponse",
  JobListResponse: "JobListResponse",
  MediaInfo: "MediaInfo",
  UploadResponse: "UploadResponse",
  JobCreateRequest: "JobCreateRequest",
  CancelResponse: "CancelResponse",
  ResumeResponse: "ResumeResponse",
  ArtifactDescriptor: "ArtifactDescriptor",
  ArtifactListResponse: "ArtifactListResponse",
  LogsResponse: "LogsResponse",
  OutputResponse: "OutputResponse",
  DeleteResponse: "DeleteResponse",
  SampleInfo: "SampleInfo",
  SampleListResponse: "SampleListResponse",
};

const ENUM_TO_UNION: Record<string, string> = {
  JobState: "JobState",
  ComponentState: "ComponentState",
  RunMode: "RunMode",
  CleanupLevel: "CleanupLevel",
};

describe("OpenAPI snapshot", () => {
  it("is present and has the schemas the app depends on", () => {
    for (const schemaName of Object.keys(SCHEMA_TO_INTERFACE)) {
      expect(
        spec.components.schemas[schemaName],
        `Schema ${schemaName} missing from the snapshot`,
      ).toBeDefined();
    }
  });
});

describe("response models match the backend", () => {
  for (const [schemaName, interfaceName] of Object.entries(SCHEMA_TO_INTERFACE)) {
    it(`${interfaceName} declares exactly the fields of ${schemaName}`, () => {
      const schema = spec.components.schemas[schemaName];
      const specFields = Object.keys(schema.properties ?? {}).sort();
      const tsFields = interfaceFields(interfaceName).sort();

      // Report both directions so a failure says which field is wrong.
      expect(tsFields, `extra or missing in ${interfaceName}`).toEqual(
        specFields,
      );
    });

    it(`${interfaceName} marks the same fields optional as ${schemaName}`, () => {
      const schema = spec.components.schemas[schemaName];
      const required = new Set(schema.required ?? []);
      const total = Object.keys(schema.properties ?? {}).length;

      // A response model with no required fields at all would make the
      // optionality check vacuous; the ones mirrored here all have some.
      if (total > 0 && required.size > 0) {
        expect(required.size).toBeLessThanOrEqual(total);
      }
      // `job_id`, `state`, `status`, `resume` and `events_url` are the fields
      // the UI cannot function without, so they must be required server-side.
      if (schemaName === "JobStatusResponse") {
        for (const field of [
          "job_id",
          "state",
          "status",
          "resume",
          "events_url",
        ]) {
          expect(required.has(field), `${field} should be required`).toBe(true);
        }
      }
    });
  }
});

describe("enums match the backend", () => {
  for (const [schemaName, unionName] of Object.entries(ENUM_TO_UNION)) {
    it(`${unionName} has exactly the values of ${schemaName}`, () => {
      const specValues = (spec.components.schemas[schemaName].enum ?? []).sort();
      const tsValues = unionValues(unionName).sort();
      expect(tsValues).toEqual(specValues);
    });
  }
});
