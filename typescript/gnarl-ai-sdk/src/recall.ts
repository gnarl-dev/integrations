import { GnarlClient, type Memory, type RecallResponse } from "gnarl-client";

/** Where memories live and whose they are. Fixed by the developer, never chosen by the model. */
export interface GnarlMemoryScope {
  /**
   * A `GnarlClient`. Default: `new GnarlClient()`, which finds the node itself —
   * `$GNARL_URL`, else the endpoint the node on this machine recorded, else
   * the Gnarly app's `http://127.0.0.1:43300`.
   */
  client?: GnarlClient;
  /** Gnarl memory namespace. With neither this nor `space`, the node's default personal space. */
  namespace?: string;
  /** `personal` stays on this device; `household` is replicated to the mesh's peers. */
  space?: string;
  /** Whose memories these are. Recall is scoped to this user by the node. */
  user?: string;
  /** Tag stored memories with this session, and restrict recall to it. */
  session?: string;
}

/** The scope fields a remember or recall request carries. */
export function scopeFields(scope: GnarlMemoryScope): { namespace?: string; space?: string; user?: string } {
  const out: { namespace?: string; space?: string; user?: string } = {};
  if (scope.namespace !== undefined) out.namespace = scope.namespace;
  if (scope.space !== undefined) out.space = scope.space;
  if (scope.user !== undefined) out.user = scope.user;
  return out;
}

export function clientOf(scope: GnarlMemoryScope): GnarlClient {
  return scope.client ?? new GnarlClient();
}

/** A recall, with its completeness alongside. */
export interface Recalled {
  memories: Memory[];
  /**
   * True when part of the memory space did not answer: `memories` is then a
   * subset, and a memory that is missing may still exist.
   */
  partial: boolean;
  coverage: unknown;
}

export async function recall(
  client: GnarlClient,
  scope: GnarlMemoryScope,
  query: string,
  k: number,
  signal?: AbortSignal,
): Promise<Recalled> {
  const res: RecallResponse = await client.memory.recall(
    { query, k, ...scopeFields(scope), ...(scope.session !== undefined ? { session: scope.session } : {}) },
    signal ? { signal } : undefined,
  );
  return { memories: res.memories, partial: res.partial === true, coverage: res.coverage ?? null };
}

/** Appended to injected memories when the recall was partial. */
export const PARTIAL_NOTE = "(Some memories may be missing: part of the memory space did not answer.)";
