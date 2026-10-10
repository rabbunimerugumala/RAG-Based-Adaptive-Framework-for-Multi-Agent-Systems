import type { Metadata } from "next";
import { Suspense } from "react";
import { PageHeader, Spinner } from "@/components/ui";
import TraceView from "./trace-view";

export const metadata: Metadata = { title: "Agent Trace" };

export default function TracePage() {
  return (
    <>
      <PageHeader
        title="Agent Trace"
        description="The exact execution trace returned by the backend: each agent's inputs, outputs, timing and decision, grouped by retrieval attempt."
      />
      {/* useSearchParams needs a Suspense boundary so the page shell can be prerendered */}
      <Suspense fallback={<Spinner label="Loading…" />}>
        <TraceView />
      </Suspense>
    </>
  );
}
