"use client";

import { BarChart3, FileText, GitCompareArrows, LayoutDashboard, Menu, MessageSquareText, Moon, Network, Sun, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "next-themes";
import { useEffect, useState, type ReactNode } from "react";
import { useHealth } from "./providers";
import { cn } from "./ui";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/ask", label: "Ask Documents", icon: MessageSquareText },
  { href: "/trace", label: "Agent Trace", icon: Network },
  { href: "/documents", label: "Document Library", icon: FileText },
  { href: "/compare", label: "Compare vs Baseline", icon: GitCompareArrows },
  { href: "/evaluation", label: "Evaluation", icon: BarChart3 },
];

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  // eslint-disable-next-line react-hooks/set-state-in-effect -- theme is only known on the client
  useEffect(() => setMounted(true), []);
  const dark = mounted && resolvedTheme === "dark";
  return (
    <button
      onClick={() => setTheme(dark ? "light" : "dark")}
      className="inline-flex size-9 items-center justify-center rounded-md border border-border bg-surface text-muted transition-colors hover:bg-surface-2 hover:text-fg"
      aria-label={mounted ? `Switch to ${dark ? "light" : "dark"} theme` : "Toggle theme"}
      title="Toggle light/dark theme"
    >
      {mounted ? dark ? <Sun className="size-4" /> : <Moon className="size-4" /> : <span className="size-4" />}
    </button>
  );
}

function BackendStatus() {
  const { health, error, loading } = useHealth();
  let dot = "bg-subtle", text = "Checking backend…";
  if (!loading && error && !health) [dot, text] = ["bg-danger", "Backend offline"];
  else if (error) [dot, text] = ["bg-warning", "Backend unreachable"];
  else if (health) {
    const genReady = health.models.generator.ready;
    [dot, text] = genReady ? ["bg-success", "Backend online"] : ["bg-warning", "Generator not ready"];
  }
  return (
    <div className="space-y-2 text-xs">
      <div className="flex items-center gap-2 text-muted">
        <span className={cn("size-2 rounded-full", dot)} aria-hidden />
        <span>{text}</span>
      </div>
      {health && (
        <p className={cn("rounded px-2 py-1 font-medium", health.mode === "dev" ? "bg-warning-soft text-warning" : "bg-surface-2 text-muted")}>
          {health.mode === "dev" ? "DEV MODE · no LLM" : `Local models · ${health.models.generator.name.replace("ollama:", "")}`}
        </p>
      )}
    </div>
  );
}

function NavLinks({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();
  return (
    <nav className="space-y-0.5" aria-label="Main">
      {NAV.map(({ href, label, icon: Icon }) => {
        const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            onClick={onNavigate}
            aria-current={active ? "page" : undefined}
            className={cn(
              "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors",
              active ? "bg-accent-soft font-medium text-accent" : "text-muted hover:bg-surface-2 hover:text-fg",
            )}
          >
            <Icon className="size-4 shrink-0" aria-hidden />
            {label}
          </Link>
        );
      })}
    </nav>
  );
}

function Brand() {
  return (
    <Link href="/" className="block">
      <p className="text-sm font-semibold leading-tight text-fg">Multi-Agent RAG</p>
      <p className="text-xs text-muted">Research prototype</p>
    </Link>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="flex min-h-screen">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded focus:bg-surface focus:px-3 focus:py-2">
        Skip to content
      </a>
      {/* desktop sidebar */}
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-border bg-surface px-3 py-5 lg:flex">
        <div className="px-3 pb-6"><Brand /></div>
        <NavLinks />
        <div className="mt-auto space-y-4 px-3">
          <BackendStatus />
          <div className="flex items-center justify-between">
            <span className="text-xs text-subtle">Theme</span>
            <ThemeToggle />
          </div>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        {/* mobile / tablet top bar */}
        <header className="sticky top-0 z-30 flex items-center justify-between border-b border-border bg-surface px-4 py-3 lg:hidden">
          <button onClick={() => setOpen(true)} className="inline-flex size-9 items-center justify-center rounded-md border border-border text-muted" aria-label="Open navigation">
            <Menu className="size-4" />
          </button>
          <Brand />
          <ThemeToggle />
        </header>
        {open && (
          <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
            <div className="absolute inset-0 bg-black/40" onClick={() => setOpen(false)} />
            <div className="absolute inset-y-0 left-0 flex w-72 flex-col bg-surface px-3 py-5 shadow-xl">
              <div className="flex items-center justify-between px-3 pb-6">
                <Brand />
                <button onClick={() => setOpen(false)} className="text-muted" aria-label="Close navigation"><X className="size-5" /></button>
              </div>
              <NavLinks onNavigate={() => setOpen(false)} />
              <div className="mt-auto px-3"><BackendStatus /></div>
            </div>
          </div>
        )}
        <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:px-6 lg:px-10 lg:py-8">
          {children}
        </main>
        <footer className="border-t border-border px-4 py-4 text-center text-xs text-subtle sm:px-6">
          RAG-Based Adaptive Framework for Multi-Agent Systems · Local research prototype
        </footer>
      </div>
    </div>
  );
}
