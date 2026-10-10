"use client";

import { ThemeProvider } from "next-themes";
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api } from "@/lib/api";
import type { Health } from "@/lib/types";

interface HealthState {
  health: Health | null;
  error: string | null;
  loading: boolean;
  refresh: () => void;
}

const HealthContext = createContext<HealthState>({ health: null, error: null, loading: true, refresh: () => {} });

export const useHealth = () => useContext(HealthContext);

function HealthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<Omit<HealthState, "refresh">>({ health: null, error: null, loading: true });

  const refresh = useCallback(() => {
    api
      .health()
      .then((health) => setState({ health, error: null, loading: false }))
      .catch((e: Error) => setState((s) => ({ health: s.health, error: e.message, loading: false })));
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 15_000);
    return () => clearInterval(id);
  }, [refresh]);

  return <HealthContext.Provider value={{ ...state, refresh }}>{children}</HealthContext.Provider>;
}

export function Providers({ children }: { children: ReactNode }) {
  // next-themes injects a blocking script that applies the stored/system theme before paint (no flash).
  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange storageKey="arag-theme">
      <HealthProvider>{children}</HealthProvider>
    </ThemeProvider>
  );
}
