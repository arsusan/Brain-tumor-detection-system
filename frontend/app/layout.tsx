"use client";

import { Brain, Home, Database, Activity } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import "./globals.css";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  return (
    <html lang="en">
      <body className="bg-white min-h-screen" suppressHydrationWarning>
        <div className="flex flex-col md:flex-row">
          <aside className="w-full md:w-64 md:fixed md:h-full bg-white border-b md:border-r border-blue-100 p-4 md:p-6 flex flex-col md:flex-col shadow-sm md:shadow-none z-10">
            <div className="flex items-center justify-between md:flex-col md:items-start gap-4 md:gap-6">
              <div className="flex items-center gap-3 text-blue-900">
                <div className="bg-blue-600 p-2 rounded-xl shadow-md">
                  <Brain size={22} className="text-white" />
                </div>
                <span className="font-bold text-lg">
                  NeuroScan <span className="text-blue-600">AI</span>
                </span>
              </div>

              <nav className="flex md:flex-col space-x-4 md:space-x-0 md:space-y-2 flex-1 md:w-full">
                <NavItem
                  href="/"
                  icon={<Home size={18} />}
                  label="Dashboard"
                  active={pathname === "/"}
                  mobileLayout={true}
                />
                <NavItem
                  href="/history"
                  icon={<Database size={18} />}
                  label="Clinical History"
                  active={pathname === "/history"}
                  mobileLayout={true}
                />
                <NavItem
                  href="/analytics"
                  icon={<Activity size={18} />}
                  label="Analytics"
                  active={pathname === "/analytics"}
                  mobileLayout={true}
                />
              </nav>
            </div>

            <div className="hidden md:block mt-auto p-4 rounded-xl bg-blue-50 border border-blue-100">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                <span className="text-xs font-semibold text-blue-900">
                  Server Online
                </span>
              </div>
              <p className="text-xs text-blue-500">FastAPI v0.1.0 Active</p>
            </div>
          </aside>

          <main className="flex-1 md:ml-64 bg-gradient-to-br from-white via-blue-50/40 to-white p-4 md:p-8 min-h-screen">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}

function NavItem({ href, icon, label, active }: any) {
  return (
    <Link
      href={href}
      className={`
        flex items-center gap-2 px-3 py-2 md:px-4 md:py-3 rounded-xl text-sm font-semibold transition-all whitespace-nowrap
        ${
          active
            ? "bg-blue-600 text-white shadow-md"
            : "text-blue-700 hover:bg-blue-50"
        }
      `}
    >
      {icon}
      <span className="hidden sm:inline md:inline">{label}</span>
    </Link>
  );
}