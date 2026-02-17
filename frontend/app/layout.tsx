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
      <body className="bg-white flex min-h-screen">

        {/* Sidebar - WHITE VERSION */}
        <aside className="w-64 bg-white border-r border-blue-100 p-6 flex flex-col fixed h-full">

          {/* Logo */}
          <div className="flex items-center gap-3 text-blue-900 mb-10">
            <div className="bg-blue-600 p-2 rounded-xl shadow-md">
              <Brain size={22} className="text-white" />
            </div>
            <span className="font-bold text-lg">
              NeuroScan <span className="text-blue-600">AI</span>
            </span>
          </div>

          <nav className="space-y-2 flex-1">
            <NavItem
              href="/"
              icon={<Home size={18} />}
              label="Dashboard"
              active={pathname === "/"}
            />
            <NavItem
              href="/history"
              icon={<Database size={18} />}
              label="Clinical History"
              active={pathname === "/history"}
            />
            <NavItem
              href="/analytics"
              icon={<Activity size={18} />}
              label="Analytics"
              active={pathname === "/analytics"}
            />
          </nav>

          {/* Server Status */}
          <div className="mt-auto p-4 rounded-xl bg-blue-50 border border-blue-100">
            <div className="flex items-center gap-2 mb-1">
              <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
              <span className="text-xs font-semibold text-blue-900">
                Server Online
              </span>
            </div>
            <p className="text-xs text-blue-500">
              FastAPI v0.1.0 Active
            </p>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 ml-64 bg-gradient-to-br from-white via-blue-50/40 to-white p-8">
          {children}
        </main>

      </body>
    </html>
  );
}

function NavItem({ href, icon, label, active }: any) {
  return (
    <Link
      href={href}
      className={`flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all
        ${
          active
            ? "bg-blue-600 text-white shadow-md"
            : "text-blue-700 hover:bg-blue-50"
        }`}
    >
      {icon}
      {label}
    </Link>
  );
}
