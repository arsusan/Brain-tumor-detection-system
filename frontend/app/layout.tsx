import { Brain, Home, Database, Settings, Activity } from 'lucide-react';
import Link from 'next/link';
import './globals.css';

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-[#f1f5f9] flex min-h-screen">
        {/* Persistent Sidebar */}
        <aside className="w-64 bg-slate-900 text-slate-400 p-6 flex flex-col fixed h-full z-50">
          <div className="flex items-center gap-3 text-white mb-10 px-2">
            <div className="bg-blue-600 p-2 rounded-xl shadow-lg shadow-blue-500/20">
              <Brain size={24} />
            </div>
            <span className="font-bold text-xl tracking-tight">NeuroScan <span className="text-blue-500">AI</span></span>
          </div>

          <nav className="space-y-2 flex-1">
            <NavItem href="/" icon={<Home size={20} />} label="Dashboard" />
            <NavItem href="/history" icon={<Database size={20} />} label="Clinical History" />
            <div className="pt-4 mt-4 border-t border-slate-800">
              <p className="px-3 text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2">System</p>
              <NavItem href="#" icon={<Activity size={20} />} label="Model Analytics" disabled />
              <NavItem href="#" icon={<Settings size={20} />} label="Settings" disabled />
            </div>
          </nav>

          <div className="bg-slate-800/50 p-4 rounded-2xl border border-slate-700/50 mt-auto">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
              <span className="text-xs font-bold text-white uppercase">Server Online</span>
            </div>
            <p className="text-[10px] leading-relaxed">FastAPI 0.1.0 Connection Established</p>
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 ml-64 min-h-screen relative">
          {children}
        </main>
      </body>
    </html>
  );
}

function NavItem({ href, icon, label, disabled = false }: any) {
  return (
    <Link 
      href={href} 
      className={`flex items-center gap-3 px-3 py-3 rounded-xl transition-all duration-200 group font-medium text-sm
      ${disabled ? 'opacity-40 cursor-not-allowed' : 'hover:bg-slate-800 hover:text-white'}`}
    >
      {icon}
      <span>{label}</span>
    </Link>
  );
}