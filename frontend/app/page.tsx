"use client";

import { useState } from 'react';
import Link from 'next/link';
import api from './lib/api'; 
import { 
  Upload, Activity, Brain, ShieldAlert, CheckCircle, 
  RefreshCcw, Database, BarChart3, History, Download, FileText, Info 
} from 'lucide-react';
import jsPDF from 'jspdf';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null;
    setFile(selectedFile);
    if (selectedFile) {
      setPreview(URL.createObjectURL(selectedFile));
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null); 

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await api.post('/predict', formData);
      setResult(response.data);
    } catch (error) {
      alert("Analysis Failed: Check if FastAPI is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  // --- Direct Export Logic ---
  const generateImmediateReport = async () => {
    if (!result) return;
    setIsExporting(true);
    try {
      const doc = new jsPDF('p', 'mm', 'a4');
      
      // Header
      doc.setFillColor(15, 23, 42);
      doc.rect(0, 0, 210, 40, 'F');
      doc.setTextColor(255, 255, 255);
      doc.setFontSize(22);
      doc.text("NEUROSCAN AI: INSTANT REPORT", 20, 25);

      // Body
      doc.setTextColor(30, 41, 59);
      doc.setFontSize(12);
      doc.text(`Patient File: ${file?.name || 'Unknown'}`, 20, 55);
      doc.text(`Analysis Date: ${new Date().toLocaleString()}`, 20, 62);
      
      doc.setFillColor(241, 245, 249);
      doc.roundedRect(20, 70, 170, 30, 3, 3, 'F');
      doc.setFont("helvetica", "bold");
      doc.text(`CLASSIFICATION: ${result.prediction.toUpperCase()}`, 30, 88);

      // Image Handling
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = result.heatmap_url.startsWith('http') ? result.heatmap_url : `http://127.0.0.1:8000/${result.heatmap_url}`;
      
      await new Promise((resolve) => {
        img.onload = () => {
          doc.addImage(img, 'JPEG', 20, 110, 100, 100);
          resolve(null);
        };
      });

      doc.setFontSize(8);
      doc.text("Disclaimer: This AI result must be verified by a neuro-radiologist.", 20, 280);
      doc.save(`NeuroScan_Report_${Date.now()}.pdf`);
    } finally {
      setIsExporting(false);
    }
  };

  const getStatusColor = (pred: string) => {
    if (pred === 'notumor') return 'text-emerald-600 bg-emerald-50 border-emerald-100';
    return 'text-rose-600 bg-rose-50 border-rose-100';
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 font-sans">
      {/* Navigation */}
      <nav className="bg-white border-b border-slate-200 px-8 py-4 flex items-center justify-between sticky top-0 z-50 shadow-sm">
        <div className="flex items-center gap-2">
          <div className="bg-blue-600 p-1.5 rounded-lg shadow-blue-200 shadow-lg">
            <Brain className="text-white" size={24} />
          </div>
          <h1 className="text-xl font-bold tracking-tight text-slate-800">NeuroScan AI <span className="text-blue-600 text-sm font-black uppercase ml-1 tracking-widest">Dash</span></h1>
        </div>
        
        <div className="flex items-center gap-6">
          <Link href="/history" className="flex items-center gap-2 text-sm font-bold text-slate-600 hover:text-blue-600 transition-colors bg-slate-50 px-4 py-2 rounded-xl border border-slate-200">
            <History size={18} />
            History Archive
          </Link>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto p-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* LEFT: UPLOAD & PREVIEW */}
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-white rounded-3xl shadow-sm border border-slate-200 p-6">
            <h2 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
              <Database size={14} /> Scan Input
            </h2>
            
            <div className="border-2 border-dashed border-slate-200 rounded-2xl p-4 text-center hover:border-blue-400 transition-all cursor-pointer bg-slate-50 relative min-h-[250px] flex flex-col items-center justify-center overflow-hidden group">
              {preview ? (
                <img src={preview} alt="Preview" className="absolute inset-0 w-full h-full object-cover group-hover:scale-110 transition-transform duration-500" />
              ) : (
                <div className="relative z-0 flex flex-col items-center">
                  <div className="p-4 bg-white rounded-full shadow-sm mb-3">
                    <Upload className="text-blue-500" size={24} />
                  </div>
                  <p className="text-sm font-bold text-slate-600">Drop MRI Scan Here</p>
                  <p className="text-[10px] text-slate-400 mt-1 uppercase font-bold">PNG, JPG up to 10MB</p>
                </div>
              )}
              <input type="file" accept="image/*" onChange={onFileChange} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" />
            </div>
            
            <button 
              onClick={handleUpload}
              disabled={!file || loading}
              className="mt-6 w-full bg-slate-900 text-white py-4 rounded-2xl font-black uppercase text-xs tracking-widest hover:bg-blue-600 disabled:bg-slate-100 disabled:text-slate-400 transition-all shadow-xl shadow-blue-900/10 flex justify-center items-center gap-3"
            >
              {loading ? <RefreshCcw className="animate-spin" size={18} /> : <Activity size={18} />}
              {loading ? "Processing..." : "Run AI Diagnostics"}
            </button>
          </div>
        </div>

        {/* RIGHT: LIVE REPORT */}
        <div className="lg:col-span-8">
          <div className="bg-white rounded-3xl shadow-sm border border-slate-200 overflow-hidden min-h-[600px] flex flex-col">
            {result ? (
              <>
                <div className="p-8 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
                  <div>
                    <h2 className="text-xl font-black text-slate-800 tracking-tight">Diagnostic Analysis</h2>
                    <p className="text-sm text-slate-500 font-medium">Real-time classification from neural engine</p>
                  </div>
                  <button 
                    onClick={generateImmediateReport}
                    disabled={isExporting}
                    className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 text-white rounded-xl text-xs font-black uppercase tracking-widest hover:bg-blue-700 transition-all shadow-lg shadow-blue-200 active:scale-95"
                  >
                    {isExporting ? <RefreshCcw size={16} className="animate-spin" /> : <Download size={16} />}
                    Export PDF Report
                  </button>
                </div>

                <div className="p-8 grid grid-cols-1 md:grid-cols-2 gap-10">
                  <div className="space-y-8">
                    <div className={`p-6 rounded-2xl border ${getStatusColor(result.prediction)}`}>
                      <p className="text-[10px] font-black uppercase tracking-[0.2em] mb-2 opacity-70">Primary Finding</p>
                      <div className="flex items-center gap-3 text-4xl font-black uppercase tracking-tighter">
                        {result.prediction === 'notumor' ? <CheckCircle size={36} /> : <ShieldAlert size={36} />}
                        {result.prediction}
                      </div>
                    </div>

                    <div className="space-y-4">
                      <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] flex items-center gap-2">
                        <BarChart3 size={14} /> Probability Distribution
                      </h3>
                      <div className="space-y-5">
                        {Object.entries(result.probabilities).map(([label, value]: [string, any]) => (
                          <div key={label} className="group">
                            <div className="flex justify-between text-[11px] font-black uppercase mb-1.5 tracking-tight">
                              <span className="text-slate-600">{label}</span>
                              <span className="text-blue-600">{value}</span>
                            </div>
                            <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden border border-slate-200/50">
                              <div className="bg-blue-600 h-full transition-all duration-1000 ease-out" style={{ width: value }} />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="space-y-6 text-center">
                    <div className="relative">
                      <p className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-4 text-left flex items-center gap-2">
                         Heatmap Localization
                      </p>
                      <div className="rounded-[2.5rem] overflow-hidden border-[12px] border-slate-50 shadow-2xl bg-slate-900 group">
                        <img 
                          src={result.heatmap_url.startsWith('http') ? result.heatmap_url : `http://127.0.0.1:8000/${result.heatmap_url}`} 
                          alt="Localization" 
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700" 
                        />
                      </div>
                    </div>
                    <div className="bg-amber-50 border border-amber-100 p-4 rounded-2xl flex gap-3 text-left">
                      <Info className="text-amber-500 shrink-0" size={18} />
                      <p className="text-[11px] text-amber-700 font-medium leading-relaxed">
                        <b>Clinical Note:</b> The heatmap highlights regions of interest that influenced the AI decision. High intensity (red) indicates suspicious tissue density.
                      </p>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-20">
                <div className="w-24 h-24 bg-slate-50 rounded-full flex items-center justify-center mb-6 border border-slate-100">
                  <Brain size={48} strokeWidth={1.5} className="text-slate-300" />
                </div>
                <h3 className="text-xl font-black text-slate-800">Neural Engine Standby</h3>
                <p className="text-slate-400 max-w-xs mt-2 text-sm">
                  Please upload a high-resolution MRI scan to begin the automated diagnostic sequence.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}