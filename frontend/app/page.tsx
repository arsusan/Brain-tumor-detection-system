"use client";

import { useState, useEffect } from 'react';
import api from './lib/api'; 
import { 
  Upload, Brain, ShieldAlert, CheckCircle, 
  RefreshCcw, Download, Info,
  PieChart, RotateCcw, Zap, X, User, History
} from 'lucide-react';
import jsPDF from 'jspdf';

const BACKEND_URL = "http://127.0.0.1:8000";

interface PredictionResult {
  prediction: string;
  probabilities: Record<string, string>;
  heatmap_url: string;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [patientName, setPatientName] = useState<string>(""); 
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    return () => { if (preview) URL.revokeObjectURL(preview); };
  }, [preview]);

  // FEATURE: Reset Analysis / Analyze Again
  const resetAnalysis = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setPatientName("");
  };

  const generateImmediateReport = async () => {
    if (!result || !file || !patientName) {
      alert("Missing patient data or analysis result.");
      return;
    }
    
    try {
      const doc = new jsPDF('p', 'mm', 'a4');
      doc.setFillColor(30, 41, 59); 
      doc.rect(0, 0, 210, 40, 'F');
      doc.setTextColor(255, 255, 255);
      doc.setFontSize(22);
      doc.setFont("helvetica", "bold");
      doc.text("NEUROSCAN AI DIAGNOSTIC REPORT", 20, 25);
      
      doc.setTextColor(30, 41, 59);
      doc.setFontSize(14);
      doc.text(`Patient Name: ${patientName.toUpperCase()}`, 20, 55);
      
      doc.setFont("helvetica", "normal");
      doc.setFontSize(12);
      doc.text(`Diagnosis: ${result.prediction.toUpperCase()}`, 20, 65);
      doc.text(`Date of Analysis: ${new Date().toLocaleString()}`, 20, 73);
      doc.text(`Reference Image: ${file.name}`, 20, 81);

      const heatmapUrl = `${BACKEND_URL}/${result.heatmap_url}?t=${Date.now()}`;
      const img = new Image();
      img.crossOrigin = "anonymous"; 
      img.src = heatmapUrl;
      
      img.onload = () => {
        doc.addImage(img, 'PNG', 45, 95, 120, 120);
        doc.setFontSize(10);
        doc.setTextColor(100, 116, 139);
        doc.text("This neural activation map highlights regions used for categorization.", 105, 225, { align: 'center' });
        
        const sanitizedPatientName = patientName.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_]/g, '');
        const sanitizedFileName = file.name.split('.')[0];
        doc.save(`${sanitizedPatientName}_${sanitizedFileName}_Report.pdf`);
      };
      
    } catch (err) {
      console.error("PDF Generation Error:", err);
    }
  };

  /**
   * UPDATED FEATURE: Sending user_name to Backend
   * This aligns with the new relational database schema (User -> Scans)
   */
  const handleUpload = async () => {
    if (!file || !patientName) {
        alert("Please enter a Patient Name before running the analysis.");
        return;
    }
    setLoading(true);
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_name', patientName); // Link to DFD Level 0 'User' Entity

    try {
      // POST request to the updated FastAPI endpoint
      const response = await api.post('/predict', formData);
      setResult(response.data);
    } catch (error) {
      alert("Analysis Failed. Check backend connection.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      <nav className="bg-white border-b p-4 sticky top-0 z-50 shadow-sm">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 p-2 rounded-lg"><Brain className="text-white" size={24} /></div>
            <span className="font-bold text-2xl text-slate-800 tracking-tight">NeuroScan <span className="text-blue-600">AI</span></span>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* LEFT: INPUT PANEL */}
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-white p-6 rounded-[2rem] shadow-sm border border-slate-200">
            <h2 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-6 flex items-center gap-2">
               <History size={14}/> Case Registration
            </h2>

            <div className="mb-6">
              <label className="text-[10px] font-bold text-slate-500 uppercase ml-1 mb-2 block">Patient Full Name</label>
              <div className="relative group">
                <User className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-blue-500 transition-colors" size={18} />
                <input 
                  type="text"
                  placeholder="e.g. John Doe"
                  value={patientName}
                  onChange={(e) => setPatientName(e.target.value)}
                  disabled={!!result || loading}
                  className="w-full pl-12 pr-4 py-4 bg-slate-50 border border-slate-200 rounded-2xl focus:ring-4 focus:ring-blue-500/10 focus:border-blue-500 outline-none transition-all font-bold text-slate-700"
                />
              </div>
            </div>

            <div className="space-y-4">
               <label className="text-[10px] font-bold text-slate-500 uppercase ml-1 block">MRI Source File</label>
               {!preview ? (
                  <div 
                    className={`border-2 border-dashed rounded-3xl p-10 text-center transition-colors cursor-pointer relative ${dragActive ? 'border-blue-500 bg-blue-50' : 'border-slate-200 hover:bg-slate-50'}`}
                    onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                    onDragLeave={() => setDragActive(false)}
                    onDrop={(e) => {
                        e.preventDefault(); setDragActive(false);
                        const f = e.dataTransfer.files[0];
                        if(f) { setFile(f); setPreview(URL.createObjectURL(f)); }
                    }}
                  >
                    <Upload className="mx-auto mb-2 text-slate-300" size={32} />
                    <p className="text-xs font-bold text-slate-500">Drop MRI or Click to Browse</p>
                    <input type="file" className="absolute inset-0 opacity-0 cursor-pointer" onChange={(e) => {
                      const f = e.target.files?.[0];
                      if(f) { setFile(f); setPreview(URL.createObjectURL(f)); }
                    }} />
                  </div>
               ) : (
                  <div className="relative rounded-2xl overflow-hidden border-4 border-white shadow-lg">
                    <img src={preview} className="w-full h-48 object-cover" alt="Preview" />
                    {!result && <button onClick={() => {setFile(null); setPreview(null);}} className="absolute top-2 right-2 bg-white/90 backdrop-blur p-2 rounded-full text-rose-500 shadow-md"><X size={16}/></button>}
                  </div>
               )}
            </div>

            <button 
              onClick={handleUpload}
              disabled={!file || !patientName || loading || !!result}
              className="w-full mt-8 bg-slate-900 text-white py-5 rounded-2xl font-black flex justify-center items-center gap-3 disabled:bg-slate-100 disabled:text-slate-400 shadow-2xl transition-all active:scale-95"
            >
              {loading ? <RefreshCcw className="animate-spin" size={20} /> : <Zap size={20} className="text-blue-400" />}
              {result ? "REPORT READY" : "GENERATE AI ANALYSIS"}
            </button>
            
            {result && (
              <button 
                onClick={resetAnalysis}
                className="w-full mt-3 border-2 border-slate-100 text-slate-500 py-4 rounded-2xl font-bold flex justify-center items-center gap-2 hover:bg-slate-50 transition-all"
              >
                <RotateCcw size={16} /> ANALYZE AGAIN
              </button>
            )}
          </div>
        </div>

        {/* RIGHT: RESULTS PANEL */}
        <div className="lg:col-span-8">
          <div className="bg-white rounded-[2.5rem] shadow-sm border min-h-[600px] overflow-hidden">
            {result ? (
              <div className="p-10 animate-in fade-in zoom-in-95 duration-500">
                <div className="flex justify-between items-start mb-10">
                  <div>
                    <h2 className="text-4xl font-black text-slate-900 tracking-tight">Clinical Findings</h2>
                    <p className="text-blue-600 font-bold mt-1 tracking-wide uppercase">CASE: {patientName}</p>
                  </div>
                  <div className="flex gap-3">
                    <button onClick={resetAnalysis} className="bg-slate-100 text-slate-600 px-6 py-4 rounded-2xl font-bold flex items-center gap-2 hover:bg-slate-200 transition-all">
                      <RotateCcw size={18} /> NEW SCAN
                    </button>
                    <button onClick={generateImmediateReport} className="bg-blue-600 text-white px-8 py-4 rounded-2xl font-black flex items-center gap-3 hover:bg-blue-700 shadow-lg shadow-blue-200 transition-all">
                      <Download size={20} /> SAVE PDF
                    </button>
                  </div>
                </div>

                <div className="grid md:grid-cols-2 gap-12">
                   <div className="space-y-8">
                      <div className={`p-8 rounded-[2rem] border-4 ${result.prediction === 'notumor' ? 'bg-emerald-50 border-emerald-100 text-emerald-900' : 'bg-rose-50 border-rose-100 text-rose-900'}`}>
                        <p className="text-[10px] font-black uppercase tracking-[0.2em] mb-4 opacity-50">Result</p>
                        <div className="flex items-center gap-4 text-4xl font-black">
                          {result.prediction === 'notumor' ? <CheckCircle size={40} /> : <ShieldAlert size={40} />}
                          {result.prediction.toUpperCase()}
                        </div>
                      </div>

                      <div className="bg-slate-50 p-8 rounded-[2rem]">
                         <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-6 flex items-center gap-2"><PieChart size={16}/> AI Confidence</h3>
                         {Object.entries(result.probabilities).map(([label, val]) => (
                            <div key={label} className="mb-4">
                               <div className="flex justify-between text-xs font-black mb-1 uppercase tracking-tighter">
                                 <span>{label}</span>
                                 <span>{val}</span>
                               </div>
                               <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                                 <div 
                                    className={`h-full transition-all duration-1000 ${result.prediction === label ? 'bg-blue-600' : 'bg-slate-400'}`} 
                                    style={{width: val}}
                                 ></div>
                               </div>
                            </div>
                         ))}
                      </div>
                   </div>

                   <div className="space-y-6">
                      <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest text-center">Neural Heatmap (Grad-CAM)</p>
                      <div className="bg-slate-900 p-2 rounded-[2.5rem] shadow-2xl aspect-square flex items-center justify-center border-8 border-white">
                        <img 
                          src={`${result.heatmap_url}${result.heatmap_url.includes('?') ? '&' : '?'}t=${Date.now()}`} 
                          className="w-full h-full object-contain rounded-2xl" 
                          alt="Heatmap" 
                        />
                      </div>
                      <div className="flex items-center gap-3 p-4 bg-blue-50 rounded-2xl border border-blue-100">
                        <Info size={20} className="text-blue-500" />
                        <p className="text-[10px] font-medium text-blue-800 leading-relaxed">Activation zones highlight morphological features used by the model.</p>
                      </div>
                   </div>
                </div>
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center p-20 text-slate-200">
                <Brain size={120} className="mb-8 opacity-20" />
                <p className="text-xl font-black text-slate-300 italic tracking-tight uppercase">Awaiting Analysis</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}