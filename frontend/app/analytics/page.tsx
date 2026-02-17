"use client";

import React from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  AreaChart, Area
} from 'recharts';
import { Target, Activity, Brain, ShieldCheck, Info, AlertTriangle, CheckCircle } from 'lucide-react';

export default function AnalyticsPage() {
  // Colors mapped to the 4 classes
  const COLORS = ['#3b82f6', '#10b981', '#6366f1', '#f43f5e'];

  // Data synchronized with evaluation_report.txt 
  const classData = [
    { name: 'Pituitary', recall: 0.97, precision: 0.90, f1: 0.94 },
    { name: 'No Tumor', recall: 0.91, precision: 0.93, f1: 0.92 },
    { name: 'Meningioma', recall: 0.84, precision: 0.72, f1: 0.77 },
    { name: 'Glioma', recall: 0.70, precision: 0.91, f1: 0.79 }
  ];

  return (
    <div className="p-10 bg-[#f8fafc] min-h-screen space-y-8">
      <header className="border-b border-slate-200 pb-6">
        <h1 className="text-3xl font-black text-slate-900 tracking-tight italic uppercase">Clinical Model Validation</h1>
        <p className="text-slate-500 font-medium">Model Version: CNN-2026-02-18 | Dataset N=656</p>
      </header>

      {/* 1. KEY PERFORMANCE INDICATORS  */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {[
          { 
            label: 'Total Accuracy', 
            val: '86%', 
            icon: Target, 
            color: 'text-blue-600', 
            def: 'Overall correct diagnoses across 656 test samples.' 
          },
          { 
            label: 'Precision (Avg)', 
            val: '0.86', 
            icon: Activity, 
            color: 'text-emerald-600', 
            def: 'Reliability: The weighted average precision across all categories.' 
          },
          { 
            label: 'Recall (Avg)', 
            val: '0.86', 
            icon: Brain, 
            color: 'text-violet-600', 
            def: 'Sensitivity: The weighted average recall across all categories.' 
          },
          { 
            label: 'F1-Score (Avg)', 
            val: '0.86', 
            icon: ShieldCheck, 
            color: 'text-rose-600', 
            def: 'The harmonic mean of Precision and Recall.' 
          },
        ].map((m, i) => (
          <div key={i} className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm group relative">
            <div className="flex justify-between mb-4">
              <m.icon className={m.color} size={24} />
              <Info size={16} className="text-slate-300 cursor-help" />
              <div className="opacity-0 group-hover:opacity-100 transition-opacity bg-slate-800 text-white text-[10px] p-2 rounded-lg absolute -top-12 left-0 w-full z-10 shadow-xl pointer-events-none">
                {m.def}
              </div>
            </div>
            <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">{m.label}</p>
            <p className="text-3xl font-black text-slate-900">{m.val}</p>
          </div>
        ))}
      </div>

      {/* 2. CLINICAL STRENGTHS & CRITICAL FINDINGS */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white p-8 rounded-[2.5rem] border border-slate-200 shadow-sm">
          <h3 className="text-sm font-black text-slate-400 uppercase tracking-widest mb-6 flex items-center gap-2">
            <CheckCircle size={18} className="text-emerald-500" /> Class-Specific Sensitivity (Recall)
          </h3>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={classData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} />
                <YAxis domain={[0, 1]} axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} />
                <Tooltip 
                  cursor={{fill: '#f8fafc'}}
                  contentStyle={{borderRadius: '16px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)'}}
                />
                <Bar dataKey="recall" radius={[10, 10, 0, 0]} barSize={60}>
                  { classData.map((entry, index) => <Cell key={index} fill={COLORS[index]} />) }
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-4 text-xs text-slate-500 italic text-center">
            Interpretation: The model is most sensitive to Pituitary tumors (97%) and Healthy tissue (91%).
          </p>
        </div>

        <div className="bg-slate-900 text-white p-8 rounded-[2.5rem] flex flex-col justify-center relative overflow-hidden">
          <div className="absolute top-0 right-0 p-8 opacity-10">
            <Brain size={120} />
          </div>
          <AlertTriangle className="text-amber-400 mb-4" size={32} />
          <h3 className="text-lg font-bold mb-2 text-white">Critical Confusion Insight</h3>
          <p className="text-sm text-slate-400 leading-relaxed">
            Evaluation identifies <b>Glioma (70% recall)</b> as the primary clinical challenge[cite: 6]. 
            The Confusion Matrix reveals <b>45 Gliomas</b> were misclassified as Meningiomas. 
            Conversely, the model is highly reliable for Pituitary tumors, missing only 4 cases out of 149[cite: 7].
          </p>
          <div className="mt-6 p-4 bg-slate-800 rounded-2xl border border-slate-700">
             <p className="text-[10px] font-bold text-amber-400 uppercase mb-1">Recommendation</p>
             <p className="text-xs text-slate-300">Implement multi-view T1/T2 weighted analysis to improve Glioma differentiation.</p>
          </div>
        </div>
      </div>

      {/* 3. VISUAL EVIDENCE (Conf. Matrix & History) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="space-y-4">
            <div className="flex items-center justify-between px-2">
                <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Statistical Convergence</p>
                <span className="text-[10px] bg-blue-100 text-blue-600 px-2 py-1 rounded-full font-bold">Training History</span>
            </div>
            <img src="/plots/training_history.png" alt="Training History" className="rounded-[2rem] border border-slate-200 shadow-lg w-full transition-transform hover:scale-[1.01]" />
            <p className="text-[10px] text-slate-400 italic px-2">History indicates high validation volatility; suggesting potential for further learning rate optimization.</p>
        </div>
        <div className="space-y-4">
            <div className="flex items-center justify-between px-2">
                <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Diagnostic Confusion Map</p>
                <span className="text-[10px] bg-rose-100 text-rose-600 px-2 py-1 rounded-full font-bold">N=656 Matrix</span>
            </div>
            <img src="/plots/final_confusion_matrix.png" alt="Confusion Matrix" className="rounded-[2rem] border border-slate-200 shadow-lg w-full transition-transform hover:scale-[1.01]" />
            <p className="text-[10px] text-slate-400 italic px-2">Diagonal intensity confirms 174 correct "No Tumor" identifications and 145 Pituitary successes.</p>
        </div>
      </div>
    </div>
  );
}