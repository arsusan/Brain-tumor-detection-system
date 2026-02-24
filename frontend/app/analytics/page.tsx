"use client";

import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { Target, Activity, Brain, ShieldCheck, Info, AlertTriangle, CheckCircle } from 'lucide-react';

// Centralized asset management for your Final Year Project
const CLOUDINARY_ASSETS = {
  trainingHistory: "https://res.cloudinary.com/djxhxejwr/image/upload/v1771883852/training_history_az8vjj.png",
  confusionMatrix: "https://res.cloudinary.com/djxhxejwr/image/upload/v1771883851/sample_test_results_bkc3ea.png",
};

export default function AnalyticsPage() {
  const COLORS = ['#3b82f6', '#10b981', '#6366f1', '#f43f5e'];

  const classData = [
    { name: 'Pituitary', recall: 0.97, precision: 0.90, f1: 0.94 },
    { name: 'No Tumor', recall: 0.91, precision: 0.93, f1: 0.92 },
    { name: 'Meningioma', recall: 0.84, precision: 0.72, f1: 0.77 },
    { name: 'Glioma', recall: 0.70, precision: 0.91, f1: 0.79 }
  ];

  return (
    <div className="p-4 md:p-10 bg-[#f8fafc] min-h-screen space-y-6 md:space-y-8">
      {/* Header */}
      <header className="border-b border-slate-200 pb-4 md:pb-6">
        <h1 className="text-2xl md:text-3xl font-black text-slate-900 tracking-tight italic uppercase">
          Clinical Model Validation
        </h1>
        <p className="text-sm md:text-base text-slate-500 font-medium mt-1">
          Model Version: CNN-2026-02-18 | Dataset N=656
        </p>
      </header>

      {/* 1. KEY PERFORMANCE INDICATORS */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 md:gap-6">
        {[
          { label: 'Total Accuracy', val: '86%', icon: Target, color: 'text-blue-600', def: 'Overall correct diagnoses across 656 test samples.' },
          { label: 'Precision (Avg)', val: '0.86', icon: Activity, color: 'text-emerald-600', def: 'Reliability: The weighted average precision across all categories.' },
          { label: 'Recall (Avg)', val: '0.86', icon: Brain, color: 'text-violet-600', def: 'Sensitivity: The weighted average recall across all categories.' },
          { label: 'F1-Score (Avg)', val: '0.86', icon: ShieldCheck, color: 'text-rose-600', def: 'The harmonic mean of Precision and Recall.' },
        ].map((m, i) => (
          <div
            key={i}
            className="bg-white p-4 md:p-6 rounded-2xl md:rounded-3xl border border-slate-200 shadow-sm group relative"
          >
            <div className="flex justify-between mb-3 md:mb-4">
              <m.icon className={m.color} size={20} />
              <Info size={14} className="text-slate-300 cursor-help" />
              <div className="opacity-0 group-hover:opacity-100 transition-opacity bg-slate-800 text-white text-[10px] p-2 rounded-lg absolute -top-10 left-1/2 -translate-x-1/2 w-40 md:w-full z-10 shadow-xl pointer-events-none text-center">
                {m.def}
              </div>
            </div>
            <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">{m.label}</p>
            <p className="text-2xl md:text-3xl font-black text-slate-900">{m.val}</p>
          </div>
        ))}
      </div>

      {/* 2. CLINICAL STRENGTHS & CRITICAL FINDINGS */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 md:gap-6">
        {/* Bar Chart */}
        <div className="lg:col-span-2 bg-white p-4 md:p-8 rounded-2xl md:rounded-[2.5rem] border border-slate-200 shadow-sm">
          <h3 className="text-xs md:text-sm font-black text-slate-400 uppercase tracking-widest mb-4 md:mb-6 flex items-center gap-2">
            <CheckCircle size={16} className="text-emerald-500" /> Class-Specific Sensitivity (Recall)
          </h3>
          <div className="h-[250px] md:h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={classData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis
                  dataKey="name"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: '#64748b', fontSize: 10 }}
                />
                <YAxis
                  domain={[0, 1]}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: '#64748b', fontSize: 10 }}
                />
                <Tooltip
                  cursor={{ fill: '#f8fafc' }}
                  contentStyle={{
                    borderRadius: '12px',
                    border: 'none',
                    boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="recall" radius={[10, 10, 0, 0]} barSize={40}>
                  {classData.map((entry, index) => (
                    <Cell key={index} fill={COLORS[index]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Critical Insight Card */}
        <div className="bg-slate-900 text-white p-6 md:p-8 rounded-2xl md:rounded-[2.5rem] flex flex-col justify-center relative overflow-hidden">
          <div className="absolute top-0 right-0 p-6 md:p-8 opacity-10">
            <Brain size={100} />
          </div>
          <AlertTriangle className="text-amber-400 mb-3 md:mb-4" size={24} />
          <h3 className="text-base md:text-lg font-bold mb-2 text-white">Critical Confusion Insight</h3>
          <p className="text-xs md:text-sm text-slate-400 leading-relaxed">
            Evaluation identifies <b>Glioma (70% recall)</b> as the primary clinical challenge.
            The Confusion Matrix reveals 45 Gliomas were misclassified as Meningiomas.
          </p>
          <div className="mt-4 md:mt-6 p-3 md:p-4 bg-slate-800 rounded-xl md:rounded-2xl border border-slate-700">
            <p className="text-[10px] font-bold text-amber-400 uppercase mb-1">Recommendation</p>
            <p className="text-xs text-slate-300">
              Implement multi-view T1/T2 weighted analysis to improve Glioma differentiation.
            </p>
          </div>
        </div>
      </div>

      {/* 3. VISUAL EVIDENCE */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 md:gap-8">
        <div className="space-y-3 md:space-y-4">
          <div className="flex items-center justify-between px-2">
            <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Statistical Convergence</p>
            <span className="text-[10px] bg-blue-100 text-blue-600 px-2 py-1 rounded-full font-bold">
              Training History
            </span>
          </div>
          <img
            src={CLOUDINARY_ASSETS.trainingHistory}
            alt="Training History"
            className="rounded-2xl md:rounded-[2rem] border border-slate-200 shadow-lg w-full transition-transform hover:scale-[1.01]"
          />
          <p className="text-[10px] text-slate-400 italic px-2">
            History indicates high validation volatility; fetched from secure Cloudinary storage.
          </p>
        </div>
        <div className="space-y-3 md:space-y-4">
          <div className="flex items-center justify-between px-2">
            <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Diagnostic Confusion Map</p>
            <span className="text-[10px] bg-rose-100 text-rose-600 px-2 py-1 rounded-full font-bold">N=656 Matrix</span>
          </div>
          <img
            src={CLOUDINARY_ASSETS.confusionMatrix}
            alt="Confusion Matrix"
            className="rounded-2xl md:rounded-[2rem] border border-slate-200 shadow-lg w-full transition-transform hover:scale-[1.01]"
          />
          <p className="text-[10px] text-slate-400 italic px-2">
            Diagonal intensity confirms correct identifications; externalized for optimized deployment.
          </p>
        </div>
      </div>
    </div>
  );
}